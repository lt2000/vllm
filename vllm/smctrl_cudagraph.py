# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import gc
import os
import threading
from pathlib import Path
from typing import Any, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MASK64 = (1 << 64) - 1
_ENABLE_ENV = "VLLM_SMCTRL_CUDAGRAPH_ENABLE"
_MASK_ENV = "VLLM_SMCTRL_CUDAGRAPH_MASK"
_MASK_FILE_ENV = "VLLM_SMCTRL_CUDAGRAPH_MASK_FILE"
_LIBSMCTRL_ENV = "VLLM_SMCTRL_LIBSMCTRL_SO_PATH"

_CAPTURE_MODE_MAP = {
    "global": 0,
    "thread_local": 1,
    "relaxed": 2,
}

_DEFAULT_CAPTURE_STREAM = None
_CUDART = None
_LIBSMCTRL = None
_LOAD_LOCK = threading.Lock()
_TPC_COUNT_CACHE: dict[int, int] = {}
_INITIALIZED_DEVICE_INDEX: Optional[int] = None
_INITIALIZED_TOTAL_TPCS: Optional[int] = None


def smctrl_cudagraph_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def _normalize_mask(mask: int) -> int:
    if mask < 0 or mask > _MASK64:
        raise ValueError(f"SM mask must fit in uint64, got {mask}")
    return mask


def _parse_mask(value: str) -> int:
    value = value.strip()
    if not value:
        return 0
    if value.startswith("~"):
        return (~int(value[1:], 0)) & _MASK64
    return _normalize_mask(int(value, 0))


class _MaskProvider:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime_mask: Optional[int] = None
        self._file_mask: int = 0
        self._file_mtime_ns: Optional[int] = None

    def set_runtime_mask(self, mask: Optional[int]) -> None:
        with self._lock:
            self._runtime_mask = None if mask is None else _normalize_mask(mask)

    def get_mask(self) -> int:
        with self._lock:
            if self._runtime_mask is not None:
                return self._runtime_mask

            mask_file = os.environ.get(_MASK_FILE_ENV, "").strip()
            if mask_file:
                try:
                    stat = os.stat(mask_file)
                except FileNotFoundError:
                    logger.warning_once(
                        "SM-control mask file %s does not exist; falling back to %s.",
                        mask_file,
                        _MASK_ENV,
                    )
                else:
                    if self._file_mtime_ns != stat.st_mtime_ns:
                        content = Path(mask_file).read_text(encoding="utf-8").strip()
                        self._file_mask = _parse_mask(content) if content else 0
                        self._file_mtime_ns = stat.st_mtime_ns
                        logger.info_once(
                            "Using SM-control mask file at %s for CUDA graph replay.",
                            mask_file,
                        )
                    return self._file_mask

            env_mask = os.environ.get(_MASK_ENV, "").strip()
            return _parse_mask(env_mask) if env_mask else 0


_MASK_PROVIDER = _MaskProvider()


def set_smctrl_cudagraph_mask(mask: Optional[int]) -> None:
    _MASK_PROVIDER.set_runtime_mask(mask)


def clear_smctrl_cudagraph_mask() -> None:
    _MASK_PROVIDER.set_runtime_mask(None)


def get_smctrl_cudagraph_mask() -> int:
    return _MASK_PROVIDER.get_mask()


def _load_cdll(candidates: list[str], label: str) -> ctypes.CDLL:
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        f"Unable to load {label}. Tried: " + "; ".join(errors))


def _get_cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is None:
        with _LOAD_LOCK:
            if _CUDART is None:
                candidates = []
                if envs.VLLM_CUDART_SO_PATH:
                    candidates.append(envs.VLLM_CUDART_SO_PATH)
                found = ctypes.util.find_library("cudart")
                if found:
                    candidates.append(found)
                candidates.extend([
                    "/usr/local/cuda/lib64/libcudart.so",
                    "libcudart.so",
                ])
                _CUDART = _load_cdll(candidates, "libcudart")
                _CUDART.cudaStreamBeginCapture.argtypes = [
                    ctypes.c_void_p, ctypes.c_int
                ]
                _CUDART.cudaStreamBeginCapture.restype = ctypes.c_int
                _CUDART.cudaStreamEndCapture.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
                ]
                _CUDART.cudaStreamEndCapture.restype = ctypes.c_int
                _CUDART.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
                _CUDART.cudaGraphDestroy.restype = ctypes.c_int
    return _CUDART


def _get_libsmctrl() -> ctypes.CDLL:
    global _LIBSMCTRL
    if _LIBSMCTRL is None:
        with _LOAD_LOCK:
            if _LIBSMCTRL is None:
                candidates = [
                    os.environ.get(_LIBSMCTRL_ENV, "").strip(),
                    ctypes.util.find_library("smctrl") or "",
                    "/home/llm/tli/libsmctrl/libsmctrl.so",
                ]
                _LIBSMCTRL = _load_cdll(candidates, "libsmctrl")
                _LIBSMCTRL.libsmctrl_graph_create.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_void_p,
                    ctypes.c_ulonglong,
                ]
                _LIBSMCTRL.libsmctrl_graph_create.restype = ctypes.c_int
                _LIBSMCTRL.libsmctrl_graph_launch.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint64,
                ]
                _LIBSMCTRL.libsmctrl_graph_launch.restype = ctypes.c_int
                _LIBSMCTRL.libsmctrl_graph_destroy.argtypes = [ctypes.c_void_p]
                _LIBSMCTRL.libsmctrl_graph_destroy.restype = None
                _LIBSMCTRL.libsmctrl_get_tpc_info_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_int,
                ]
                _LIBSMCTRL.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    return _LIBSMCTRL


def _check_cuda(status: int, where: str) -> None:
    if status != 0:
        raise RuntimeError(f"{where} failed with CUDA error {status}")


def _get_default_capture_stream() -> torch.cuda.Stream:
    global _DEFAULT_CAPTURE_STREAM
    if _DEFAULT_CAPTURE_STREAM is None:
        _DEFAULT_CAPTURE_STREAM = torch.cuda.Stream()
    return _DEFAULT_CAPTURE_STREAM


def _low_bits(count: int) -> int:
    if count <= 0:
        return 0
    if count >= 64:
        return _MASK64
    return (1 << count) - 1


def _get_current_device_tpc_count() -> int:
    global _INITIALIZED_DEVICE_INDEX, _INITIALIZED_TOTAL_TPCS
    if _INITIALIZED_TOTAL_TPCS is not None:
        return _INITIALIZED_TOTAL_TPCS

    device = torch.cuda.current_device()
    cached = _TPC_COUNT_CACHE.get(device)
    if cached is not None:
        _INITIALIZED_DEVICE_INDEX = device
        _INITIALIZED_TOTAL_TPCS = cached
        return cached

    num_tpcs = ctypes.c_uint32()
    err = _get_libsmctrl().libsmctrl_get_tpc_info_cuda(
        ctypes.byref(num_tpcs), device)
    if err != 0:
        raise OSError(err, "libsmctrl_get_tpc_info_cuda failed")

    if num_tpcs.value <= 0:
        raise RuntimeError("GPU reported zero TPCs")
    if num_tpcs.value > 64:
        raise RuntimeError(
            f"SM-control percentage API currently supports up to 64 TPCs, got {num_tpcs.value}"
        )

    _TPC_COUNT_CACHE[device] = int(num_tpcs.value)
    _INITIALIZED_DEVICE_INDEX = device
    _INITIALIZED_TOTAL_TPCS = int(num_tpcs.value)
    return int(num_tpcs.value)


def initialize_smctrl_for_current_device(device_index: Optional[int] = None
                                         ) -> dict[str, Any]:
    global _INITIALIZED_DEVICE_INDEX, _INITIALIZED_TOTAL_TPCS
    if device_index is None:
        device_index = torch.cuda.current_device()
    cached = _TPC_COUNT_CACHE.get(device_index)
    if cached is None:
        num_tpcs = ctypes.c_uint32()
        err = _get_libsmctrl().libsmctrl_get_tpc_info_cuda(
            ctypes.byref(num_tpcs), int(device_index))
        if err != 0:
            raise OSError(err, "libsmctrl_get_tpc_info_cuda failed")
        if num_tpcs.value <= 0:
            raise RuntimeError("GPU reported zero TPCs")
        if num_tpcs.value > 64:
            raise RuntimeError(
                f"SM-control percentage API currently supports up to 64 TPCs, got {num_tpcs.value}"
            )
        cached = int(num_tpcs.value)
        _TPC_COUNT_CACHE[int(device_index)] = cached

    _INITIALIZED_DEVICE_INDEX = int(device_index)
    _INITIALIZED_TOTAL_TPCS = cached
    return {
        "device_index": _INITIALIZED_DEVICE_INDEX,
        "total_tpcs": _INITIALIZED_TOTAL_TPCS,
    }


def _mask_for_enabled_tpcs(total_tpcs: int, enabled_tpcs: int) -> int:
    enabled_tpcs = max(1, min(total_tpcs, enabled_tpcs))
    enabled_mask = _low_bits(enabled_tpcs)
    actual_mask = _low_bits(total_tpcs)
    return actual_mask & (~enabled_mask & _MASK64)


def _build_state(mask: int, requested_percentage: Optional[float],
                 strategy: str = "contiguous_prefix") -> dict[str, Any]:
    total_tpcs = _get_current_device_tpc_count()
    actual_mask = _normalize_mask(mask) & _low_bits(total_tpcs)
    disabled_tpcs = int(actual_mask.bit_count())
    enabled_tpcs = total_tpcs - disabled_tpcs
    effective_percentage = 100.0 * enabled_tpcs / total_tpcs
    return {
        "requested_percentage": requested_percentage,
        "effective_percentage": effective_percentage,
        "total_tpcs": total_tpcs,
        "enabled_tpcs": enabled_tpcs,
        "mask": hex(actual_mask),
        "mask_int": int(actual_mask),
        "strategy": strategy,
        "device_index": int(_INITIALIZED_DEVICE_INDEX
                             if _INITIALIZED_DEVICE_INDEX is not None else
                             torch.cuda.current_device()),
    }


def get_smctrl_cudagraph_state() -> dict[str, Any]:
    return _build_state(get_smctrl_cudagraph_mask(), requested_percentage=None)


def set_smctrl_cudagraph_percentage(percentage: float) -> dict[str, Any]:
    if not smctrl_cudagraph_enabled():
        raise RuntimeError(
            "SM-control CUDA graphs are not enabled. Set "
            f"{_ENABLE_ENV}=1 when starting vLLM.")
    if percentage <= 0 or percentage > 100:
        raise ValueError("percentage must be in the range (0, 100].")

    total_tpcs = _get_current_device_tpc_count()
    enabled_tpcs = int(round(total_tpcs * (percentage / 100.0)))
    enabled_tpcs = max(1, min(total_tpcs, enabled_tpcs))
    mask = _mask_for_enabled_tpcs(total_tpcs, enabled_tpcs)
    set_smctrl_cudagraph_mask(mask)
    state = _build_state(mask, requested_percentage=percentage)
    logger.info(
        "Set SM-control percentage to %.3f%% on cuda:%d -> enabled_tpcs=%d/%d mask=%s",
        percentage,
        state["device_index"],
        state["enabled_tpcs"],
        state["total_tpcs"],
        state["mask"],
    )
    return state


class _SMControlCaptureContext:
    def __init__(self, graph: "SMControlCUDAGraph") -> None:
        self.graph = graph
        self.stream_ctx = None

    def __enter__(self) -> "SMControlCUDAGraph":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.stream_ctx = torch.cuda.stream(self.graph.stream)
        self.stream_ctx.__enter__()
        torch._C._cuda_beginAllocateCurrentStreamToPool(
            torch.cuda.current_device(), self.graph.pool)
        _check_cuda(
            self.graph.cudart.cudaStreamBeginCapture(
                ctypes.c_void_p(self.graph.stream.cuda_stream),
                self.graph.capture_mode,
            ),
            "cudaStreamBeginCapture",
        )
        return self.graph

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is None:
                _check_cuda(
                    self.graph.cudart.cudaStreamEndCapture(
                        ctypes.c_void_p(self.graph.stream.cuda_stream),
                        ctypes.byref(self.graph.graph),
                    ),
                    "cudaStreamEndCapture",
                )
                err = self.graph.libsmctrl.libsmctrl_graph_create(
                    ctypes.byref(self.graph.handle),
                    self.graph.graph,
                    0,
                )
                if err != 0:
                    raise OSError(err, "libsmctrl_graph_create failed")
        finally:
            torch._C._cuda_endAllocateCurrentStreamToPool(
                torch.cuda.current_device(), self.graph.pool)
            self.stream_ctx.__exit__(exc_type, exc_val, exc_tb)


class SMControlCUDAGraph:
    def __init__(self,
                 pool=None,
                 stream: Optional[torch.cuda.Stream] = None,
                 capture_error_mode: str = "global") -> None:
        if capture_error_mode not in _CAPTURE_MODE_MAP:
            raise ValueError(
                f"Unsupported capture error mode: {capture_error_mode}")
        self.cudart = _get_cudart()
        self.libsmctrl = _get_libsmctrl()
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.stream = stream if stream is not None else _get_default_capture_stream()
        self.capture_mode = _CAPTURE_MODE_MAP[capture_error_mode]
        self.graph = ctypes.c_void_p()
        self.handle = ctypes.c_void_p()
        logger.info_once(
            "Using libsmctrl-backed CUDA graph replay for vLLM. "
            "mask_env=%s mask_file_env=%s libsmctrl=%s",
            _MASK_ENV,
            _MASK_FILE_ENV,
            os.environ.get(_LIBSMCTRL_ENV, "/home/llm/tli/libsmctrl/libsmctrl.so"),
        )

    def capture(self) -> _SMControlCaptureContext:
        return _SMControlCaptureContext(self)

    def replay(self) -> None:
        mask = get_smctrl_cudagraph_mask()
        err = self.libsmctrl.libsmctrl_graph_launch(
            self.handle,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ctypes.c_uint64(mask),
        )
        if err != 0:
            raise OSError(err, "libsmctrl_graph_launch failed")

    def reset(self) -> None:
        if self.handle.value:
            self.libsmctrl.libsmctrl_graph_destroy(self.handle)
            self.handle = ctypes.c_void_p()
        if self.graph.value:
            _check_cuda(self.cudart.cudaGraphDestroy(self.graph),
                        "cudaGraphDestroy")
            self.graph = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.reset()
        except Exception:
            pass
