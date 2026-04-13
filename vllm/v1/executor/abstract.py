# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.uniproc_executor import (  # noqa
    ExecutorWithExternalLauncher as ExecutorWithExternalLauncherV0)
from vllm.executor.uniproc_executor import (  # noqa
    UniProcExecutor as UniProcExecutorV0)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (KVMemoryOpStatus, ModelRunnerOutput,
                             PendingKVGrowth, WorkerExecutionResult)

FailureCallback = Callable[[], None]


class Executor(ExecutorBase):
    """
    Abstract class for v1 executors, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in ExecutorBase"""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = (
            parallel_config.distributed_executor_backend)
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("Unknown distributed executor backend: "
                             f"{distributed_executor_backend}")
        return executor_class

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config",
                            args=(kv_cache_configs, ))
        self.collective_rpc("compile_or_warm_up_model")

    def register_failure_callback(self, callback: FailureCallback):
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        output = self.collective_rpc("determine_available_memory")
        return output

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        output = self.collective_rpc("get_kv_cache_spec")
        return output

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        self._ensure_dynamic_kv_state()
        if self.vllm_config.cache_config.enable_vmm_dynamic:
            self._maybe_dispatch_dynamic_kv_growth(
                scheduler_output.num_new_segs, scheduler_output.num_block_per_seg)
            result = self.collective_rpc(
                "execute_model_with_kv_status",
                args=(scheduler_output, 0, False),
            )[0]
            assert isinstance(result, WorkerExecutionResult)
            self._update_pending_kv_growth_status([result.kv_mem_op_status])
            assert result.model_output is not None
            return result.model_output
        output = self.collective_rpc("execute_model",
                                     args=(scheduler_output, ))
        return output[0]

    def _ensure_dynamic_kv_state(self) -> None:
        if not hasattr(self, "_next_kv_mem_op_id"):
            self._next_kv_mem_op_id = 1
        if not hasattr(self, "_pending_kv_growth"):
            self._pending_kv_growth: Optional[PendingKVGrowth] = None
        if not hasattr(self, "_pending_kv_growth_state"):
            self._pending_kv_growth_state = "idle"
        if not hasattr(self, "_pending_kv_growth_error"):
            self._pending_kv_growth_error: Optional[str] = None

    def _maybe_dispatch_dynamic_kv_growth(self, seg_delta: int,
                                          seg_size: int) -> None:
        if seg_delta == 0:
            return
        self._ensure_dynamic_kv_state()
        if seg_delta < 0:
            self.collective_rpc("seg_manager", args=(0, seg_delta, seg_size))
            return
        if self._pending_kv_growth is not None:
            raise RuntimeError("Cannot dispatch dynamic KV growth while a "
                               "previous growth operation is still pending.")
        op_id = self._next_kv_mem_op_id
        self._next_kv_mem_op_id += 1
        self.collective_rpc("seg_manager", args=(op_id, seg_delta, seg_size))
        self._pending_kv_growth = PendingKVGrowth(op_id=op_id,
                                                  seg_delta=seg_delta,
                                                  seg_size=seg_size)
        self._pending_kv_growth_state = "running"
        self._pending_kv_growth_error = None

    def _update_pending_kv_growth_status(
            self, statuses: list[KVMemoryOpStatus]) -> None:
        self._ensure_dynamic_kv_state()
        if self._pending_kv_growth is None:
            return
        expected_op_id = self._pending_kv_growth.op_id
        relevant_statuses = [
            status for status in statuses if status.op_id == expected_op_id
        ]
        if not relevant_statuses:
            return
        if any(status.state == "failed" for status in relevant_statuses):
            self._pending_kv_growth_state = "failed"
            errors = [
                status.error for status in relevant_statuses if status.error
            ]
            self._pending_kv_growth_error = "; ".join(errors) or None
            return
        expected_statuses = getattr(self, "world_size", len(statuses))
        if (len(relevant_statuses) >= expected_statuses
                and all(status.state == "done"
                        for status in relevant_statuses[:expected_statuses])):
            self._pending_kv_growth_state = "done"
            return
        self._pending_kv_growth_state = "running"

    def has_pending_kv_growth(self) -> bool:
        self._ensure_dynamic_kv_state()
        return self._pending_kv_growth is not None

    def take_ready_kv_growth(self) -> Optional[PendingKVGrowth]:
        self._ensure_dynamic_kv_state()
        if self._pending_kv_growth is None:
            return None
        if self._pending_kv_growth_state == "failed":
            raise RuntimeError("Dynamic KV growth failed on at least one "
                               "worker."
                               + (f" Details: {self._pending_kv_growth_error}"
                                  if self._pending_kv_growth_error else ""))
        if self._pending_kv_growth_state != "done":
            return None
        ready_growth = self._pending_kv_growth
        self._pending_kv_growth = None
        self._pending_kv_growth_state = "idle"
        self._pending_kv_growth_error = None
        return ready_growth

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))


class UniProcExecutor(UniProcExecutorV0, Executor):
    pass


class ExecutorWithExternalLauncher(ExecutorWithExternalLauncherV0, Executor):

    def determine_available_memory(self) -> list[int]:  # in bytes
        # same as determine_num_available_blocks in v0,
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
