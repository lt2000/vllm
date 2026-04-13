# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from vllm import envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert (vllm_config.cache_config.enable_prefix_caching
            ), "V1 turns on prefix caching by default."

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "builtin"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to builtin
    args = parser.parse_args(["--prefix-caching-hash-algo", "builtin"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "builtin"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(
        UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform
    device_name = current_platform.get_device_name().lower()
    if "h100" in device_name or "h200" in device_name:
        # For H100 and H200, we use larger default values.
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(
        UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501


def test_dynamic_kv_flags_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([
        "--enable-vmm-dynamic",
        "--num-blocks-per-seg",
        "128",
        "--init-num-segs",
        "2",
        "--gpu-cache-high-threshold",
        "0.92",
        "--gpu-cache-low-threshold",
        "0.71",
        "--max-gpu-blocks",
        "4096",
        "--mem-manager-client-id",
        "3",
    ])
    engine_args = EngineArgs.from_cli_args(args=args)

    assert engine_args.enable_vmm_dynamic is True
    assert engine_args.num_blocks_per_seg == 128
    assert engine_args.init_num_segs == 2
    assert engine_args.gpu_cache_high_threshold == pytest.approx(0.92)
    assert engine_args.gpu_cache_low_threshold == pytest.approx(0.71)
    assert engine_args.max_gpu_blocks == 4096
    assert engine_args.mem_manager_client_id == 3


def test_create_model_config_forwards_mem_manager_client_id(monkeypatch):
    captured = {}

    class FakeModelConfig:

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vllm.engine.arg_utils.ModelConfig", FakeModelConfig)

    engine_args = EngineArgs(model="facebook/opt-125m",
                             mem_manager_client_id=7)
    engine_args.create_model_config()

    assert captured["mem_manager_client_id"] == 7
