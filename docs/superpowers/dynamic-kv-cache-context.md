# vLLM-SMCTRL Dynamic KV Cache Context

This document captures the current state of the dynamic KV cache work in
`vllm-smctrl` so a later session can recover context quickly without rereading
the full repository diff.

## Current state

- Repository: `/home/llm/tli/vllm-smctrl`
- Active feature branch: `dynamic-kv-smctrl`
- Latest implementation commit: `4ac7cdf`
- Commit subject: `feat: add dynamic kv cache allocation MVP`
- Remote branch: `origin/dynamic-kv-smctrl`

## What was implemented

This branch adds an MVP of dynamic KV cache allocation to `vllm-smctrl`.

The implemented scope is deliberately narrower than ElasticServe:

- Dynamic KV cache growth and shrink for the vLLM v1 path
- Segment-based KV capacity management
- Scheduler-driven grow/shrink decisions
- External `mem_manager` admission control using NVML
- A minimal CUDA VMM-backed allocator extension for KV cache segments
- Focused tests for config wiring, segment metadata, scheduler logic, and
  worker bookkeeping

The branch does **not** include the following ElasticServe features:

- KV cache sharing across instances
- exported/imported KV handles between processes
- `model_info % 6` role routing
- prefill/decode disaggregation
- mirror scaling metadata migration
- communication proxy logic

## High-level design

The implementation is split into four layers.

### 1. Config and CLI layer

This layer exposes dynamic KV controls through `CacheConfig`,
`ModelConfig`, and CLI parsing.

Key knobs:

- `enable_vmm_dynamic`
- `num_blocks_per_seg`
- `init_num_segs`
- `gpu_cache_high_threshold`
- `gpu_cache_low_threshold`
- `max_gpu_blocks`
- `mem_manager_client_id`

Purpose:

- define the initial active KV size separately from the maximum possible KV
  size
- route expansion requests to the right `mem_manager`
- give the scheduler explicit high/low watermarks for growth and shrink

### 2. KV metadata and block management layer

This layer introduces segmented KV metadata.

Core ideas:

- initial active KV capacity is `init_num_segs * num_blocks_per_seg`
- total reservable capacity is `max_num_blocks`
- `BlockPool` tracks both active blocks and total reservable blocks
- active blocks are grouped into segments
- only trailing free segments are eligible for release

This layer is the in-memory bookkeeping that makes runtime grow/shrink possible.

### 3. Scheduler and admission-control layer

This layer makes the grow/shrink decision.

Behavior:

- if KV usage exceeds `gpu_cache_high_threshold`, the scheduler calculates how
  many new segments are needed
- before actually growing, it asks `mem_manager` for permission
- if usage drops below `gpu_cache_low_threshold`, it releases trailing empty
  segments
- segment deltas are forwarded to worker processes through `SchedulerOutput`
  and the multiprocess executor

### 4. Worker and allocator layer

This layer turns the segment delta into actual GPU memory operations.

Behavior:

- `GPUModelRunner` computes aligned offsets for each segment
- `vmm_tensor` reserves large virtual address regions up front
- initial KV tensors are created from that reserved region
- growth maps new physical memory into the reserved region
- shrink unmaps and releases the corresponding physical memory

## Key files by subsystem

### Config and CLI

- `vllm/config/cache.py`
- `vllm/config/__init__.py`
- `vllm/engine/arg_utils.py`

### KV metadata and block management

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`

### Scheduler integration

- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/executor/multiproc_executor.py`

### Worker integration

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`

### Runtime support

- `multi_memory_manager/central_memory_manager/mem_manager.py`
- `multi_memory_manager/central_memory_manager/requirements.txt`
- `vmm_tensor/setup.py`
- `vmm_tensor/vmm_tensor/__init__.py`
- `vmm_tensor/vmm_tensor_refactor.cpp`

### New focused tests

- `tests/v1/core/test_dynamic_block_pool.py`
- `tests/v1/core/test_dynamic_scheduler.py`

### Extended existing tests

- `tests/test_config.py`
- `tests/v1/core/test_kv_cache_utils.py`
- `tests/v1/core/test_single_type_kv_cache_manager.py`
- `tests/v1/engine/test_engine_args.py`
- `tests/v1/worker/test_gpu_model_runner.py`

## Important implementation details

### Dynamic KV block sizing

The most important semantic change is:

- `num_blocks`: initial active KV blocks
- `max_num_blocks`: profiled maximum KV blocks available for dynamic use

This is set in `vllm/v1/core/kv_cache_utils.py`.

That separation is what allows the system to boot with a small active KV cache
and expand later.

### Segment-aware block pool

`BlockPool` was extended to:

- track `num_gpu_blocks` and `num_max_gpu_blocks`
- support per-segment free deques
- support `add_segment()` and `remove_segment()`
- allocate from active segments only when segment mode is enabled

### Segmented KV manager

`SegmentedFullAttentionManager` is the new manager for
`SegmentedFullAttentionSpec`.

It is intentionally simple:

- segments are appended at the tail
- only trailing fully free segments are removed
- no in-place segment compaction or block migration is attempted

This keeps the MVP stable and understandable.

### Scheduler grow/shrink policy

Implemented in `vllm/v1/core/sched/scheduler.py`.

Relevant helper methods:

- `_init_mem_manager_queues()`
- `_request_memory_from_mem_manager()`
- `_kv_cache_schedule()`

The scheduler now logs:

- outgoing mem-manager requests
- mem-manager replies
- KV growth actions
- KV shrink actions

These logs were added during end-to-end debugging and are useful for future
validation.

### Worker-side segment bookkeeping

The `GPUModelRunner` has new helper methods for aligned segment layout:

- `align_size()`
- `get_offset_and_aligned_size()`
- `block_align_size()`
- `block_get_offset_and_aligned_size()`
- `get_reserved_size()`
- `get_allocation_size()`
- `seg_manager_impl()`
- `_seg_manager()`

These methods are the bridge between scheduler-level segment deltas and the VMM
allocator’s map/unmap operations.

### VMM allocator packaging

The new `vmm_tensor` package is intentionally small.

It currently exposes:

- `VMMTensor.create_tensors()`
- `VMMTensor.batch_allocate_async()`
- `VMMTensor.batch_free_async()`
- `VMMTensor.is_batch_memory_op_running()`

The extension had two runtime issues during implementation:

1. `vmm_tensor.__init__` initially returned `VMMTensor = None` because `_C`
   failed to load before `torch` had loaded its dependent shared libraries.
   Fix: import `torch` first in `vmm_tensor/__init__.py`.
2. `_C` initially failed with `undefined symbol: cuMemAddressFree`.
   Fix: explicitly link against the CUDA driver library in
   `vmm_tensor/setup.py`.

### mem_manager reply bug found and fixed

During the first service-level test, the engine died with:

- `sysv_ipc.BusyError: No available messages of the specified type`

Cause:

- `_request_memory_from_mem_manager()` sent the request
- immediately called `receive(block=False)`
- if the reply had not arrived yet, scheduler crashed

Fix:

- poll the reply queue in a loop
- sleep briefly on `sysv_ipc.BusyError`
- continue once a reply arrives

This is one of the most important runtime fixes from the validation phase.

## Focused verification that passed

The following checks were rerun successfully in the original repository
directory before finalizing the branch.

### Local focused tests

- `python multi_memory_manager/central_memory_manager/mem_manager.py --help`
- `python -m pytest tests/test_config.py -q -k dynamic_kv_cache_config_validation`
- `python -m pytest tests/v1/engine/test_engine_args.py -q -k "dynamic_kv_flags_from_cli or create_model_config_forwards_mem_manager_client_id"`
- `python -m pytest tests/v1/core/test_dynamic_block_pool.py::test_block_pool_add_and_remove_segment -q`
- `python -m pytest tests/v1/core/test_single_type_kv_cache_manager.py::test_segment_manager_selects_only_trailing_free_segments -q`
- `python -m pytest tests/v1/core/test_kv_cache_utils.py::test_get_kv_cache_config_uniform_type_dynamic_initial_blocks -q`
- `python -m pytest tests/v1/core/test_dynamic_scheduler.py -q`
- `python -m pytest tests/v1/worker/test_gpu_model_runner.py::test_block_get_offset_and_aligned_size_is_monotonic tests/v1/worker/test_gpu_model_runner.py::test_seg_manager_impl_updates_block_accounting -q`

### VMM allocator smoke tests

Validated on GPU 2:

- `VMMTensor.create_tensors()` succeeds
- `VMMTensor.batch_allocate_async()` succeeds
- `VMMTensor.batch_free_async()` succeeds

### End-to-end service validation

Validated with:

- environment: `vllm-graph-dev`
- model: `/mnt/sdb/models/Qwen3-4B`
- GPUs: `2,3`
- `mem_manager` watching devices `2 3`

Observed runtime evidence:

- scheduler logs `Requesting ... KV segment(s) from mem_manager`
- `mem_manager` logs `mem_manager recv: ...`
- `mem_manager` logs `mem_manager reply: {'res': 'yes', ...}`
- scheduler logs `Growing KV cache by ...`
- scheduler logs `Shrinking KV cache by ...`

This confirms that the MVP grow/shrink control flow executed in a real serve
path, not just in unit tests.

## Exact commands used for service validation

### Environment

```bash
source /home/llm/.zshrc
conda activate vllm-graph-dev
```

### Install runtime dependencies used during validation

```bash
python -m pip install pytest pynvml sysv_ipc
```

### Build and install `vmm_tensor`

```bash
cd /home/llm/tli/vllm-smctrl/vmm_tensor
python -m pip install . --no-build-isolation --force-reinstall
```

### Start `mem_manager`

```bash
cd /home/llm/tli/vllm-smctrl
python -u multi_memory_manager/central_memory_manager/mem_manager.py \
  --client-id 0 \
  --devices 2 3
```

### Start the validation server

```bash
cd /home/llm/tli/vllm-smctrl
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=2,3

python -u -m vllm.entrypoints.cli.main serve /mnt/sdb/models/Qwen3-4B \
  --port 18000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.5 \
  --enable-vmm-dynamic \
  --num-blocks-per-seg 128 \
  --init-num-segs 1 \
  --gpu-cache-high-threshold 0.45 \
  --gpu-cache-low-threshold 0.25 \
  --max-gpu-blocks 4096 \
  --mem-manager-client-id 0
```

### Health check

```bash
curl -i -s --max-time 5 http://127.0.0.1:18000/health
```

### Trigger dynamic growth

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python -m vllm.entrypoints.cli.main bench serve \
  --model /mnt/sdb/models/Qwen3-4B \
  --dataset-name random \
  --random-input-len 1792 \
  --random-output-len 512 \
  --request-rate 4 \
  --num-prompts 8 \
  --max-concurrency 4 \
  --port 18000 \
  --endpoint /v1/completions \
  --seed 456
```

## Known limitations and caveats

### 1. This is an MVP, not full ElasticServe parity

Still missing on purpose:

- KV sharing between instances
- cross-process VMM handle exchange
- disaggregated prefill/decode serving
- role routing with `model_info`

### 2. Full repository verification is not clean in this environment

The focused tests above are passing, but full verification is still affected by
unrelated environment constraints:

- some tests require remote Hugging Face access
- editable rebuild of the whole repo still hits the existing
  `torch::nvtoolsext` / `CUDA::nvToolsExt` CMake issue

### 3. Running from source in this machine relied on local compiled artifacts

For source-tree testing in this environment, the repository needed access to
already-built `vllm._C` and `vllm_flash_attn` artifacts from the
`vllm-graph-dev` environment.

If a future session sees:

- `ModuleNotFoundError: No module named 'vllm._C'`
- `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'`

then either:

- rebuild/install the full project successfully, or
- temporarily relink those existing artifacts from the environment as was done
  during this session

### 4. End-to-end validation confirmed scheduler grow/shrink control flow

The strongest runtime evidence currently comes from:

- scheduler logs
- mem-manager logs
- explicit VMM allocator smoke tests

If a future session wants stronger proof that worker-side map/unmap is firing
inside serve, the next step is to add temporary worker-side logs around
`seg_manager_impl()` and `batch_allocate_async()` / `batch_free_async()`.

## Recommended next steps

If continuing this work, the most logical next tasks are:

1. Add temporary worker logs around segment map/unmap during serve
2. Add a direct worker-level integration test for `_seg_manager()`
3. Make the dynamic path less dependent on local symlinked compiled artifacts
4. Decide whether to keep the current VMM extension minimal or evolve it
   toward a more reusable allocator abstraction
5. Only after the MVP is stable, consider whether to port KV sharing or
   disaggregation features

## Resume checklist for a future session

If you want a future agent to resume quickly, the prompt should ask it to:

1. read this file first
2. confirm current branch and commit
3. use environment `vllm-graph-dev`
4. use GPUs `2,3,4,5` for testing
5. treat the implementation as an MVP without ElasticServe sharing/disagg logic

Suggested opener for a later session:

```text
Please first read docs/superpowers/dynamic-kv-cache-context.md and restore the
current context for the dynamic KV cache MVP on branch dynamic-kv-smctrl before
making any changes.
```
