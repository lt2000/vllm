# vLLM-SMCTRL Dynamic KV Cache Context

This document captures the current state of the dynamic KV cache work in
`vllm-smctrl` so a later session can recover context quickly without rereading
the full repository diff.

## Current state

- Repository: `/home/llm/tli/vllm-smctrl`
- Active feature branch: `dynamic-kv-smctrl`
- Original MVP commit: `4ac7cdf`
- Original commit subject: `feat: add dynamic kv cache allocation MVP`
- Current local HEAD during this context refresh: `d968879ac`
- Current working tree includes additional uncommitted follow-up changes after
  the MVP:
  - deferred-commit async KV growth path
  - executor/worker mem-op status reporting
  - online validation and benchmark notes
  - follow-up runtime bugfixes found during online validation
- Remote branch: `origin/dynamic-kv-smctrl`

## What was implemented

This branch adds an MVP of dynamic KV cache allocation to `vllm-smctrl`.

The implemented scope is deliberately narrower than ElasticServe:

- Dynamic KV cache growth and shrink for the vLLM v1 path
- Segment-based KV capacity management
- Scheduler-driven grow/shrink decisions
- Deferred-commit async growth on top of the original MVP
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
- `vllm/v1/engine/core.py`
- `vllm/v1/executor/abstract.py`
- `vllm/v1/executor/multiproc_executor.py`

### Worker integration

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/block_table.py`

### Runtime support

- `multi_memory_manager/central_memory_manager/mem_manager.py`
- `multi_memory_manager/central_memory_manager/requirements.txt`
- `vmm_tensor/setup.py`
- `vmm_tensor/vmm_tensor/__init__.py`
- `vmm_tensor/vmm_tensor_refactor.cpp`
- `docs/superpowers/dynamic-kv-cache-async-sequence-diagrams.md`

### New focused tests

- `tests/v1/core/test_dynamic_block_pool.py`
- `tests/v1/core/test_dynamic_scheduler.py`
- `tests/v1/engine/test_dynamic_kv_async.py`
- `tests/v1/executor/test_dynamic_kv_executor.py`
- `tests/v1/worker/test_block_table.py`

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

### Deferred-commit async growth follow-up

After the original MVP, the growth path was reworked to avoid exposing new KV
segments to the scheduler before worker-side VMM mapping had completed.

The old behavior was:

- scheduler decided to grow
- scheduler immediately called `add_segs()`
- worker mapped the new memory asynchronously later

That design allowed overlap between compute and map, but it also meant the
logical block pool could observe new blocks before the worker-side map was
ready.

The current local design is:

- scheduler only *proposes* growth for the current step
- executor records a `PendingKVGrowth(op_id, seg_delta, seg_size)`
- workers start async VMM map through `seg_manager(op_id, seg_delta, seg_size)`
- workers report `KVMemoryOpStatus(op_id, state, error)` back through the
  normal `execute_model` return path
- executor caches and aggregates worker mem-op status
- `EngineCore` calls `_commit_ready_kv_growth()` before the next
  `scheduler.schedule()`
- scheduler only sees `ready` capacity, never `pending` capacity

This is the current async-growth state machine.

Important consequence:

- `grow` now uses deferred logical commit
- `free` is still only allocator-level async (`batch_free_async`) and is **not**
  managed by an equivalent deferred state machine yet

Current state objects introduced for the follow-up:

- `KVMemoryOpStatus`
- `PendingKVGrowth`
- `WorkerExecutionResult`

Current code touchpoints:

- `vllm/v1/engine/core.py`
- `vllm/v1/executor/abstract.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/outputs.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`

For a fuller design discussion and sequence diagrams, also read:

- `docs/superpowers/dynamic-kv-cache-async-sequence-diagrams.md`

### Worker-side mem-op status reporting

The worker side now tracks current growth status in `GPUModelRunner`:

- after async grow dispatch, status becomes `running` or `done`
- `get_kv_mem_op_status()` refreshes the state by querying
  `VMMTensor.is_batch_memory_op_running()`
- `GPUWorker.execute_model_with_kv_status()` returns:
  - `model_output` for the output rank or aggregated path
  - `kv_mem_op_status` for every worker

The executor then:

- aggregates all worker mem-op statuses
- transitions pending growth to `done` only when all relevant workers report the
  same `op_id` as completed
- hands the completed pending growth to `EngineCore` on the next step through
  `take_ready_kv_growth()`

### Current async design limitation

The current async design should be treated as:

- complete enough for deferred async *growth*
- not yet a full unified async mem-op pipeline

Still missing or intentionally deferred:

- explicit async `free` completion tracking
- multiple inflight growth operations
- partial readiness / per-segment readiness
- richer worker error recovery beyond failing the pending op

The current design is best described as:

- one pending growth op at a time
- worker reports completion through the normal model execution return path
- engine commits that growth just before the next scheduler step

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

### Initial dynamic capacity validation bug found and fixed

During later code review, another startup-time bug was found:

- `num_blocks = init_num_segs * num_blocks_per_seg`
- `max_num_blocks` was still derived from profiling
- if `num_blocks > max_num_blocks`, the code later crashed inside
  `BlockPool` with `IndexError`

Fix:

- validate the relationship in `vllm/v1/core/kv_cache_utils.py`
- raise a clear `ValueError` before `BlockPool` construction

This matters because future sessions may otherwise misdiagnose the problem as a
low-level pool bug rather than an invalid dynamic-KV startup configuration.

### Online no-prefix scheduler bug found and fixed

While validating the async design online with `--no-enable-prefix-caching`,
`EngineCore` crashed with:

- `AttributeError: type object 'BlockTable' has no attribute 'get_num_required_blocks'`

Cause:

- dynamic KV scheduler uses `vllm.v1.worker.block_table.BlockTable` to estimate
  waiting backlog in `_kv_cache_schedule()`
- the v1 `BlockTable` class did not define the static helper
  `get_num_required_blocks`
- the helper existed only in the older v0-ish block table code path

Fix:

- add `BlockTable.get_num_required_blocks()` to
  `vllm/v1/worker/block_table.py`
- add a regression test in `tests/v1/worker/test_block_table.py`

This fix is important for online async-KV validation because the bug is easier
to trigger when prefix caching is disabled and backlog-based growth decisions
become active.

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

### Async-growth follow-up verification

After the deferred-commit async design was implemented, the following focused
tests passed in `vllm-graph-dev`:

- `pytest -q tests/v1/core/test_dynamic_scheduler.py`
- `pytest -q tests/v1/engine/test_dynamic_kv_async.py`
- `pytest -q tests/v1/executor/test_dynamic_kv_executor.py`
- `pytest -q tests/v1/worker/test_block_table.py`

In addition, the larger targeted regression batch passed:

- dynamic scheduler tests
- engine async-growth commit tests
- executor pending/ready growth tests
- block-table helper test
- existing dynamic block-pool / worker-bookkeeping / config tests

One successful regression batch during this session was:

- `17 passed` in `vllm-graph-dev`

### Online benchmark evidence for async-growth overhead

Online validation was rerun using:

- environment: `vllm-graph-dev`
- model: `/mnt/sdb/models/Qwen3-4B`
- GPU: `0`
- backend: `vllm serve` + `vllm bench serve`
- prefix caching disabled to reduce benchmark distortion

Two dynamic configurations were compared:

1. `dynamic-preallocated`
   - `--enable-vmm-dynamic`
   - high `--init-num-segs`
   - intended to avoid runtime grow under the selected workload
2. `dynamic-forced-grow`
   - `--enable-vmm-dynamic`
   - small `--init-num-segs`
   - intended to force runtime grow/commit during the benchmark

The strongest evidence that async growth was actually exercised is:

- `server_prealloc24.log`: zero `Requesting` / `Prepared KV cache growth` /
  `Committing KV cache growth`
- `server_forced.log`: repeated `Requesting`, `Prepared KV cache growth`,
  `Committing KV cache growth`, and `Shrinking KV cache`

Observed service-side counts from one benchmark session:

- `server_prealloc24.log`
  - `Requesting`: `0`
  - `Prepared KV cache growth`: `0`
  - `Committing KV cache growth`: `0`
  - `Shrinking KV cache`: `0`
- `server_forced.log`
  - `Requesting`: `13`
  - `Prepared KV cache growth`: `13`
  - `Committing KV cache growth`: `13`
  - `Shrinking KV cache`: `6`

Representative benchmark comparison:

- Workload W1
  - 8 requests
  - max concurrency 8
  - random input len 768
  - random output len 64
- Workload W2
  - 12 requests
  - max concurrency 8
  - random input len 1024
  - random output len 128

Measured client-side deltas, `forced-grow` relative to `preallocated`:

- W1
  - request throughput: about `-2.25%`
  - mean TTFT: about `-1.76%` (treated as noise, not as a speedup claim)
  - mean TPOT / ITL: about `+3.46%`
- W2
  - request throughput: about `-1.47%`
  - mean TTFT: about `-3.74%` (again treated as noise)
  - mean TPOT / ITL: about `+2.35%`

Interpretation:

- async growth is definitely occurring online
- on the tested single-GPU workloads, the visible overhead of async growth was
  low, roughly in the low-single-digit-percent range
- most observable cost showed up in TPOT / ITL rather than a dramatic TTFT
  regression

These online numbers should be treated as indicative, not final:

- they are single-session measurements
- workload scale was intentionally modest
- stronger conclusions would require repeated runs and variance analysis

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

### Additional online benchmark commands used later

These were useful for comparing preallocated dynamic mode against forced-grow
dynamic mode on a single GPU:

#### Preallocated dynamic serve

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_V1=1 \
vllm serve /mnt/sdb/models/Qwen3-4B \
  --port 18000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --enable-vmm-dynamic \
  --num-blocks-per-seg 64 \
  --init-num-segs 24 \
  --gpu-cache-high-threshold 0.45 \
  --gpu-cache-low-threshold 0.25 \
  --max-gpu-blocks 4096 \
  --mem-manager-client-id 0 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching
```

#### Forced-grow dynamic serve

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_V1=1 \
vllm serve /mnt/sdb/models/Qwen3-4B \
  --port 18000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --enable-vmm-dynamic \
  --num-blocks-per-seg 64 \
  --init-num-segs 1 \
  --gpu-cache-high-threshold 0.45 \
  --gpu-cache-low-threshold 0.25 \
  --max-gpu-blocks 4096 \
  --mem-manager-client-id 0 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching
```

#### Benchmark W1

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm bench serve \
  --backend openai \
  --port 18000 \
  --endpoint /v1/completions \
  --model /mnt/sdb/models/Qwen3-4B \
  --dataset-name random \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 8 \
  --random-input-len 768 \
  --random-output-len 64 \
  --disable-tqdm \
  --temperature 0 \
  --save-result \
  --result-dir /tmp/vllm_bench_async
```

#### Benchmark W2

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm bench serve \
  --backend openai \
  --port 18000 \
  --endpoint /v1/completions \
  --model /mnt/sdb/models/Qwen3-4B \
  --dataset-name random \
  --num-prompts 12 \
  --request-rate inf \
  --max-concurrency 8 \
  --random-input-len 1024 \
  --random-output-len 128 \
  --disable-tqdm \
  --temperature 0 \
  --save-result \
  --result-dir /tmp/vllm_bench_async
```

## Known limitations and caveats

### 1. This is an MVP, not full ElasticServe parity

Still missing on purpose:

- KV sharing between instances
- cross-process VMM handle exchange
- disaggregated prefill/decode serving
- role routing with `model_info`

### 1b. Async growth is ahead of async free

Current local state should be understood as:

- `grow` has a deferred-commit async control path
- `free` still uses allocator-level async only
- there is not yet a symmetric executor/engine completion state machine for
  `free`

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

### 5. A clean `pip install -e .` is still not guaranteed in this environment

During this later session, `vllm-graph-dev` still hit the same general editable
build problem family:

- CMake / CUDA environment issues during full editable install
- pre-existing binary artifacts remained necessary for practical source-tree
  online validation

Future sessions should assume source-tree online validation may still require:

- the `vllm-graph-dev` environment
- already-built `vllm._C` / flash-attn artifacts
- `vmm_tensor` installed in that environment

## Recommended next steps

If continuing this work, the most logical next tasks are:

1. Add temporary worker logs around segment map/unmap during serve
2. Add a direct worker-level integration test for `_seg_manager()`
3. Decide whether async `free` should get the same explicit completion tracking
   as async `grow`
4. Make the dynamic path less dependent on local compiled artifacts and the
   current environment-specific setup
5. Repeat the online benchmark matrix with more repetitions and larger
   workloads to reduce variance
6. Decide whether to keep the current VMM extension minimal or evolve it
   toward a more reusable allocator abstraction
7. Only after the MVP is stable, consider whether to port KV sharing or
   disaggregation features

## Resume checklist for a future session

If you want a future agent to resume quickly, the prompt should ask it to:

1. read this file first
2. confirm current branch, local HEAD, and whether the worktree is dirty
3. use environment `vllm-graph-dev`
4. if testing online, start with GPU `0` unless there is a reason to spread to
   more GPUs
5. read `docs/superpowers/dynamic-kv-cache-async-sequence-diagrams.md` after
   this file if the task is about async growth semantics
6. treat the implementation as an MVP without ElasticServe sharing/disagg logic

Suggested opener for a later session:

```text
Please first read docs/superpowers/dynamic-kv-cache-context.md and restore the
current context for the dynamic KV cache MVP on branch dynamic-kv-smctrl before
making any changes.
```
