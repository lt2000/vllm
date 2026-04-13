# How To Sync `vllm-smctrl` To `vllm-graph-dev`

This document explains how to make code changes under
`/home/llm/tli/vllm-smctrl` take effect inside the Conda environment
`vllm-graph-dev`.

It is written for the current machine state, not for a generic fresh machine.

## Current environment linkage

At the time this document was written, `vllm-graph-dev` is effectively using
the repository source tree directly:

- `import vllm` resolves to:
  - `/home/llm/tli/vllm-smctrl/vllm/__init__.py`
- `import vllm._C` resolves to:
  - `/home/llm/tli/vllm-smctrl/vllm/_C.abi3.so`
- `vmm_tensor` is installed into the environment site-packages

That means:

- most **Python-only** changes under `vllm-smctrl` are picked up immediately
- `vmm_tensor` changes are **not** picked up automatically and must be
  reinstalled
- changes to compiled native `vllm` extensions are **not** guaranteed to be
  picked up unless the corresponding `.so` files in the repo are rebuilt

## Quick rule of thumb

Use this decision table after editing files:

| What you changed | Need reinstall? | Need restart running server/process? | Recommended action |
|---|---|---|---|
| Pure Python in `vllm/**/*.py` | No | Yes | Restart the Python process using vLLM |
| Tests / docs only | No | No | Just rerun tests or continue |
| `vmm_tensor/**/*` | Yes | Yes | Reinstall `vmm_tensor` in `vllm-graph-dev` |
| `csrc/**/*`, `setup.py`, native extension code affecting `vllm._C` | Usually yes | Yes | Rebuild native extensions or re-establish working binaries |

## 0. Activate the environment

```bash
conda activate vllm-graph-dev
```

## 1. Verify the current linkage

Before doing any sync work, confirm what the environment is importing:

```bash
python -c "import vllm, importlib.util; print('vllm=', vllm.__file__); print('vllm._C=', importlib.util.find_spec('vllm._C').origin)"
python -c "import vmm_tensor; print('vmm_tensor=', vmm_tensor.__file__)"
python -m pip show vllm vmm_tensor
```

Expected on this machine:

- `vllm` points at `/home/llm/tli/vllm-smctrl/...`
- `vllm._C` points at `/home/llm/tli/vllm-smctrl/vllm/_C.abi3.so`
- `vmm_tensor` points at environment site-packages

If `vllm` does **not** point at the repo, stop and fix the environment linkage
before assuming your source edits are active.

## 2. If you changed only Python files under `vllm/`

Examples:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/worker/gpu_worker.py`

Then usually you do **not** need to reinstall anything.

Why:

- the environment imports `vllm` directly from the repo
- Python source changes become visible on the next interpreter startup

What to do:

1. Save your files.
2. Stop any running `vllm serve`, Python REPL, or benchmark process that still
   has the old code loaded.
3. Start the process again.

Typical restart workflow:

```bash
pkill -f 'vllm serve /mnt/sdb/models/Qwen3-4B' || true
pkill -f 'mem_manager.py --client-id 0' || true
```

Then relaunch the service / test / benchmark.

### Recommended verification for Python-only changes

Run at least one targeted test or import:

```bash
pytest -q tests/v1/core/test_dynamic_scheduler.py
```

or

```bash
python -c "from vllm.v1.executor.multiproc_executor import MultiprocExecutor; print('import ok')"
```

## 3. If you changed `vmm_tensor`

Examples:

- `vmm_tensor/vmm_tensor_refactor.cpp`
- `vmm_tensor/setup.py`
- `vmm_tensor/vmm_tensor/__init__.py`

These changes are **not** picked up automatically because `vmm_tensor` is
installed into the environment as a separate package.

### Required sync step

```bash
cd /home/llm/tli/vllm-smctrl/vmm_tensor
CUDA_HOME=/usr/local/cuda python -m pip install . --no-build-isolation --force-reinstall
```

### Verify

```bash
python -c "import vmm_tensor; print(vmm_tensor.__file__)"
python -c "from vmm_tensor import VMMTensor; print(VMMTensor)"
```

### After reinstall

Restart any running service / benchmark / test process that uses `vmm_tensor`.

## 4. If you changed native `vllm` extensions

Examples:

- files under `csrc/`
- native extension build wiring in `setup.py`
- anything that requires a new `vllm/_C.abi3.so`, `_moe_C.abi3.so`,
  `_flashmla_C.abi3.so`, or flash-attn extension rebuild

This is the hardest case on this machine.

### Important caveat

A clean `pip install -e .` has **not** been fully reliable here because the
 editable build may still hit CMake / CUDA / NVTX environment issues.

In particular, a full editable rebuild can still fail with the known
`CUDA::nvToolsExt` / `torch::nvtoolsext` CMake problem.

### Recommended path

Try the straightforward rebuild first:

```bash
cd /home/llm/tli/vllm-smctrl
python -m pip install -e . --no-build-isolation --no-deps -v
```

Then verify:

```bash
python -c "import vllm, importlib.util; print(vllm.__file__); print(importlib.util.find_spec('vllm._C').origin)"
```

If this succeeds, restart your service / benchmark / test process.

### If editable rebuild fails

Do **not** assume your native changes are active.

At that point you have three realistic options:

1. Fix the CMake / CUDA environment and rebuild properly.
2. Restore a working set of `.so` binaries and continue only with Python-level
   testing.
3. Move the experiment to a machine where full native rebuilds are clean.

For this machine specifically, the repo is currently usable because the working
`vllm._C` and related `.so` files already exist under:

- `/home/llm/tli/vllm-smctrl/vllm/`

If those `.so` files disappear or become stale, Python-level changes may still
load, but native behavior will no longer match your source changes.

## 5. If you changed tests or docs only

Examples:

- `tests/**/*`
- `docs/**/*`

No environment sync is required.

Just rerun the relevant test or reopen the document.

## 6. Minimal sync workflows by scenario

### Scenario A: You changed scheduler / engine / worker Python only

```bash
conda activate vllm-graph-dev
cd /home/llm/tli/vllm-smctrl

python -c "import vllm; print(vllm.__file__)"
pytest -q tests/v1/core/test_dynamic_scheduler.py
```

Then restart the service if you need online validation.

### Scenario B: You changed `vmm_tensor`

```bash
conda activate vllm-graph-dev
cd /home/llm/tli/vllm-smctrl/vmm_tensor

CUDA_HOME=/usr/local/cuda python -m pip install . --no-build-isolation --force-reinstall
python -c "from vmm_tensor import VMMTensor; print(VMMTensor)"
```

Then restart the service.

### Scenario C: You changed native `vllm` C++ / CUDA

```bash
conda activate vllm-graph-dev
cd /home/llm/tli/vllm-smctrl

python -m pip install -e . --no-build-isolation --no-deps -v
python -c "import vllm, importlib.util; print(vllm.__file__); print(importlib.util.find_spec('vllm._C').origin)"
```

If this fails, your native changes are not synced yet.

## 7. Recommended post-sync checks

After any meaningful change, run a check appropriate to what changed.

### Dynamic KV scheduler / engine changes

```bash
pytest -q \
  tests/v1/core/test_dynamic_scheduler.py \
  tests/v1/engine/test_dynamic_kv_async.py \
  tests/v1/executor/test_dynamic_kv_executor.py
```

### Block table or waiting-backlog related changes

```bash
pytest -q tests/v1/worker/test_block_table.py
```

### Config validation changes

```bash
pytest -q \
  tests/test_config.py::test_dynamic_kv_cache_config_validation \
  tests/v1/core/test_kv_cache_utils.py::test_get_kv_cache_config_uniform_type_dynamic_rejects_initial_blocks_above_profiled_max
```

### Online smoke check

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
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
  --max-num-batched-tokens 8192
```

Then:

```bash
curl -sSf http://127.0.0.1:18000/health
```

## 8. Common failure modes

### `ModuleNotFoundError: No module named 'vllm._C'`

Meaning:

- Python is seeing the repo source tree
- but the native extension binary is missing or invalid

Action:

- rebuild / reinstall native `vllm`, or restore a working `_C.abi3.so`

### `ModuleNotFoundError: No module named 'vmm_tensor'`

Meaning:

- the environment package is missing

Action:

```bash
cd /home/llm/tli/vllm-smctrl/vmm_tensor
CUDA_HOME=/usr/local/cuda python -m pip install . --no-build-isolation --force-reinstall
```

### Service starts but behavior looks old

Meaning:

- you changed Python code, but an old service process is still running

Action:

- stop the old process completely
- restart `vllm serve`

### `pip install -e .` fails during CMake configuration

Meaning:

- this machine still has the known full-native editable-build issue

Action:

- do not assume native changes were applied
- either fix the build environment or avoid native-extension-dependent testing

## 9. Practical recommendation

For this machine, the most reliable workflow is:

1. Keep Python-level development in the repo source tree.
2. Treat Python-only changes as instant-on after process restart.
3. Reinstall `vmm_tensor` explicitly whenever it changes.
4. Avoid touching native `vllm` extensions unless you are prepared to debug the
   editable build environment.

That is the shortest path to keeping `vllm-smctrl` and `vllm-graph-dev`
effectively in sync.
