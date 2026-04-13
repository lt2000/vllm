# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import KVMemoryOpStatus, PendingKVGrowth


def test_take_ready_kv_growth_after_all_workers_report_done():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.shutting_down = True
    executor._pending_kv_growth = PendingKVGrowth(op_id=7,
                                                  seg_delta=2,
                                                  seg_size=128)
    executor._pending_kv_growth_state = "running"
    executor._pending_kv_growth_error = None

    MultiprocExecutor._update_pending_kv_growth_status(
        executor,
        [
            KVMemoryOpStatus(op_id=7, state="done"),
            KVMemoryOpStatus(op_id=7, state="done"),
        ],
    )

    ready_growth = executor.take_ready_kv_growth()

    assert ready_growth == PendingKVGrowth(op_id=7, seg_delta=2, seg_size=128)
    assert executor._pending_kv_growth is None


def test_take_ready_kv_growth_waits_while_any_worker_still_running():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.shutting_down = True
    executor._pending_kv_growth = PendingKVGrowth(op_id=9,
                                                  seg_delta=1,
                                                  seg_size=64)
    executor._pending_kv_growth_state = "running"
    executor._pending_kv_growth_error = None

    MultiprocExecutor._update_pending_kv_growth_status(
        executor,
        [
            KVMemoryOpStatus(op_id=9, state="done"),
            KVMemoryOpStatus(op_id=9, state="running"),
        ],
    )

    assert executor.take_ready_kv_growth() is None
    assert executor._pending_kv_growth == PendingKVGrowth(op_id=9,
                                                          seg_delta=1,
                                                          seg_size=64)
