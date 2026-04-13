# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.core.sched.scheduler import Scheduler


class _FakeMessageQueue:

    def __init__(self, response_payload=None):
        self.sent = []
        self.response_payload = response_payload

    def send(self, payload):
        self.sent.append(payload)

    def receive(self, block=False):
        return self.response_payload, 0


def test_scheduler_requests_segment_growth_when_usage_crosses_high_threshold():
    sched = Scheduler.__new__(Scheduler)
    sched.cache_config = SimpleNamespace(num_blocks_per_seg=128,
                                         block_size=16,
                                         gpu_cache_high_threshold=0.9,
                                         gpu_cache_low_threshold=0.75)
    sched.kv_cache_config = SimpleNamespace(is_adaptive_kv=False)
    sched.kv_cache_manager = SimpleNamespace(
        usage=0.95,
        get_num_blocks=lambda: 256,
        get_num_free_blocks=lambda: 8,
        get_actual_segments=lambda delta: (delta, 128),
        add_segs=Mock(),
        select_free_segs=lambda need_free: ([], 0),
        remove_segs=Mock(),
    )
    sched.waiting = []
    sched._request_memory_from_mem_manager = Mock(return_value=1)

    seg_delta, seg_size = Scheduler._kv_cache_schedule(sched)

    assert seg_delta == 1
    assert seg_size == 128
    sched._request_memory_from_mem_manager.assert_called_once_with(1)
    sched.kv_cache_manager.add_segs.assert_called_once_with(1, 128)


def test_scheduler_frees_only_trailing_empty_segments():
    sched = Scheduler.__new__(Scheduler)
    sched.cache_config = SimpleNamespace(num_blocks_per_seg=128,
                                         block_size=16,
                                         gpu_cache_high_threshold=0.9,
                                         gpu_cache_low_threshold=0.75)
    sched.kv_cache_config = SimpleNamespace(is_adaptive_kv=False)
    sched.kv_cache_manager = SimpleNamespace(
        usage=0.40,
        get_num_blocks=lambda: 512,
        get_num_free_blocks=lambda: 256,
        get_actual_segments=lambda delta: (delta, 128),
        add_segs=Mock(),
        select_free_segs=lambda need_free: ([2, 3], 256),
        remove_segs=Mock(),
    )
    sched.waiting = []
    sched._request_memory_from_mem_manager = Mock(return_value=0)

    seg_delta, seg_size = Scheduler._kv_cache_schedule(sched)

    assert seg_delta == -2
    assert seg_size == 128
    sched.kv_cache_manager.remove_segs.assert_called_once_with(2)


def test_request_memory_from_mem_manager_round_trip():
    sched = Scheduler.__new__(Scheduler)
    sched.mem_manager_client_id = 3
    sched._manager_mq = _FakeMessageQueue()
    sched._reply_mq = _FakeMessageQueue(
        pickle.dumps({"res": "yes", "size": 128, "max_free_mem": 4096}))
    sched.block_size_in_bytes = 1024
    sched.cache_config = SimpleNamespace(num_blocks_per_seg=128)
    sched.parallel_config = SimpleNamespace()
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_layers_by_block_type=lambda parallel_config,
            layer_type: 4))

    seg_delta = Scheduler._request_memory_from_mem_manager(sched, 2)

    assert seg_delta == 2
    sent_request = pickle.loads(sched._manager_mq.sent[0])
    assert sent_request["client_id"] == 3
    assert sent_request["req_mem"] > 0
