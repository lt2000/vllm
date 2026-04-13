# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool


def test_block_pool_add_and_remove_segment():
    pool = BlockPool(num_gpu_blocks=4,
                     enable_caching=False,
                     num_segments=1,
                     segment_sizes=[4],
                     num_max_gpu_blocks=8)

    assert pool.get_num_blocks() == 4
    assert pool.get_num_total_gpu_blocks() == 8

    pool.add_segment(1, 2)
    assert pool.get_num_blocks() == 6
    assert pool.get_num_free_blocks() == 5

    pool.remove_segment(1, 2)
    assert pool.get_num_blocks() == 4
