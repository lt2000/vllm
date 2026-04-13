# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.block_table import BlockTable


def test_get_num_required_blocks_accounts_for_lookahead_slots():
    required_blocks = BlockTable.get_num_required_blocks([1, 2, 3],
                                                         block_size=4,
                                                         num_lookahead_slots=2)

    assert required_blocks == 2
