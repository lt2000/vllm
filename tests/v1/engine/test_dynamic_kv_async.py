# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore


def test_engine_core_commits_ready_kv_growth_before_schedule():
    engine_core = EngineCore.__new__(EngineCore)
    ready_growth = SimpleNamespace(seg_delta=2, seg_size=128)
    engine_core.model_executor = SimpleNamespace(
        take_ready_kv_growth=Mock(return_value=ready_growth))
    engine_core.scheduler = SimpleNamespace(commit_ready_kv_growth=Mock(),
                                            set_pending_kv_growth=Mock())

    EngineCore._commit_ready_kv_growth(engine_core)

    engine_core.model_executor.take_ready_kv_growth.assert_called_once_with()
    engine_core.scheduler.commit_ready_kv_growth.assert_called_once_with(2, 128)
    engine_core.scheduler.set_pending_kv_growth.assert_called_once_with(False)


def test_engine_core_skips_commit_when_kv_growth_not_ready():
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.model_executor = SimpleNamespace(
        take_ready_kv_growth=Mock(return_value=None))
    engine_core.scheduler = SimpleNamespace(commit_ready_kv_growth=Mock(),
                                            set_pending_kv_growth=Mock())

    EngineCore._commit_ready_kv_growth(engine_core)

    engine_core.scheduler.commit_ready_kv_growth.assert_not_called()
    engine_core.scheduler.set_pending_kv_growth.assert_not_called()
