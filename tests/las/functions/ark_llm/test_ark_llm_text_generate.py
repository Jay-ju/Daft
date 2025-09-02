# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pyarrow as pa
import pytest

from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.ark_llm.ark_llm_text_generate import ArkLLMTextGenerate
from tests.conftest import get_tests_daft_runner_name


class MockArkLLMTextGenerateTransform(ArkLLMTextGenerate):
    def __init__(self, mock_callable: callable, **kwargs):
        version = "test_version"
        api_key = "test_ak"
        model = "model"

        super().__init__(version=version, api_key=api_key, model=model, **kwargs)
        self.mock_callable = mock_callable

    async def _async_requests(self, requests: list[dict[Any, Any]]) -> pa.Array:
        return await self.mock_callable(requests)


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_process_normal_case():
    ArkLLMGenerate._finish_reason_check = False
    mock_callable = AsyncMock(
        return_value=[
            {"choices": [{"message": {"content": "response1"}}]},
            {"choices": [{"message": {"content": "response2"}}]},
            None,
            None,
            None,
        ]
    )
    mock_ark_llm_text_generate = MockArkLLMTextGenerateTransform(mock_callable=mock_callable)

    requests = pa.array(["问题1", "问题2", None, "", "  "])
    result = mock_ark_llm_text_generate.transform(requests)

    assert result.to_pylist() == ["response1", "response2", None, None, None]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_process_empty_case():
    ArkLLMGenerate._finish_reason_check = False
    mock_callable = AsyncMock(return_value=[])
    mock_ark_llm_text_generate = MockArkLLMTextGenerateTransform(mock_callable=mock_callable)
    requests = pa.array([])
    result = mock_ark_llm_text_generate.transform(requests)
    assert result.to_pylist() == []

    mock_callable = AsyncMock(return_value=[None])
    mock_ark_llm_text_generate = MockArkLLMTextGenerateTransform(mock_callable=mock_callable)
    requests = pa.array([None])
    result = mock_ark_llm_text_generate.transform(requests)
    assert result.to_pylist() == [None]

    mock_callable = AsyncMock(return_value=[None])
    mock_ark_llm_text_generate = MockArkLLMTextGenerateTransform(mock_callable=mock_callable)
    requests = pa.array([""])
    result = mock_ark_llm_text_generate.transform(requests)
    assert result.to_pylist() == [None]

    mock_callable = AsyncMock(return_value=[None, None, None, None])
    mock_ark_llm_text_generate = MockArkLLMTextGenerateTransform(mock_callable=mock_callable)
    requests = pa.array(["", None, "   "])
    result = mock_ark_llm_text_generate.transform(requests)
    assert result.to_pylist() == [None, None, None, None]
