# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result

INPUT_COLUMN_NAME = "messages"
OUTPUT_COLUMN_NAME = "llm_result"


@pytest.fixture
def mock_client():
    with patch("daft.las.functions.ark_llm.ark_llm_generate.LasArkClient") as mock:
        client = MagicMock()
        # Use the current event loop to create a future object
        client.batch_process = MagicMock(return_value=asyncio.Future())
        mock.return_value = client
        yield client


@pytest.fixture
def sample_table():
    data = ["test1", "test2", "test3", "", None, "test4"]
    df = pd.DataFrame({INPUT_COLUMN_NAME: [[{"role": "user", "content": query}] for query in data]})
    return df


def test_normal_transform(mock_client, sample_table):
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response2"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response4"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response5"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response6"}, "finish_reason": "length"}]},
        ]
    )
    mock_client.batch_process.return_value = future

    ds = daft.from_pandas(sample_table)
    ds = ds.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(ArkLLMGenerate, construct_args={"model": "test_model", "version": "11"})(col(INPUT_COLUMN_NAME)),
    )

    # Check the result
    expect_df = sample_table.copy().assign(
        llm_result=["response1", "response2", "response3", "response4", "response5", "response6"]
    )
    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])


def test_error_handling(mock_client, sample_table):
    """Test error handling in transform method."""
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "length"}]},
            {"error": "Invalid request"},
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response6"}, "finish_reason": "stop"}, {"a": "b"}]},
        ]
    )
    mock_client.batch_process.return_value = future

    ds = daft.from_pandas(sample_table)
    ds = ds.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(ArkLLMGenerate, construct_args={"model": "test_model", "version": "11"})(col(INPUT_COLUMN_NAME)),
    )

    # Check the result
    expect_df = sample_table.copy().assign(
        llm_result=["response1", "response1", "response3", None, "response1", "response6"]
    )
    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])


def test_service_unavailable(mock_client, sample_table):
    """Test transform method when service is unavailable."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        future = loop.create_future()
        future.set_exception(Exception("Service unavailable"))
        mock_client.batch_process.return_value = future

        ds = daft.from_pandas(sample_table)
        ds = ds.with_column(
            OUTPUT_COLUMN_NAME,
            las_udf(ArkLLMGenerate, construct_args={"model": "test_model", "version": "11"})(col(INPUT_COLUMN_NAME)),
        )

        try:
            ds.to_pandas()
        except Exception as e:
            assert "Service unavailable" in str(e)
    finally:
        loop.close()


def test_with_finish_reason_check(mock_client, sample_table):
    ArkLLMGenerate._finish_reason_check = True
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response2"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response4"}}]},
            {"choices": [{"message": {"content": None, "finish_reason": "Invalid request"}}]},
            {"choices": [{"message": {"content": "response6"}, "finish_reason": "length"}]},
        ]
    )
    mock_client.batch_process.return_value = future

    ds = daft.from_pandas(sample_table)
    ds = ds.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(ArkLLMGenerate, construct_args={"model": "test_model", "version": "11"})(col(INPUT_COLUMN_NAME)),
    )

    print(ds.to_pandas())
    # Check the result
    expect_df = sample_table.copy().assign(
        llm_result=[
            {"llm_result": "response1", "finish_reason": "stop"},
            {"llm_result": "response2", "finish_reason": "length"},
            {"llm_result": "response3", "finish_reason": "length"},
            {"llm_result": "response4", "finish_reason": None},
            {"llm_result": None, "finish_reason": None},
            {"llm_result": "response6", "finish_reason": "length"},
        ]
    )

    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])
