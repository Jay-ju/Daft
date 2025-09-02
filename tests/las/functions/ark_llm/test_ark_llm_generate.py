# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.udf import las_udf
from tests.conftest import get_tests_daft_runner_name
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
    df = pd.DataFrame(
        {
            INPUT_COLUMN_NAME: [
                [{"role": "user", "content": "test1"}],
                [{"role": "user", "content": "test2"}],
                [{"role": "user", "content": "test3"}],
                [],
                None,
                [{"role": "user", "content": "test4"}],
            ]
        }
    )
    return df


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_normal_transform(mock_client, sample_table):
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response2"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "length"}]},
            None,
            None,
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
    expect_df = sample_table.copy().assign(llm_result=["response1", "response2", "response3", None, None, "response6"])
    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_error_handling(mock_client, sample_table):
    """Test error handling in transform method."""
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "length"}]},
            {"error": "Invalid request"},
            None,
            None,
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
    expect_df = sample_table.copy().assign(llm_result=["response1", "response3", None, None, None, "response6"])
    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
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


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_with_finish_reason_check(mock_client, sample_table):
    ArkLLMGenerate._finish_reason_check = True
    future = asyncio.Future()
    future.set_result(
        [
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "response2"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "response4"}}]},
            None,
            None,
            {"choices": [{"message": {"content": ""}, "finish_reason": "Invalid request"}]},
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
        llm_result=[
            {"llm_result": "response1", "finish_reason": "stop"},
            {"llm_result": "response2", "finish_reason": "length"},
            {"llm_result": "response4", "finish_reason": None},
            {"llm_result": None, "finish_reason": "skip_empty_payload"},
            {"llm_result": None, "finish_reason": "skip_empty_payload"},
            {"llm_result": "", "finish_reason": "Invalid request"},
        ]
    )

    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])
    ArkLLMGenerate._finish_reason_check = False


def test_no_valid_indices():
    sample_table = pd.DataFrame({INPUT_COLUMN_NAME: [None, []]})

    ds = daft.from_pandas(sample_table)
    ds = ds.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(ArkLLMGenerate, construct_args={"model": "test_model", "version": "11"})(col(INPUT_COLUMN_NAME)),
    )

    # Check the result
    expect_df = sample_table.copy().assign(llm_result=[None, None])

    assert_dataframe_result(ds.to_pandas(), expect_df, [INPUT_COLUMN_NAME, OUTPUT_COLUMN_NAME])


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
class TestArkLLMGenerateTransform:
    valid_messages = [[{"role": "user", "content": "你好"}], [{"role": "user", "content": "今天天气怎么样"}]]
    invalid_messages = [[], None]
    mixed_messages = valid_messages + invalid_messages
    mock_response = {"llm_result": "这是模型生成的回复", "finish_reason": "stop"}

    @pytest.fixture
    def operator(self):
        return ArkLLMGenerate(model="doubao-1.5-lite-32k", version="250115", api_key="test_key")

    def test_transform_with_valid_messages(self, operator):
        with patch.object(operator, "process", return_value=pa.array([self.mock_response] * len(self.valid_messages))):
            input_array = pa.array(self.valid_messages)
            result = operator.transform(input_array)

            assert isinstance(result, pa.Array)
            assert len(result) == len(self.valid_messages)
            assert pc.all(pc.is_valid(result)).as_py()

    def test_transform_with_invalid_messages(self, operator):
        input_array = pa.array(self.invalid_messages)
        result = operator.transform(input_array)

        assert isinstance(result, pa.Array)
        assert len(result) == len(self.invalid_messages)
        assert pc.all(pc.is_null(result)).as_py()

    def test_transform_with_mixed_messages(self, operator):
        with patch.object(
            operator,
            "process",
            return_value=pa.array(
                [self.mock_response, self.mock_response]
                + [None] * (len(self.mixed_messages) - len(self.valid_messages))
            ),
        ):
            input_array = pa.array(self.mixed_messages)
            result = operator.transform(input_array)

            assert isinstance(result, pa.Array)
            assert len(result) == len(self.mixed_messages)

            assert not pc.is_null(result[0]).as_py()
            assert not pc.is_null(result[1]).as_py()

            assert pc.is_null(result[2]).as_py()
            assert pc.is_null(result[3]).as_py()

    def test_transform_with_empty_input(self, operator):
        input_array = pa.array([], type=pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])))
        result = operator.transform(input_array)

        assert isinstance(result, pa.Array)
        assert len(result) == 0

    def test_transform_return_type_without_finish_reason(self, operator):
        with patch.object(operator, "process", return_value=pa.array(["response1", "response2"])):
            input_array = pa.array(self.valid_messages)
            result = operator.transform(input_array)

            assert isinstance(result, pa.Array)
            assert result.type == pa.string()

    def test_transform_mask_logic(self, operator):
        input_array = pa.array(self.mixed_messages)

        mask = pc.and_(pc.is_valid(input_array), pc.greater(pc.list_value_length(input_array), 0))
        expected_indices = pc.indices_nonzero(mask).to_pylist()

        computed_mask = pc.and_(pc.is_valid(input_array), pc.greater(pc.list_value_length(input_array), 0))
        computed_indices = pc.indices_nonzero(computed_mask).to_pylist()

        assert computed_indices == expected_indices
