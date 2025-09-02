# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_embedding_vision import DoubaoEmbeddingVision
from daft.las.functions.udf import las_udf
from tests.conftest import get_tests_daft_runner_name

INPUT_COLUMN_NAME = "image_path"
OUTPUT_COLUMN_NAME = "llm_result"


@pytest.mark.ark_llm
def test_doubao_embedding_vision(tos_test_data_dir):
    input_dict = {INPUT_COLUMN_NAME: [f"tos://{tos_test_data_dir}/image/cat.png"], "text": ["猫"]}

    df = daft.from_pydict(input_dict)
    df = df.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(
            DoubaoEmbeddingVision,
            construct_args={
                "version": "250328",
                "image_format": "png",
            },
        )(col(INPUT_COLUMN_NAME), col("text")),
    )

    result_df = df.to_pandas()
    assert OUTPUT_COLUMN_NAME in result_df.columns
    assert len(result_df) == 1
    assert len(result_df[OUTPUT_COLUMN_NAME][0]) == 2048


class TestDoubaoEmbeddingVision(DoubaoEmbeddingVision):
    def __init__(self, mock_callable: callable, **kwargs):
        version = "test_version"

        super().__init__(version=version, **kwargs)
        self.mock_callable = mock_callable

    async def _async_requests(self, requests: list[dict[Any, Any]]) -> pa.Array:
        return await self.mock_callable(requests)


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_image_embedding_without_text():
    """Test image embedding without text."""
    mock_callable = AsyncMock(return_value=[{"data": {"embedding": [0.1, 0.2, 0.3]}}] * 2)
    embedder = TestDoubaoEmbeddingVision(mock_callable=mock_callable, multimodal_type="image", source_type="url")

    media_data = pa.array(["http://test.com/img1.jpg", "http://test.com/img2.jpg"])
    result = embedder.transform(media_datas=media_data)

    assert result.type == pa.list_(pa.float32())
    assert len(result) == 2

    from pytest import approx

    result_list = result.to_pandas().tolist()
    assert result_list[0].tolist() == approx([0.1, 0.2, 0.3])
    assert result_list[1].tolist() == approx([0.1, 0.2, 0.3])


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_video_embedding_with_text():
    """Test video embedding with text."""
    mock_callable = AsyncMock(return_value=[{"data": {"embedding": [0.1, 0.2, 0.3]}}] * 2)
    embedder = TestDoubaoEmbeddingVision(
        mock_callable=mock_callable, multimodal_type="video", source_type="url", dimensions=None
    )
    media_data = pa.array(["http://test.com/video1.mp4", "http://test.com/video2.mp4"])
    texts = pa.array(["a cat", "a dog"])
    result = embedder.transform(media_datas=media_data, text_contents=texts)

    # Check the result type and structure
    assert result.type == pa.list_(pa.float32())
    assert len(result) == 2

    from pytest import approx

    result_list = result.to_pandas().tolist()
    assert result_list[0].tolist() == approx([0.1, 0.2, 0.3])
    assert result_list[1].tolist() == approx([0.1, 0.2, 0.3])

    call_args = embedder.mock_callable.call_args[0][0]
    expected_list = [
        {
            "input": [
                {"type": "text", "text": "a cat"},
                {"type": "video_url", "videoUrl": "http://test.com/video1.mp4"},
            ],
            "model_name": "doubao-embedding-vision",
            "version": "test_version",
        },
        {
            "input": [
                {"type": "text", "text": "a dog"},
                {"type": "video_url", "videoUrl": "http://test.com/video2.mp4"},
            ],
            "model_name": "doubao-embedding-vision",
            "version": "test_version",
        },
    ]
    assert call_args == expected_list


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_error_handling():
    """Test error handling."""
    mock_callable = AsyncMock(return_value=[{"error": "Invalid input"}] * 2)
    embedder = TestDoubaoEmbeddingVision(mock_callable=mock_callable)
    media_data = pa.array(["http://invalid_url1", "http://invalid_url2"])
    result = embedder.transform(media_datas=media_data)

    # Check the error handling
    assert result.to_pylist() == [None, None]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_binary_input():
    """Test binary input."""
    mock_callable = AsyncMock(return_value=[{"data": {"embedding": [0.1, 0.2, 0.3]}}] * 2)
    embedder = TestDoubaoEmbeddingVision(
        mock_callable=mock_callable,
        multimodal_type="image",
        source_type="binary",
    )
    media_data = pa.array([b"fake_image_data", b"fake_image_data"])
    result = embedder.transform(media_datas=media_data)

    # Check the result type and structure
    assert result
    assert isinstance(result, pa.Array)
    assert len(result) == 2
    assert result.type == pa.list_(pa.float32())
    # Check the result length
    assert len(result[0].as_py()) == 3

    # Check the binary input
    call_args = embedder.mock_callable.call_args[0][0]
    assert call_args[0]["input"][0]["imageUrl"].startswith("data:image/jpeg;base64")


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_image_embedding_with_empty():
    """Test image embedding with empty."""
    mock_callable = AsyncMock(
        return_value=[{"data": {"embedding": [0.1, 0.2, 0.3]}}, None, None, {"data": {"error": "error"}}]
    )
    embedder = TestDoubaoEmbeddingVision(mock_callable=mock_callable, multimodal_type="image", source_type="url")

    media_data = pa.array(["http://test.com/img1.jpg", "", None, "http://test.com/img2.jpg"])
    result = embedder.transform(media_datas=media_data)

    assert result.type == pa.list_(pa.float32())
    assert len(result) == 4

    from pytest import approx

    result_list = result.to_pandas().tolist()
    assert result_list[0].tolist() == approx([0.1, 0.2, 0.3])
    assert result_list[1] is None
    assert result_list[2] is None
    assert result_list[3] is None
