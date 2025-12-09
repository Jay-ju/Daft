# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.udf import las_udf
from tests.conftest import get_tests_daft_runner_name

INPUT_COLUMN_NAME = "messages"
OUTPUT_COLUMN_NAME = "llm_result"


@pytest.mark.ark_llm
def test_doubao_1_5_thinking_vision_pro_video(tos_test_data_dir):
    input_dict = {"video_path": [f"{tos_test_data_dir}/video/sample.mp4"]}

    df = daft.from_pydict(input_dict)
    df = df.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(
            ArkLLMVisionUnderstanding,
            construct_args={
                "model": "doubao-1.5-thinking-vision-pro",
                "version": "250428",
                "inference_type": "online",
                "system_text": "视频里是什么",
            },
        )(videos=col("video_path")),
    )

    result_df = df.to_pandas()
    assert "llm_result" in result_df.columns
    assert len(result_df) == 1
    assert result_df["llm_result"][0]


@pytest.mark.ark_llm
def test_doubao_1_5_thinking_vision_pro_many_sources(tos_test_data_dir, http_test_data_dir):
    input_dict = {
        "video_path": [
            [f"{tos_test_data_dir}/video/sample.mp4", f"{http_test_data_dir}/video/singer.mp4"],
            [f"{tos_test_data_dir}/video/file_example_MP4_480_1_5MG.mp4"],
        ],
        "image_path": [f"{tos_test_data_dir}/image/forest.jpg", None],
        "text": ["里面图片和视频都是描述什么内容", "里面图片和视频都是关于什么场景的，使用10个字以内回答。"],
    }

    df = daft.from_pydict(input_dict)
    df = df.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(
            ArkLLMVisionUnderstanding,
            construct_args={
                "model": "doubao-1.5-thinking-vision-pro",
                "version": "250428",
                "inference_type": "online",
                "system_text": "你是一个专业的视频理解模型，你的任务是根据视频内容和图片内容，回答用户的问题。",
            },
        )(videos=col("video_path"), images=col("image_path"), texts=col("text")),
    )

    result_df = df.to_pandas()
    assert "llm_result" in result_df.columns
    assert len(result_df) == 2
    assert result_df["llm_result"][0]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
class TestArkLLMImageUnderstandingBuildVideoMessage(unittest.TestCase):
    def setUp(self):
        self.model = "test_model"
        self.version = "test_version"
        self.ark_llm = ArkLLMVisionUnderstanding(model=self.model, version=self.version)


class MockArkLLMTextGenerateTransform(ArkLLMVisionUnderstanding):
    def __init__(self, mock_callable: callable, **kwargs):
        version = "test_version"
        api_key = "test_ak"
        model = "model"

        super().__init__(version=version, api_key=api_key, model=model, **kwargs)
        self.mock_callable = mock_callable

    async def _async_requests(self, requests: list[dict[Any, Any]]) -> pa.Array:
        return await self.mock_callable(requests)


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_process_normal_image_without_prompt():
    mock_callable = AsyncMock(
        return_value=[
            {"choices": [{"message": {"content": "response1"}}]},
            None,
            None,
            None,
            {"choices": [{"message": {"content": "response5"}}]},
        ]
    )
    mock_class = MockArkLLMTextGenerateTransform(
        mock_callable=mock_callable,
        source_type="url",
    )

    requests = pa.array(
        [
            "http://test.image.url",
            None,
            "",
            "  ",
            "http://test.image.url2",
        ]
    )
    result = mock_class.transform(images=requests)

    assert result.to_pylist() == ["response1", None, None, None, "response5"]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_process_video_with_prompt():
    ArkLLMGenerate._finish_reason_check = True
    mock_callable = AsyncMock(
        return_value=[
            {"choices": [{"message": {"content": "response1"}, "finish_reason": "stop"}]},
            None,
            {"choices": [{"message": {"content": "response3"}, "finish_reason": "stop"}]},
            None,
            {"choices": [{"message": {"content": "response6"}, "finish_reason": "stop"}, {"a": "b"}]},
        ]
    )

    mock_class = MockArkLLMTextGenerateTransform(mock_callable=mock_callable, source_type="url")

    requests = pa.array(
        [
            "http://test.video.url",
            None,
            "",
            "  ",
            "http://test.video.url2",
        ]
    )
    prompts = pa.array(
        [
            "prompt1",
            None,
            "prompt3",
            "  ",
            None,
        ]
    )
    result = mock_class.transform(videos=requests, texts=prompts)

    assert result.to_pylist() == [
        {"llm_result": "response1", "finish_reason": "stop"},
        {"llm_result": None, "finish_reason": "skip_empty_payload"},
        {"llm_result": "response3", "finish_reason": "stop"},
        {"llm_result": None, "finish_reason": "skip_empty_payload"},
        {"llm_result": "response6", "finish_reason": "stop"},
    ]
    ArkLLMGenerate._finish_reason_check = False


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_binary_input():
    mock_callable = AsyncMock(return_value=[{"choices": [{"message": {"content": "response1"}}]}])
    mock_class = MockArkLLMTextGenerateTransform(mock_callable=mock_callable, source_type="binary")
    requests = pa.array([b"base64_data"])

    result = mock_class.transform(images=requests)
    assert result.to_pylist() == ["response1"]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
def test_process_no_valid_indices():
    mock_callable = AsyncMock(return_value=[None, None])

    mock_class = MockArkLLMTextGenerateTransform(mock_callable=mock_callable, source_type="url")
    requests = pa.array(["", None])

    result = mock_class.transform(videos=requests)
    assert result.to_pylist() == [None, None]


@pytest.mark.skipif(get_tests_daft_runner_name() != "native", reason="requires Native Runner to be in use")
class TestArkLLMVisionUnderstandingPrepareModelMessages(unittest.TestCase):
    def setUp(self):
        self.model = "test_model"
        self.version = "test_version"
        self.ark_llm = ArkLLMVisionUnderstanding(model=self.model, version=self.version)

    def test_prepare_model_messages_with_images_only(self):
        """Test _prepare_model_messages with images only."""
        images = ["http://test.image1.url", "http://test.image2.url"]
        videos = None
        texts = None
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://test.image1.url"}}]}],
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://test.image2.url"}}]}],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_videos_only(self):
        """Test _prepare_model_messages with videos only."""
        images = None
        videos = ["http://test.video1.url", "http://test.video2.url"]
        texts = None
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "http://test.video1.url"}}]}],
            [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "http://test.video2.url"}}]}],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_texts_only(self):
        """Test _prepare_model_messages with texts only."""
        images = None
        videos = None
        texts = ["text1", "text2"]
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [{"role": "user", "content": [{"type": "text", "text": "text1"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "text2"}]}],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_images_and_texts(self):
        """Test _prepare_model_messages with images and texts."""
        images = ["http://test.image.url"]
        videos = None
        texts = ["Describe this image"]
        data_len = 1

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "http://test.image.url"}},
                    ],
                }
            ]
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_system_content(self):
        """Test _prepare_model_messages with system content."""
        ark_llm_with_system = ArkLLMVisionUnderstanding(
            model=self.model, version=self.version, system_text="You are a helpful assistant"
        )

        images = ["http://test.image.url"]
        videos = None
        texts = None
        data_len = 1

        result = ark_llm_with_system._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://test.image.url"}}]},
            ]
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_none_values(self):
        """Test _prepare_model_messages with None values in arrays."""
        images = ["http://test.image.url", None]
        videos = None
        texts = ["text1", "text2"]
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "text1"},
                        {"type": "image_url", "image_url": {"url": "http://test.image.url"}},
                    ],
                }
            ],
            [{"role": "user", "content": [{"type": "text", "text": "text2"}]}],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_empty_arrays(self):
        """Test _prepare_model_messages with empty arrays."""
        images = pa.array([])
        videos = None
        texts = None
        data_len = 0

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = []
        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_list_inputs(self):
        """Test _prepare_model_messages with list inputs that need flattening."""
        with patch.object(self.ark_llm, "_flatten_if_list", side_effect=lambda x: x if isinstance(x, list) else [x]):
            images = [["http://test.image1.url", "http://test.image2.url"]]
            videos = None
            texts = [["text1", "text2"]]
            data_len = 1

            result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

            expected = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "text1"},
                            {"type": "text", "text": "text2"},
                            {"type": "image_url", "image_url": {"url": "http://test.image1.url"}},
                            {"type": "image_url", "image_url": {"url": "http://test.image2.url"}},
                        ],
                    }
                ]
            ]

            self.assertEqual(result, expected)

    def test_prepare_model_messages_with_multiple_images_and_videos(self):
        """Test _prepare_model_messages with multiple images and videos in lists."""
        images = [["http://test.image1.url", "http://test.image2.url"], ["http://test.image3.url"]]
        videos = [
            ["http://test.video1.url", "http://test.video2.url"],
            ["http://test.video3.url", "http://test.video4.url"],
        ]

        texts = ["Describe these images and videos", "What's in these media files?"]
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe these images and videos"},
                        {"type": "image_url", "image_url": {"url": "http://test.image1.url"}},
                        {"type": "image_url", "image_url": {"url": "http://test.image2.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video1.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video2.url"}},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in these media files?"},
                        {"type": "image_url", "image_url": {"url": "http://test.image3.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video3.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video4.url"}},
                    ],
                }
            ],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_mixed_single_and_list_inputs(self):
        """Test _prepare_model_messages with mixed single items and lists."""
        images = [
            ["http://test.image1.url"],
            ["http://test.image2.url", "http://test.image3.url"],
        ]

        videos = [
            ["http://test.video1.url", "http://test.video2.url"],
            ["http://test.video3.url"],
        ]

        texts = ["First prompt", "Second prompt"]
        data_len = 2

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First prompt"},
                        {"type": "image_url", "image_url": {"url": "http://test.image1.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video1.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video2.url"}},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Second prompt"},
                        {"type": "image_url", "image_url": {"url": "http://test.image2.url"}},
                        {"type": "image_url", "image_url": {"url": "http://test.image3.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video3.url"}},
                    ],
                }
            ],
        ]

        self.assertEqual(result, expected)

    def test_prepare_model_messages_with_complex_multimedia_combinations(self):
        """Test _prepare_model_messages with complex combinations of multimedia."""
        images = [
            ["http://test.image1.url", "http://test.image2.url", "http://test.image3.url"],
            None,
            ["http://test.image4.url"],
        ]
        videos = [
            ["http://test.video1.url"],
            ["http://test.video2.url", "http://test.video3.url"],
            None,
        ]
        texts = ["Single prompt"]
        data_len = 3

        result = self.ark_llm._prepare_model_messages(images, videos, texts, data_len)

        expected = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Single prompt"},
                        {"type": "image_url", "image_url": {"url": "http://test.image1.url"}},
                        {"type": "image_url", "image_url": {"url": "http://test.image2.url"}},
                        {"type": "image_url", "image_url": {"url": "http://test.image3.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video1.url"}},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Single prompt"},
                        {"type": "video_url", "video_url": {"url": "http://test.video2.url"}},
                        {"type": "video_url", "video_url": {"url": "http://test.video3.url"}},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Single prompt"},
                        {"type": "image_url", "image_url": {"url": "http://test.image4.url"}},
                    ],
                }
            ],
        ]

        self.assertEqual(result, expected)
