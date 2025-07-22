# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.udf import las_udf

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
                "multimodal_type": "video",
                "prompt": "视频里是什么",
            },
        )(col("video_path")),
    )

    result_df = df.to_pandas()
    assert "llm_result" in result_df.columns
    assert len(result_df) == 1
    assert result_df["llm_result"][0]


def test_full_message_with_system():
    """Test full message structure with system content."""
    vision_generate = ArkLLMVisionUnderstanding(
        model="test_model",
        version="test_version",
        multimodal_type="image",
        source_type="url",
        system_text="System instruction",
    )
    # test gen message
    result = vision_generate._build_image_message("http://test.media.url")

    except_res = [
        {"role": "system", "content": [{"type": "text", "text": "System instruction"}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://test.media.url"}}]},
    ]
    assert result == except_res


def test_gen_message_video_with_user_prompt():
    """Test video type and user prompt parameter."""
    vision_generate = ArkLLMVisionUnderstanding(
        model="test_model",
        version="test_version",
        multimodal_type="video",
        source_type="url",
        video_fps=2.0,
    )

    result = vision_generate._build_video_message("http://test.video.url", user_prompt="Analyze this video")
    except_res = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video"},
                {"type": "video_url", "video_url": {"url": "http://test.video.url", "fps": 2.0}},
            ],
        }
    ]

    assert result == except_res


def test_gen_message_image_with_system_content():
    """Test image type and system content parameter."""
    vision_generate = ArkLLMVisionUnderstanding(
        model="test_model",
        version="test_version",
        multimodal_type="image",
        source_type="url",
        system_text="You are a helpful assistant.",
        image_url_detail="detail info",
    )

    result = vision_generate._build_image_message("http://test.image.url")

    except_res = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "http://test.image.url", "detail": "detail info"}}],
        },
    ]

    assert result == except_res


def test_gen_message_video_with_tos_url():
    """Test video type and user prompt parameter."""
    vision_generate = ArkLLMVisionUnderstanding(
        model="test_model",
        version="test_version",
        multimodal_type="video",
        source_type="url",
        video_fps=2.0,
    )

    os.environ["TOS_PRE_SIGN_URL_EXPIRES"] = "3600"
    result = vision_generate._build_video_message("tos://test_bucket/object_for_test", user_prompt="Analyze this video")
    sign_url = result[0]["content"][1]["video_url"]["url"]
    assert sign_url and sign_url.startswith("https") and "X-Tos-Expires=3600" in sign_url


class TestArkLLMImageUnderstandingBuildVideoMessage(unittest.TestCase):
    def setUp(self):
        self.model = "test_model"
        self.version = "test_version"
        self.ark_llm = ArkLLMVisionUnderstanding(model=self.model, version=self.version, multimodal_type="video")

    @patch.object(ArkLLMVisionUnderstanding, "_create_video_content")
    @patch.object(ArkLLMVisionUnderstanding, "_assemble_message")
    def test_build_video_message_with_user_prompt(self, mock_assemble, mock_create):
        """Test video type and user prompt parameter."""
        # Mock data
        mock_create.return_value = {"video": "test_video_data"}
        mock_assemble.return_value = {"role": "user", "content": "test_content"}

        # Test data
        media_data = "test_media_data"
        user_prompt = "test_prompt"

        # Build message
        result = self.ark_llm._build_video_message(media_data, user_prompt)

        # Assert
        mock_create.assert_called_once_with(media_data)
        mock_assemble.assert_called_once_with(video_content={"video": "test_video_data"}, user_prompt=user_prompt)
        self.assertEqual(result, {"role": "user", "content": "test_content"})

    @patch.object(ArkLLMVisionUnderstanding, "_create_video_content")
    @patch.object(ArkLLMVisionUnderstanding, "_assemble_message")
    def test_build_video_message_without_user_prompt(self, mock_assemble, mock_create):
        """Test video type and without user prompt parameter."""
        mock_create.return_value = {"video": "test_video_data"}
        mock_assemble.return_value = {"role": "user", "content": "default_content"}

        media_data = "test_media_data"

        result = self.ark_llm._build_video_message(media_data)

        mock_create.assert_called_once_with(media_data)
        mock_assemble.assert_called_once_with(video_content={"video": "test_video_data"}, user_prompt=None)
        self.assertEqual(result, {"role": "user", "content": "default_content"})

    @patch.object(ArkLLMVisionUnderstanding, "_create_video_content")
    def test_build_video_message_with_empty_media_data(self, mock_create):
        """Test video type and empty media data."""
        mock_create.side_effect = ValueError("Invalid media data")

        media_data = ""

        with self.assertRaises(ValueError) as context:
            self.ark_llm._build_video_message(media_data)

        self.assertEqual(str(context.exception), "Invalid media data")

    @patch.object(ArkLLMVisionUnderstanding, "_create_video_content")
    @patch.object(ArkLLMVisionUnderstanding, "_assemble_message")
    def test_build_video_message_with_special_characters(self, mock_assemble, mock_create):
        """Test video type and media data with special characters."""
        mock_create.return_value = {"video": "special_chars_data"}
        mock_assemble.return_value = {"role": "user", "content": "special_content"}

        media_data = "data_with_特殊字符"
        user_prompt = "prompt_with_特殊字符"

        result = self.ark_llm._build_video_message(media_data, user_prompt)

        mock_create.assert_called_once_with(media_data)
        mock_assemble.assert_called_once_with(video_content={"video": "special_chars_data"}, user_prompt=user_prompt)
        self.assertEqual(result, {"role": "user", "content": "special_content"})
