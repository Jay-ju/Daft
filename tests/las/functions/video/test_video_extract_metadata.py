from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoExtractMetadata


def generate_test_data(tos_test_data_dir):
    """生成测试数据."""
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/video/non-exist.mp4",  # 不存在的文件
            f"{tos_test_data_dir}/video/music_sample.mp4",  # TOS 视频文件
        ],
    }

    input_df = pd.DataFrame(samples)

    return input_df


def test_video_extract_metadata_basic(tos_test_data_dir):
    """测试 VideoExtractMetadata 基本功能."""
    input_df = generate_test_data(tos_test_data_dir)

    # 创建算子参数
    constructor_kwargs = {
        "timeout": 600,
    }

    # 创建 Daft DataFrame
    df = daft.from_pandas(input_df)

    # 应用算子
    result_df = (
        df.with_column(
            "metadata",
            las_udf(
                VideoExtractMetadata,
                construct_args=constructor_kwargs,
                num_gpus=0,
                batch_size=1,
                concurrency=1,
            )(col("input_path")),
        )
        .select(col("metadata"))
        .to_pandas()
    )

    # 验证不存在的文件返回 None
    assert result_df["metadata"][0] is None, "Non-existent file should return None"

    # 验证视频文件的元数据 - 使用 music_sample.mp4 的真实值
    metadata = result_df["metadata"][1]
    assert metadata is not None, "Video file should return metadata"

    # 基础格式信息
    assert abs(metadata["duration"] - 7.367) < 0.01, f"Duration should be ~7.367, got {metadata['duration']}"
    assert "mp4" in metadata["format_name"], f"Format should contain 'mp4', got {metadata['format_name']}"
    assert metadata["bit_rate"] == 1058326, f"Bit rate should be 1058326, got {metadata['bit_rate']}"

    # 视频流信息
    assert metadata["has_video"] is True, "Should have video stream"
    assert metadata["video_codec"] == "h264", f"Video codec should be h264, got {metadata['video_codec']}"
    assert metadata["video_width"] == 720, f"Video width should be 720, got {metadata['video_width']}"
    assert metadata["video_height"] == 720, f"Video height should be 720, got {metadata['video_height']}"
    assert metadata["video_fps"] == 30.0, f"Video fps should be 30.0, got {metadata['video_fps']}"

    # 音频流信息
    assert metadata["has_audio"] is True, "Should have audio stream"
    assert metadata["audio_codec"] == "aac", f"Audio codec should be aac, got {metadata['audio_codec']}"
    assert (
        metadata["audio_sample_rate"] == 44100
    ), f"Audio sample rate should be 44100, got {metadata['audio_sample_rate']}"
    assert metadata["audio_channels"] == 2, f"Audio channels should be 2, got {metadata['audio_channels']}"
