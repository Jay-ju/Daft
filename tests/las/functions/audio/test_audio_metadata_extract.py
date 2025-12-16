from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioMetadataExtract
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir):
    """生成测试数据."""
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/audio/non-exist.mp3",  # 不存在的文件
            f"{tos_test_data_dir}/audio/sample.wav",  # TOS 音频文件
        ],
    }

    input_df = pd.DataFrame(samples)

    # 预期结果 - 不存在的文件应返回 None
    # 实际文件应返回元数据结构
    expected_results = [
        None,  # 不存在的文件
        {  # 音频文件的元数据(sample.wav的实际值)
            "duration": 49.71102,
            "format_name": "wav",
            "bit_rate": 1411212,
            "audio_codec": "pcm_s16le",
            "audio_sample_rate": 44100,
            "audio_channels": 2,
        },
    ]

    expected_df = pd.DataFrame({"metadata": expected_results})

    return input_df, expected_df


def test_audio_metadata_extract_basic(tos_test_data_dir):
    """测试 AudioMetadataExtract 基本功能."""
    input_df, _expected_df = generate_test_data(tos_test_data_dir)

    # 创建算子参数
    constructor_kwargs = {
        "timeout": None,
    }

    # 创建 Daft DataFrame
    df = daft.from_pandas(input_df)

    # 应用算子
    result_df = (
        df.with_column(
            "metadata",
            las_udf(
                AudioMetadataExtract,
                construct_args=constructor_kwargs,
                num_gpus=0,
                batch_size=1,
                concurrency=1,
            )(col("input_path")),
        )
        .select(col("metadata"))
        .to_pandas()
    )

    # 验证结果
    # 1. 不存在的文件返回 None
    assert result_df["metadata"][0] is None, "Non-existent file should return None"

    # 2. 验证音频文件的元数据 - 使用实际值
    metadata = result_df["metadata"][1]
    assert metadata is not None, "Audio file should return metadata"
    assert metadata["duration"] == 49.71102, f"Expected duration 49.71102, got {metadata['duration']}"
    assert metadata["format_name"] == "wav", f"Expected format_name 'wav', got {metadata['format_name']}"
    assert metadata["bit_rate"] == 1411212, f"Expected bit_rate 1411212, got {metadata['bit_rate']}"
    assert metadata["audio_codec"] == "pcm_s16le", f"Expected audio_codec 'pcm_s16le', got {metadata['audio_codec']}"
    assert (
        metadata["audio_sample_rate"] == 44100
    ), f"Expected audio_sample_rate 44100, got {metadata['audio_sample_rate']}"
    assert metadata["audio_channels"] == 2, f"Expected audio_channels 2, got {metadata['audio_channels']}"
