# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.audio import AudioExtractAndSplit
from daft.las.functions.udf import las_udf


def test_audio_extract_and_split(local_test_data_dir: str, tos_test_data_dir: str):
    df = daft.from_pydict(
        {
            "audio_path": [
                "",  # 空路径
                f"{local_test_data_dir}/audio/non-exist.wav",  # 本地不存在
                f"{tos_test_data_dir}/audio/test_music.m4a",  # 有效 TOS 文件
            ]
        }
    )

    df = df.with_column(
        "audio_chunks",
        las_udf(
            AudioExtractAndSplit,
            construct_args={"split_duration": 5},  # 5s
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    result_df = df.to_pandas()

    assert result_df.iloc[0]["audio_chunks"] is None

    assert result_df.iloc[1]["audio_chunks"] is None

    chunks = result_df.iloc[2]["audio_chunks"]
    assert len(chunks) == 2, f"Expected 2 chunk, got {len(chunks)}"
