# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_standardization import AudioStandardization
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")

    # 示例输入数据
    samples = {"audio_bytes": [f"tos://{TOS_TEST_DIR}/audio_standardization/耙耙柑大叔.aac"]}

    # 构建 Daft DataFrame
    df = daft.from_pydict(samples)

    # 应用 AudioStandardization 算子
    df = df.with_column(
        "standardized_audio",
        las_udf(
            AudioStandardization,
            construct_args={
                "target_sr": 16000,
                "target_channels": 1,
                "target_dbfs": -20.0,
                "target_gain_range": [-3.0, 3.0],
                "concurrency": 2,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=2,
        )(col("audio_bytes")),
    )

    df.show()
    # ╭────────────────────────────────┬────────────────────────────────╮
    # │ audio_bytes                    ┆ standardized_audio             │
    # │ ---                            ┆ ---                            │
    # │ Utf8                           ┆ Binary                         │
    # ╞════════════════════════════════╪════════════════════════════════╡
    # │ tos: // las - ai - cn - beijing / qa / op… ┆ b"RIFF\xce\x9a\x08\x00WAVEfmt… │
    # ╰────────────────────────────────┴────────────────────────────────╯
