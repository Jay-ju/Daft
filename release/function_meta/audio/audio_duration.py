from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioDuration
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 环境变量配置
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    sample_path = f"tos://{TOS_TEST_DIR}/audio_duration/sample.mp3"

    df = daft.from_pydict({"audio_path": [sample_path]})
    df = df.with_column("duration_result", las_udf(AudioDuration)(col("audio_path")))
    df.show()
    # ╭────────────────────────────────┬─────────────────╮
    # │ audio_path                     ┆ duration_result │
    # │ ---                            ┆ ---             │
    # │ Utf8                           ┆ Float32         │
    # ╞════════════════════════════════╪═════════════════╡
    # │ tos://tos_bucket/audio_durati… ┆ 49.711          │
    # ╰────────────────────────────────┴─────────────────╯
