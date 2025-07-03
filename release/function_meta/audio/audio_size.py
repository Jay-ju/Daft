from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_size import AudioSize
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_size/sample.mp3"]}

    df = daft.from_pydict(samples)
    df = df.with_column("size_result", las_udf(AudioSize)(col("audio_path")))

    df.show()
    # ╭────────────────────────────────┬─────────────╮
    # │ audio_path                     ┆ size_result │
    # │ ---                            ┆ ---         │
    # │ Utf8                           ┆ Float32     │
    # ╞════════════════════════════════╪═════════════╡
    # │ tos://tos_bucket/audio_size/s… ┆ 795426      │
    # ╰────────────────────────────────┴─────────────╯
