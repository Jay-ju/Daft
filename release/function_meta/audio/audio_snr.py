from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioSNR
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "audio_path": [f"tos://{TOS_TEST_DIR}/audio_snr/sample.mp3"],
    }

    ds = daft.from_pydict(samples)

    snr_udf = las_udf(
        AudioSNR,
        construct_args={
            "n_components": 2,
            "max_iter": 200,
        },
    )

    ds = ds.with_column("snr_db", snr_udf(col("audio_path")))
    ds.show()
    # 示例输出:
    # ╭──────────────────────────────────────────┬──────────╮
    # │ audio_path                               ┆ snr_db   │
    # │ ---                                      ┆ ---      │
    # │ Utf8                                     ┆ Float64  │
    # ╞══════════════════════════════════════════╪══════════╡
    # │ tos://tos_bucket/audio_snr/sample.wav    ┆ 12.34    │
    # ╰──────────────────────────────────────────┴──────────╯
