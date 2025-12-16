from __future__ import annotations

import logging
import os

import daft
from daft import col
from daft.las.functions.audio import AudioMetascore
from daft.las.functions.udf import las_udf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_metascore/sample.wav"]}
    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_metascore",
        las_udf(
            AudioMetascore,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    df.show()
    # ╭────────────────────────────────────────────┬────────────────────────────────────────────────────────────╮
    # │ audio_path                                 ┆ audio_metascore                                            │
    # │ ---                                        ┆ ---                                                        │
    # │ String                                     ┆ Struct[CE: Float64, CU: Float64, PC: Float64, PQ: Float64] │
    # ╞════════════════════════════════════════════╪════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_metascore/sample... ┆ {CE: 5.909880638122559,                                    │
    # │                                            ┆ CU: 6…                                                     │
    # ╰────────────────────────────────────────────┴────────────────────────────────────────────────────────────╯
