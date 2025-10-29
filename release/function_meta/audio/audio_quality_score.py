from __future__ import annotations

import logging
import os

import daft
from daft import col
from daft.las.functions.audio import AudioQualityScore
from daft.las.functions.udf import las_udf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    model_path = os.getenv("MODEL_PATH", "./models")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_quality_score/sample.wav"]}
    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_quality_score",
        las_udf(
            AudioQualityScore,
            construct_args={"model_path": model_path, "device": "cuda"},
            num_gpus=1,
            batch_size=8,
            concurrency=1,
        )(col("audio_path")),
    )

    df.show()
    # ╭────────────────────────────────┬───────────────────────────────────────────────────╮
    # │ audio_path                     ┆ audio_quality_score                               │
    # │ ---                            ┆ ---                                               │
    # │ Utf8                           ┆ Struct[ovrl: Float64, sig: Float64, bak: Float64] │
    # ╞════════════════════════════════╪═══════════════════════════════════════════════════╡
    # │ tos://las-ai-qa-online/qa/tes… ┆ {ovrl: 1.746929928948411,                         │
    # │                                ┆ sig…                                              │
    # ╰────────────────────────────────┴───────────────────────────────────────────────────╯
