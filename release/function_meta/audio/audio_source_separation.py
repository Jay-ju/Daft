from __future__ import annotations

import logging
import os

import daft
from daft import col
from daft.las.functions.audio import AudioSourceSeparation
from daft.las.functions.udf import las_udf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    model_path = os.getenv("MODEL_PATH", "./models")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_source_separation/test_music.m4a"]}
    rank = 0
    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_vocal",
        las_udf(
            AudioSourceSeparation,
            construct_args={"model_path": model_path},
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    df.show()
    # ╭────────────────────────────────┬────────────────────────────────╮
    # │ audio_path                     ┆ audio_vocal                    │
    # │ ---                            ┆ ---                            │
    # │ Utf8                           ┆ Binary                         │
    # ╞════════════════════════════════╪════════════════════════════════╡
    # │ tos: // las - ai - cn - beijing / qa / op… ┆ b"RIFF\xbaJ\r\x00WAVEfmt \x10… │
    # ╰────────────────────────────────┴────────────────────────────────╯
