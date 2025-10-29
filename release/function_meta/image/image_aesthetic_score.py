from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image import ImageAestheticScore
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "input_path": [f"tos://{TOS_TEST_DIR}/image_aesthetic_score/forest.jpg"],
    }
    ds = daft.from_pydict(samples)

    # Using Daft to calculate aesthetic scores for images
    constructor_kwargs = {
        "image_src_type": "image_url",
        "batch_size": 1,
    }

    ds = ds.with_column(
        "aesthetic_score",
        las_udf(ImageAestheticScore, construct_args=constructor_kwargs)(col("input_path")),
    )

    ds.show()
    # ╭────────────────────────────────┬────────────────────╮
    # │ input_path                     ┆ aesthetic_score    │
    # │ ---                            ┆ ---                │
    # │ Utf8                           ┆ Float64            │
    # ╞════════════════════════════════╪════════════════════╡
    # │ tos://las-ai-qa-online/qa/tes… ┆ 0.5603389739990234 │
    # ╰────────────────────────────────┴────────────────────╯
