from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.en_text_quality_scorer import EnTextQualityScorer
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {"text": ["This is a well-written scientific article about quantum physics.", None]}
    batch_size = 4
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "llm-data-textbook-quality-fasttext-classifier-v2/model_quantized.bin"

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "quality_score",
        las_udf(
            EnTextQualityScorer,
            construct_args={
                "batch_size": batch_size,
                "model_path": model_path,
                "model_name": model_name,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────────────────┬─────────────────────╮
    # │ text                                                                     ┆ quality_score       │
    # │ ---                                                                      ┆ ---                 │
    # │ Utf8                                                                     ┆ Float64             │
    # ╞══════════════════════════════════════════════════════════════════════════╪═════════════════════╡
    # │ This is a well-written scientific article about quantum physics.         ┆ 0.6918241381645203  │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                                                                     ┆ None                │
    # ╰──────────────────────────────────────────────────────────────────────────┴─────────────────────╯
