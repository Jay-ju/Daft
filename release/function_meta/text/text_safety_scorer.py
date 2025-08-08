from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.text_safety_scorer import TextSafetyScorer
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "爱与和平是世界的主旋律。",
            "我讨厌所有人，最好都去死。",
            None,
        ]
    }

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "thu-coai/ShieldLM-6B-chatglm3"
    lang = "zh"
    batch_size = 3
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "safety_scores",
        las_udf(
            TextSafetyScorer,
            construct_args={
                "lang": lang,
                "model_path": model_path,
                "model_name": model_name,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=3,
            concurrency=1,
        )(col("text")),
    )

    ds.show()
    # ╭─────────────────────────────┬────────────────────────────────────────────────────────────────╮
    # │ text                        ┆ safety_scores                                                  │
    # │ ---                         ┆ ---                                                            │
    # │ Utf8                        ┆ Struct[safe: Float64, unsafe: Float64, controversial: Float64] │
    # ╞═════════════════════════════╪════════════════════════════════════════════════════════════════╡
    # │ 爱与和平是世界的主旋律。        ┆ {safe: 0.8632398843765259,                                     │
    # │                             ┆ unsafe: 0.04434245824813843,                                   │
    # │                             ┆ controversial: 0.09241761267185211}                            │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 我讨厌所有人，最好都去死。      ┆ {safe: 0.0004654618678614497,                                  │
    # │                             ┆ unsafe: 0.9993947744369507,                                    │
    # │                             ┆ controversial: 0.00013975700130686164}                         │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                        ┆ None                                                           │
    # ╰─────────────────────────────┴────────────────────────────────────────────────────────────────╯
