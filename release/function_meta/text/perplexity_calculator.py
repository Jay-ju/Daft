from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.perplexity_calculator import PerplexityCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "人工智能技术正在快速发展，人工智能技术已经广泛应用于各个领域。",
            "这是一个正常的句子。",
            "乱码文本 12345 !@#$%",
        ]
    }
    lang = "zh"

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "perplexity",
        las_udf(
            PerplexityCalculator,
            construct_args={
                "lang": lang,
                "model_path": os.getenv("MODEL_PATH", "./models"),
                "model_name": "kenlm/wikipedia",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ perplexity                                  │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Float64                                      │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ 人工智能技术正在快速发展，人工智能技术已经广泛应        ┆ 335.9                                        │
    # │ 用于各个领域。                                ┆                                             │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 这是一个正常的句子。                           ┆ 530.3                                        │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 乱码文本 12345 !@#$%                           ┆ 9081.4                                       │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
