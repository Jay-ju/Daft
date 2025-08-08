from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.multilingual_text_quality_scorer import MultilingualTextQualityScorer
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。",
            "これは量子物理学とその現代技術への応用に関するよく書かれた科学論文です。",
            "이것은 양자물리학과 현대 기술에의 응용에 관한 잘 쓰여진 과학 논문입니다.",
            None,
        ]
    }

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "multilingual-e5-small-aligned-quality"
    dtype = "float32"
    batch_size = 100

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "quality_score",
        las_udf(
            MultilingualTextQualityScorer,
            construct_args={
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "batch_size": batch_size,
                "rank": 0,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────────────────┬─────────────────────╮
    # │ text                                                                     ┆ quality_score       │
    # │ ---                                                                      ┆ ---                 │
    # │ Utf8                                                                     ┆ Float32             │
    # ╞══════════════════════════════════════════════════════════════════════════╪═════════════════════╡
    # │ 这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。          ┆ 0.6294931           │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ これは量子物理学とその現代技術への応用に関するよく書かれた科学論文です。             ┆ 0.73835784          │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 이것은 양자물리학과 현대 기술에의 응용에 관한 잘 쓰여진 과학 논문입니다。                  ┆ 0.7729334           │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                                                                     ┆ None                │
    # ╰──────────────────────────────────────────────────────────────────────────┴─────────────────────╯
