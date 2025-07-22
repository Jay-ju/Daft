from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.word_repetition_calculator import WordRepetitionCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "人工智能技术正在快速发展，人工智能技术在各领域的应用越来越广泛。人工智能技术可以帮助我们解决复杂问题，人工智能技术的未来充满无限可能。",
            "这是一个没有重复词汇的文本示例，每个词语都只出现一次。",
            None,
        ]
    }
    repetition = 2
    lang = "zh"
    tokenization = True

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "repetition_ratio",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": repetition,
                "lang": lang,
                "tokenization": tokenization,
                "model_path": os.getenv("MODEL_PATH", "./models"),
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ repetition_ratio                            │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Float64                                      │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ 人工智能技术正在快速发展，人工智能技术在各领域的    ┆ 0.14285714285714285                         │
    # │ 应用越来越广泛。人工智能技术可以帮助我们解决复杂问题  ┆                                              │
    # │ ，人工智能技术的未来充满无限可能。              ┆                                              │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 这是一个没有重复词汇的文本示例，每个词语都只出现  ┆ 0.0                                         │
    # │ 一次。                                        ┆                                              │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                                         ┆ None                                         │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
