from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.text_length_calculator import TextLengthCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "Hello World",
            "你好世界",
            "Python编程 is fun",
            None,
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "length",
        las_udf(
            TextLengthCalculator,
            construct_args={},
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ length                                       │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Int64                                        │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ Hello World                                  ┆ 11                                           │
    # │ 你好世界                                      ┆ 4                                            │
    # │ Python编程 is fun                             ┆ 15                                           │
    # │ None                                         ┆ None                                            │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
