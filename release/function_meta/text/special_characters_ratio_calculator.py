from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.special_characters_ratio_calculator import SpecialCharactersRatioCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "Hello world!",
            "1234567890",
            "     ",
            "这是中文文本",
            "你好 Hello 😊 123 !!!",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "special_ratio",
        las_udf(
            SpecialCharactersRatioCalculator,
            construct_args={"character_type": "all"},
        )(col("text")),
    )

    ds.show()
    # ╭───────────────────────┬─────────────────────╮
    # │ text                  ┆ special_ratio       │
    # │ ---                   ┆ ---                 │
    # │ Utf8                  ┆ Float64             │
    # ╞═══════════════════════╪═════════════════════╡
    # │ Hello world!          ┆ 0.16666666666666666 │
    # │ 1234567890            ┆ 1.0                 │
    # │      (5 spaces)       ┆ 1.0                 │
    # │ 这是中文文本            ┆ 0.0                 │
    # │ 你好 Hello 😊 123 !!!  ┆ 0.6111111111111112  │
    # ╰───────────────────────┴─────────────────────╯
