from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.alphanumeric_ratio_calculator import AlphanumericRatioCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "HelloWorld123",
            "Hello, world!",
            "!!!@@@###$$$",
            "Test 123! Is it working?",
            "你好Hello123",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "alphanumeric_ratio",
        las_udf(
            AlphanumericRatioCalculator,
            construct_args={"tokenization": False},
        )(col("text")),
    )

    ds.show()
    # ╭─────────────────────────┬─────────────────────╮
    # │ text                    ┆ alphanumeric_ratio  │
    # │ ---                     ┆ ---                 │
    # │ Utf8                    ┆ Float64             │
    # ╞═════════════════════════╪═════════════════════╡
    # │ HelloWorld123           ┆ 1.0                 │
    # │ Hello, world!           ┆ 0.7692307692307693  │
    # │ !!!@@@###$$$            ┆ 0.0                 │
    # │ Test 123! Is it work…   ┆ 0.75                │
    # │ 你好Hello123            ┆ 1.0                 │
    # ╰─────────────────────────┴─────────────────────╯
