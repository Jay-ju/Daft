from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.repeated_lines_calculator import RepeatedLinesCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "第一行内容\n第二行内容\n第一行内容\n第三行内容",
            "第一行内容\n第一行内容\n第一行内容",
            "第一行内容\n\n第二行内容\n   \n第三行内容",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "repeated_ratio",
        las_udf(
            RepeatedLinesCalculator,
            construct_args={},
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ repeated_ratio                              │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Float64                                      │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ 第一行内容\n第二行内容\n第一行内容\n第三行内容        ┆ 0.25                                        │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 第一行内容\n第一行内容\n第一行内容                ┆ 0.6666666666666666                         │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 第一行内容\n\n第二行内容\n   \n第三行内容        ┆ 0.0                                         │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
