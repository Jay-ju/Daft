from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.url_ratio_calculator import UrlRatioCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "Hello world",
            "Check out https://example.com for more info",
            "Visit http://test.com and https://demo.org",
            "No URLs here, just text",
            "Mixed content: https://short.com and some text",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "url_ratio",
        las_udf(
            UrlRatioCalculator,
            construct_args={},
        )(col("text")),
    )

    ds.show()
    # ╭─────────────────────────────────────────┬──────────╮
    # │ text                                    ┆ url_ratio │
    # │ ---                                     ┆ ---       │
    # │ Utf8                                    ┆ Float64   │
    # ╞═════════════════════════════════════════╪══════════╡
    # │ Hello world                             ┆ 0.0       │
    # │ Check out https://example.com for m…    ┆ 0.4418604651162791 │
    # │ Visit http://test.com and https://d…    ┆ 0.7380952380952381 │
    # │ No URLs here, just text                 ┆ 0.0       │
    # │ Mixed content: https://short.com an…    ┆ 0.3695652173913043 │
    # ╰─────────────────────────────────────────┴──────────╯
