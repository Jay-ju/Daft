from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.whitespace_normalizer import WhitespaceNormalizer
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "Hello\u2000World\u2001Test",
            "中文\u2002文本\u2003测试",
            "English\u2004中文\u2005Mixed",
            "Multiple\u2006\u2007\u2008Spaces",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ normalized_text                             │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Utf8                                         │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ Hello World Test                             ┆ Hello World Test                             │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 中文 文本 测试                                ┆ 中文 文本 测试                               │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ English 中文 Mixed                            ┆ English 中文 Mixed                            │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ Multiple   Spaces                            ┆ Multiple Spaces                              │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
