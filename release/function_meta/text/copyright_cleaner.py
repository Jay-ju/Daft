from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.copyright_cleaner import CopyrightCleaner
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "/* \n * Copyright (c) 2023 Jane Smith\n * 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n */ 你好",
            "# 版权 (c) 2023 Jane Smith\n# 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n 你好",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            CopyrightCleaner,
            construct_args={},
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                         ┆ cleaned_text                                │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Utf8                                         │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ /* \n * Copyright (c) 2023 Jane Smith\n * 本    ┆  你好                                        │
    # │ 代码依据 Apache License 2.0 授权，详见随附的     ┆                                             │
    # │ LICENSE 文件。\n */ 你好                        ┆                                             │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ # 版权 (c) 2023 Jane Smith\n# 本代码依据 Apache    ┆  你好                                        │
    # │ License 2.0 授权，详见随附的 LICENSE 文件。\n     ┆                                             │
    # │ 你好                                          ┆                                             │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
