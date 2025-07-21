# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.clean_html_tag import CleanHtmlTag
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        """<!-- 这是 HTML 注释，不会在浏览器中显示 -->
    <!DOCTYPE html>
    <html>
    <head>
        <!-- 引入外部 CSS -->
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <!--
            多行注释示例：
            保持代码缩进一致（如 2/4 空格）。
            标签属性使用双引号。
        -->
        <div class="content" id="main-content">
            <p class="text">Hello World!</p>
        </div>
    </body>
    </html>""",
        None,
    ]
}

separator = "\n"
strip = True
input_df = pd.DataFrame(samples)


def test_clean_html_tag():
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            CleanHtmlTag,
            construct_args={"separator": separator, "strip": strip},
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["cleaned_text"][1] is None
    assert actual_df["cleaned_text"][0] == "Hello World!"
