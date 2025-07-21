from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.clean_html_tag import CleanHtmlTag
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
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
    ds = daft.from_pydict(samples)

    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            CleanHtmlTag,
            construct_args={"separator": separator, "strip": strip},
        )(col("text")),
    )
    ds.show()

    # ╭──────────────────────────────────────────────┬──────────────╮
    # │ text                                         ┆ cleaned_text │
    # │ ---                                          ┆ ---          │
    # │ Utf8                                         ┆ Utf8         │
    # ╞══════════════════════════════════════════════╪══════════════╡
    # │ <!-- 这是 HTML 注释，不会在浏览器中显示 -->…      ┆ Hello World! │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                                         ┆ None         │
    # ╰──────────────────────────────────────────────┴──────────────╯
