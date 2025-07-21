# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.clean_html_tag import CleanHtmlTag
from function_meta.meta import (
    OP_BUCKET,
    OP_ENVIRONMENT,
    OP_REGION,
    OP_VERSION,
    Category,
    DataItem,
    ExtraMetaModel,
    OpMetaModel,
    SubCategory,
    ValueType,
)


def get_meta() -> OpMetaModel:
    return OpMetaModel(
        Name="html 标签移除",
        Clazz=CleanHtmlTag,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_CLEAN,
        Tags=["文本清洗", "HTML"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/clean_html_tag/clean_html_tag.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对 html 标签进行移除。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""<!-- 这是 HTML 注释，不会在浏览器中显示 -->
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
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="Hello World!",
            Description="",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
