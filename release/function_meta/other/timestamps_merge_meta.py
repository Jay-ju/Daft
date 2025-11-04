# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.other import TimestampsMerge
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
        Name="时间戳片段合并",
        Clazz=TimestampsMerge,
        Category=Category.OTHER,
        SubCategory=SubCategory.OTHER,
        Tags=["时间戳合并", "语音", "视频", "后处理"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/timestamps_merge/timestamps_merge.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子对时间戳列表进行合并拆分等规范化操作。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="[[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]]",
            Description="处理之前的时间戳列表",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="[[0, 4.34], [5.5, 10.12]]",
            Description="处理之后的时间戳列表",
        ),
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
