# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.multimodal.qwen_vl_image_understanding import QwenVLImageUnderstanding
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
        Name="图片内容理解（Qwen VL 系列模型）",
        Clazz=QwenVLImageUnderstanding,
        Category=Category.MULTI_MODAL,
        SubCategory=SubCategory.IMAGE_TO_TEXT,
        Tags=["图片理解", "多模态理解"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_vl_image_understanding/qwen_vl_image_understanding.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子理解图像内容，并按照指令生成描述。"
    before = [
        DataItem(
            Type=ValueType.Picture.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/qwen_vl_image_understanding/cat_ip_adapter.jpeg",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value="这张图片展示了一只拟人化的猫，它穿着一套复古风格的服装，包括一件蓝色的外套、棕色的背心和白色的衬衫，还系着一条黑色的领结。猫的耳朵竖立，眼睛大而明亮，显得非常可爱。\n\n背景是一个宁静的乡村场景，有一座茅草屋顶的小屋，周围环绕着茂密的绿色植物和盛开的花朵。小路两旁种满了各种各样的植物和花卉，远处可以看到树木和山丘，整个画面给人一种宁静和谐的感觉。",
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
