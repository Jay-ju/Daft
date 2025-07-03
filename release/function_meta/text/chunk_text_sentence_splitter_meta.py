# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.text.chunk_text_sentence_splitter import ChunkTextSentenceSplitter
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
        Name="文本 chunk 切分（基于句子结构）",
        Clazz=ChunkTextSentenceSplitter,
        Category=Category.TEXT,
        SubCategory=SubCategory.TEXT_PROCESSING,
        Tags=["文本切分"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/chunk_text/chunk_text.py"
    code_description = "下面的代码展示了如何使用 daft 运行算子按照文本句子结构对其做切分。"
    before = [
        DataItem(
            Type=ValueType.Text.name,
            Value="""2023年，中国乘用车市场零售总量恢复至疫情前的水平，显示出市场的强劲复苏。新能源乘用车行业同年实现了高质量增长，全年销量超过770万辆，平均每三辆新售乘用车中就有一辆为新能源车型。在政策支持和市场供需的共同推动下，新能源车市场展现出更多市场化特征。预计2024年，这一增长趋势将持续。随着新能源技术和商业模式的不断成熟，乘用车产品正逐步迈入“体验型商品”阶段，消费者在购车时更加关注整体体验，而不仅仅是车辆的社会属性或耐用品属性。
与此同时，经过多年的快速发展和普及，2023年我国新车金融渗透率出现了首次回落，整体渗透率为56%，较上一年下降2%，新能源车的金融渗透率略低于整体水平。市场竞争日益激烈，加之宏观金融环境的影响，汽车金融领域的价格战愈发激烈。此外，产品同质化、需求多样化满足不足以及渠道模式固化等问题依然存在。研究认为，围绕用户综合体验进行汽车金融服务转型，将成为行业突破瓶颈、实现差异化发展的关键方向。""",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.List.name,
            Value=[
                "2023年，中国乘用车市场零售总量恢复至疫情前的水平，显示出市场的强劲复苏。新能源乘用车行业同年实现了高质量增长，全年销量超过770万辆，平均每三辆新售乘用车中就有一辆为新能源车型。在政策支持和市场供需的共同推动下，",
                "在政策支持和市场供需的共同推动下，新能源车市场展现出更多市场化特征。预计2024年，这一增长趋势将持续。随着新能源技术和商业模式的不断成熟，乘用车产品正逐步迈入“体验型商品”阶段，消费者在购车时更加关注整体体验，",
                "消费者在购车时更加关注整体体验，而不仅仅是车辆的社会属性或耐用品属性。\n与此同时，经过多年的快速发展和普及，2023年我国新车金融渗透率出现了首次回落，整体渗透率为56%，较上一年下降2%，新能源车的金融渗透率略低于整体水平。",
                "新能源车的金融渗透率略低于整体水平。市场竞争日益激烈，加之宏观金融环境的影响，汽车金融领域的价格战愈发激烈。此外，产品同质化、需求多样化满足不足以及渠道模式固化等问题依然存在。研究认为，",
                "研究认为，围绕用户综合体验进行汽车金融服务转型，将成为行业突破瓶颈、实现差异化发展的关键方向。",
            ],
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
