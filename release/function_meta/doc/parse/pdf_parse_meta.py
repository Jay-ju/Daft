# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from daft.las.functions.doc import PDFParse
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
        Name="PDF 文档智能解析",
        Clazz=PDFParse,
        Category=Category.DOC,
        SubCategory=SubCategory.DOC_PARSE,
        Tags=["pdf", "文档解析", "文本提取", "ocr"],
    )


def get_extra_meta() -> ExtraMetaModel:
    code = f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/pdf_parse/pdf_parse.py"
    code_description = "下面的代码展示了如何使用 Daft（适用于分布式）运行算子解析pdf文档。"
    before = [
        DataItem(
            Type=ValueType.File.name,
            Value=f"https://{OP_BUCKET}.tos-{OP_REGION}.volces.com/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/pdf_parse/sample.pdf",
            Description="",
        ),
    ]
    after = [
        DataItem(
            Type=ValueType.Text.name,
            Value='我的小狗 \n我家有一只可爱的小狗,它的名字叫小白｡小白是一只白色的泰迪犬,它有一双圆圆的大眼睛,像两 颗黑珍珠一样闪闪发光｡它的鼻子小小的,湿湿的,闻起来总是很灵敏｡小白的耳朵很长,走起路来 一甩一甩的,特别有趣｡ \n小白非常聪明,每次我放学回家,它都会跑到门口迎接我,用它的小爪子轻轻地拍打我的腿,好像在 说:"欢迎回家!"我每次看到它,心情都会变得特别好｡ \n小白最喜欢吃骨头和狗粮,每次我给它喂食的时候,它都会开心地摇着尾巴,围着我转圈圈｡它吃东 西的时候特别可爱,总是先用鼻子闻一闻,然后才开始小口小口地吃｡ \n小白也很喜欢玩耍｡每到周末,我都会带它去公园玩｡它喜欢在草地上奔跑,追逐蝴蝶和小鸟｡有时 候,我会和它玩捡球的游戏,我把球扔出去,它就会飞快地跑过去,把球叼回来,放在我的脚边,等 着我再扔一次｡ \n小白不仅是我的好朋友,还是我的小帮手｡有一次,我的作业本掉在了床底下,我怎么也够不到｡小 白看到后,马上跑过来,用它的小爪子把作业本拨了出来,真是帮了我一个大忙｡ \n我非常喜欢我的小狗小白,它给我的生活带来了很多快乐和温暖｡我希望它能一直陪伴在我的身边, 和我一起成长｡',
            Description="",
        )
    ]

    return ExtraMetaModel(
        BeforeData=before,
        AfterData=after,
        Code=code,
        CodeDescription=code_description,
        Published=True,
    )
