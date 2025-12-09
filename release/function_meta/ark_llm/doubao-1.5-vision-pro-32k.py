from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"images": [f"tos://{TOS_TEST_DIR}/ark_llm_vision_understanding/cat_ip_adapter.jpeg"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "llm_result",
        las_udf(
            ArkLLMVisionUnderstanding,
            construct_args={
                "model": "doubao-1.5-vision-pro-32k",
                "system_text": "你是一个专业的图片理解助手，能够分析图片中的内容并提供详细的描述。",
                "inference_type": "online",
            },
        )(images=col("images")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ images                         ┆ llm_result                                                  │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/op… ┆ 图中是一只拟人化的猫，毛色为浅棕色和白色相间，有着大大的蓝…           │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
