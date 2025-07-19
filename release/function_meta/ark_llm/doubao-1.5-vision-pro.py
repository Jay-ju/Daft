from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"videos": [f"tos://{TOS_TEST_DIR}/ark_llm_vision_understanding/eating_56.mp4"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "llm_result",
        las_udf(
            ArkLLMVisionUnderstanding,
            construct_args={
                "model": "doubao-1.5-vision-pro",
                "version": "250328",
                "multimodal_type": "video",
                "prompt": "视频里有什么？",
                "inference_type": "batch",
            },
        )(col("videos")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ videos                         ┆ llm_result                                                  │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/op… ┆  视频中展示了一个卡通风格的场景。首先，画面中央是一个装饰有…          │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
