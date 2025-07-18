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
                "model": "doubao-1.5-thinking-vision-pro",
                "version": "250428",
                "multimodal_type": "video",
                "prompt": "视频里有什么？",
                "inference_type": "online",
            },
        )(col("videos")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────╮
    # │ videos                         ┆ llm_result                                              │
    # │ ---                            ┆ ---                                                     │
    # │ Utf8                           ┆ Utf8                                                    │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ 视频中呈现了一个**多层卡通蛋糕**（表面有红色糖霜、顶部… │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────╯
