from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_thinking_vision import ArkLLMThinkingVision
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"videos": [f"tos://{TOS_TEST_DIR}/ark_llm_thinking_vision/eating_56.mp4"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "llm_result",
        las_udf(
            ArkLLMThinkingVision,
            construct_args={
                "model": "doubao-seed-1.6",
                "version": "250615",
                "multimodal_type": "video",
                "prompt": "视频里有什么？",
                "inference_type": "online",
            },
        )(col("videos")),
    )

    df = df.with_column("reasoning_content", col("llm_result").struct.get("reasoning_content"))
    df = df.with_column("llm_result", col("llm_result").struct.get("llm_result"))

    output_pd_df = df.to_pandas()
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬────────────────────────────────────────────┬────────────────────────────────────────────╮
    # │ videos                         ┆ llm_result                                 ┆ reasoning_content                          │
    # │ ---                            ┆ ---                                        ┆ ---                                        │
    # │ Utf8                           ┆ Utf8                                       ┆ Utf8                                       │
    # ╞════════════════════════════════╪════════════════════════════════════════════╪════════════════════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/op… ┆ 视频中呈现的是一段动画内容：起初展示的是一        ┆ 用户现在需要描述视频里的内容。首先看画面：        │
    # │                                ┆ 个**多层卡通风…                              ┆ 开头是一个多层蛋…                             │
    # ╰────────────────────────────────┴────────────────────────────────────────────┴────────────────────────────────────────────╯
