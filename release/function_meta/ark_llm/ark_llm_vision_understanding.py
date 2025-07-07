from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_vision_understanding import ArkLLMVisionUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 配置模型的访问信息，包括ak和账号id，请根据实际情况修改
    access_key_id = "you_access_key_id"
    accound_id = "your_account_id"
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"videos": [f"tos://{TOS_TEST_DIR}/video_keyframes/sample.mp4"]}

    df = daft.from_pydict(samples)
    df = df.with_column(
        "llm_result",
        las_udf(
            ArkLLMVisionUnderstanding,
            construct_args={
                "model": "doubao-1.5-thinking-vision-pro",
                "version": "250428",
                "access_key": access_key_id,
                "account_id": accound_id,
                "multimodal_type": "video",
                "prompt": "视频里有什么？",
                "inference_type": "online",
            },
        )(col("videos")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬──────────────────────────────────────────────────╮
    # │ videos                         ┆ llm_result                                       │
    # │ ---                            ┆ ---                                              │
    # │ Utf8                           ┆ Utf8                                             │
    # ╞════════════════════════════════╪══════════════════════════════════════════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ 视频中呈现的是一位女性在户外演奏手风琴的场景：           │
    # │                                ┆                                                  │
    # │                                ┆ - …                                              │
    # ╰────────────────────────────────┴──────────────────────────────────────────────────╯
