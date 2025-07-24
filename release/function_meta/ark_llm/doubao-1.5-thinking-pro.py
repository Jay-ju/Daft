from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_thinking_vision import ArkLLMThinkingVision
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    queries = {
        "query": [
            "帮我规划5月去新疆的10天旅行安排",
        ]
    }

    df = daft.from_pydict(queries)

    df = df.with_column(
        "llm_result",
        las_udf(
            ArkLLMThinkingVision,
            construct_args={
                "model": "doubao-1.5-thinking-pro",
                "version": "250415",
                "multimodal_type": "text",
                "inference_type": "online",
            },
        )(col("query")),
    )

    df = df.with_column("reasoning_content", col("llm_result").struct.get("reasoning_content"))
    df = df.with_column("llm_result", col("llm_result").struct.get("llm_result"))

    output_pd_df = df.to_pandas()
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭──────────────────────────────────┬──────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────╮
    # │ query                            ┆ llm_result                                           ┆ reasoning_content                                        │
    # │ ---                              ┆ ---                                                  ┆ ---                                                      │
    # │ Utf8                             ┆ Utf8                                                 ┆ Utf8                                                     │
    # ╞══════════════════════════════════╪══════════════════════════════════════════════════════╪══════════════════════════════════════════════════════════╡
    # │ 帮我规划5月去新疆的10天旅行安排… ┆                                                      ┆ 好的，用户让我帮忙规划一个5月份去新疆的10天旅行安排。首… │
    # │                                  ┆                                                      ┆                                                          │
    # │                                  ┆ 5 月是新疆旅行的 “初夏黄金期”，草原渐绿、花海初绽… ┆                                                          │
    # ╰──────────────────────────────────┴──────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────╯
