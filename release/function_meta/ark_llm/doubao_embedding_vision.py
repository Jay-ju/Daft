from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_embedding_vision import DoubaoEmbeddingVision
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需提前配置环境变量 LAS_ACCESS_KEY 和 LAS_ACCOUND_ID ： LAS_ACCESS_KEY 是账号的 AK ， LAS_ACCOUND_ID 是账号 id
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "image_path": [f"tos://{TOS_TEST_DIR}/doubao_embedding_vision/cat_ip_adapter.jpeg"],
        "text": ["猫"],
    }

    df = daft.from_pydict(samples)
    df = df.with_column(
        "llm_result",
        las_udf(
            DoubaoEmbeddingVision,
            construct_args={
                "version": "250328",
                "image_format": "jpeg",
            },
        )(col("image_path"), col("text")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────────────────┬──────┬────────────────────────────────╮
    # │ image_path                     ┆ text ┆ llm_result                     │
    # │ ---                            ┆ ---  ┆ ---                            │
    # │ Utf8                           ┆ Utf8 ┆ List[Float32]                  │
    # ╞════════════════════════════════╪══════╪════════════════════════════════╡
    # │ https://las-ai-cn-beijing.tos… ┆ 猫   ┆ [0.019165039, 0.018798828, 0.… │
    # ╰────────────────────────────────┴──────┴────────────────────────────────
