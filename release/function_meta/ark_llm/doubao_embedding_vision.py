from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_embedding_vision import DoubaoEmbeddingVision
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "image_path": [f"tos://{TOS_TEST_DIR}/doubao_embedding_vision/cat_ip_adapter.jpeg"],
        "text": ["猫"],
    }

    df = daft.from_pydict(samples)
    # 计算图片和文本的向量化数据
    df = df.with_column(
        "embeeding_for_image_text",
        las_udf(
            DoubaoEmbeddingVision,
            construct_args={
                "image_format": "jpeg",
            },
        )(col("image_path"), col("text")),
    )

    # 计算图片向量化数据
    df = df.with_column(
        "embeeding_for_image",
        las_udf(
            DoubaoEmbeddingVision,
            construct_args={
                "image_format": "jpeg",
            },
        )(col("image_path")),
    )

    # 计算文本向量化数据
    df = df.with_column(
        "embeeding_for_text",
        las_udf(
            DoubaoEmbeddingVision,
            construct_args={
                "multimodal_type": "text",
            },
        )(col("text")),
    )
    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭──────────────────────────────┬──────┬─────────────────────────────┬─────────────────────┬─────────────────────────────╮
    # │ image_path                   ┆ text ┆ embeeding_for_image_text    ┆ embeeding_for_image ┆ embeeding_for_text          │
    # │ ---                          ┆ ---  ┆ ---                         ┆ ---                 ┆ ---                         │
    # │ Utf8                         ┆ Utf8 ┆ List[Float32]               ┆ List[Float32]       ┆ List[Float32]               │
    # ╞══════════════════════════════╪══════╪═════════════════════════════╪═════════════════════╪═════════════════════════════╡
    # │ tos://las-ai-cn-beijing/qa/o ┆ 猫   ┆ [0.01953125, 0.017089844,   ┆ [0.047607422,       ┆ [0.036376953, 0.017333984,  │
    # │ p…                           ┆      ┆ 0.0…                        ┆ 0.044433594, -0…    ┆ 0.…                         │
    # ╰──────────────────────────────┴──────┴─────────────────────────────┴─────────────────────┴─────────────────────────────╯
