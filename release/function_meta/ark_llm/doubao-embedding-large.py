from __future__ import annotations

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_embedding_text import DoubaoEmbeddingText
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    samples = {"text": ["Hello World!", None]}

    df = daft.from_pydict(samples)
    # 计算文本的向量化数据
    df = df.with_column(
        "embeddings",
        las_udf(
            DoubaoEmbeddingText,
            construct_args={
                "model": "doubao-embedding-large",
            },
        )(col("text")),
    )

    df.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭──────────────┬────────────────────────────────╮
    # │ text         ┆ embeddings                     │
    # │ ---          ┆ ---                            │
    # │ Utf8         ┆ List[Float32]                  │
    # ╞══════════════╪════════════════════════════════╡
    # │ Hello World! ┆ [0.080078125, 1.8359375, 0.84… │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None         ┆ None                           │
    # ╰──────────────┴────────────────────────────────╯
