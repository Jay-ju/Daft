from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.embedding.bge_sparse_dense_embedding import BgeSparseDenseEmbedding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {"text": ["Hello World!", None]}
    is_output_token_vec = True
    dtype = "float16"
    batch_size = 512
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "BAAI/bge-m3"
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "embeddings",
        las_udf(
            BgeSparseDenseEmbedding,
            construct_args={
                "is_output_token_vec": is_output_token_vec,
                "dtype": dtype,
                "batch_size": batch_size,
                "model_path": model_path,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
        )(col("text")),
    )

    ds.show()
    # ╭──────────────┬───────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ text         ┆ embeddings                                                                                    │
    # │ ---          ┆ ---                                                                                           │
    # │ Utf8         ┆ Struct[dense_embedding: List[Float32], sparse_embedding: Map[Utf8: Float32], token_embedding: │
    # │              ┆ List[List[Float32]]]                                                                          │
    # ╞══════════════╪═══════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ Hello World! ┆ {dense_embedding: [-0.0420532…                                                                │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None         ┆ {dense_embedding: None,                                                                       │
    # │              ┆ spars…                                                                                        │
    # ╰──────────────┴───────────────────────────────────────────────────────────────────────────────────────────────╯
