# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.embedding.bge_sparse_dense_embedding import BgeSparseDenseEmbedding
from daft.las.functions.udf import las_udf

samples = {"text": ["Hello World!", None]}
input_df = pd.DataFrame(samples)

is_output_token_vec = True
dtype = "float16"
batch_size = 1
model_name = "BAAI/bge-m3"
rank = 0
num_gpus = 0


def test_bge_sparse_dense_embedding(local_models_dir):
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embeddings",
        las_udf(
            BgeSparseDenseEmbedding,
            construct_args={
                "is_output_token_vec": is_output_token_vec,
                "dtype": dtype,
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embeddings"][0]["dense_embedding"]) == 1024
    assert math.fabs(actual_df["embeddings"][0]["dense_embedding"][0] + 0.04205322) < 0.001
    assert actual_df["embeddings"][1]["dense_embedding"] is None
    assert "Hello" in actual_df["embeddings"][0]["sparse_embedding"][0]
    assert (
        actual_df["embeddings"][0]["token_embedding"] is None
        or math.fabs(actual_df["embeddings"][0]["token_embedding"][0][0] - 0.00668225) < 0.001
    )
