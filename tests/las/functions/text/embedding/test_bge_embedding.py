# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.embedding.bge_embedding import BgeEmbedding
from daft.las.functions.udf import las_udf

daft.set_execution_config(actor_udf_ready_timeout=600)
daft.set_execution_config(min_cpu_per_task=0)

samples = {"text": ["Hello World!", None]}
input_df = pd.DataFrame(samples)

dtype = "float16"
batch_size = 1

model_name = "BAAI/bge-large-zh-v1.5"
model_name = "BAAI/bge-large-en-v1.5"
model_name = "BAAI/bge-multilingual-gemma2"
model_name = "BAAI/bge-m3"

rank = 0
num_gpus = 1


def test_bge_embedding(local_models_dir):
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embeddings",
        las_udf(
            BgeEmbedding,
            construct_args={
                "dtype": dtype,
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embeddings"][0]) == 1024
    assert math.fabs(actual_df["embeddings"][0][0] + 0.042053223) < 0.001
    assert actual_df["embeddings"][1] is None
