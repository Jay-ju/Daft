# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.en_text_quality_scorer import EnTextQualityScorer
from daft.las.functions.udf import las_udf

samples = {"text": ["This is a well-written scientific article about quantum physics.", None]}
input_df = pd.DataFrame(samples)

batch_size = 1
model_name = "llm-data-textbook-quality-fasttext-classifier-v2/model_quantized.bin"
num_gpus = 0


def test_en_text_quality_scorer(local_models_dir):
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "quality_score",
        las_udf(
            EnTextQualityScorer,
            construct_args={
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert isinstance(actual_df["quality_score"][0], float)
    assert 0.0 <= actual_df["quality_score"][0] <= 2.0
    assert pd.isna(actual_df["quality_score"][1])
