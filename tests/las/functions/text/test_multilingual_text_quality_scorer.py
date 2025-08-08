# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import numpy as np
import pandas as pd

import daft
from daft import col
from daft.las.functions.text.multilingual_text_quality_scorer import MultilingualTextQualityScorer
from daft.las.functions.udf import las_udf


def test_multilingual_text_quality_scorer_basic(local_models_dir):
    samples = {
        "text": [
            "这是一篇关于人工智能技术发展的高质量学术论文，内容详实且具有很强的参考价值。",
            "これは量子物理学とその現代技術への応用に関するよく書かれた科学論文です。",
            "이것은 양자물리학과 현대 기술에의 응용에 관한 잘 쓰여진 과학 논문입니다.",
            None,
        ]
    }

    df = pd.DataFrame(samples)
    ds = daft.from_pandas(df)

    ds = ds.with_column(
        "quality_score",
        las_udf(
            MultilingualTextQualityScorer,
            construct_args={
                "model_path": local_models_dir,
                "model_name": "multilingual-e5-small-aligned-quality",
                "dtype": "float32",
                "batch_size": 4,
                "rank": 0,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 4
    assert isinstance(actual_df["quality_score"][0], np.float32)
    assert isinstance(actual_df["quality_score"][1], np.float32)
    assert isinstance(actual_df["quality_score"][2], np.float32)
    assert pd.isna(actual_df["quality_score"][3])

    assert 0.0 <= actual_df["quality_score"][0] <= 1.0
    assert 0.0 <= actual_df["quality_score"][1] <= 1.0
    assert 0.0 <= actual_df["quality_score"][2] <= 1.0
