# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.text.multilingual_text_translate import MultilingualTextTranslate
from daft.las.functions.udf import las_udf

model_name = "Seed-X-Instruct-7B"
max_model_len = 2048
max_num_seqs = 128

tensor_parallel_size = 1
enable_prefix_caching = True
gpu_memory_utilization = 0.9
use_cot = False
source_language = "Chinese"
target_language = "English"
max_tokens = 1024
batch_size = 1
dtype = "bfloat16"
seed = 42


@pytest.mark.skip(reason="""T4 GPU not support Flash Attention 2.""")
def test_multilingual_text_translate(local_models_dir):
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
        "translate_text",
        las_udf(
            MultilingualTextTranslate,
            construct_args={
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "gpu_memory_utilization": gpu_memory_utilization,
                "use_cot": use_cot,
                "source_language": source_language,
                "target_language": target_language,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "seed": seed,
            },
            num_gpus=tensor_parallel_size,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 4
    assert "high-quality" in actual_df["translate_text"][0]
    assert "well-written" in actual_df["translate_text"][1]
