# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.multimodal.kimi_audio_understanding import KimiAudioUnderstanding
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{tos_test_data_dir}/audio/黑神话悟空对话.mp3",
    ]
    return pd.DataFrame({"audio_path": paths})


@pytest.mark.skip(reason="Skip Kimi audio understanding test")
def test_kimi_audio_understanding(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "audio_analysis",
        las_udf(
            KimiAudioUnderstanding,
            construct_args={
                "audio_src_type": "audio_url",
                "model_path": local_models_dir,
                "model_name": "moonshotai/Kimi-Audio-7B-Instruct",
                "prompt": """
                你是一名专业的音频分析助手。请全面理解输入音频，并按以下维度进行有条理的说明：
                内容概括：简要描述音频的主要内容、主题或事件。
                说话人信息：识别有多少个说话人，并简要描述他们的说话风格、性别或情绪。
                情绪与语气：总结音频中情绪状态及其变化（如平静、激动、紧张、愉快）。
                关键信息提取：提取音频中的核心信息点或指令，按时间顺序简要列出。
                背景环境：分析是否存在背景音乐、噪音或特殊声效，并描述其特征。
                请使用结构化自然语言输出，既简洁又完整，避免遗漏重要信息。
                """,
                "batch_size": 1,
                "rank": 0,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    results = actual_df["audio_analysis"].tolist()
    assert results[0] == ""
    assert results[1] == ""
    assert len(results[2]) > 0
