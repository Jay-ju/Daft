from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.kimi_audio_understanding import KimiAudioUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/kimi_audio_understanding/黑神话悟空对话.mp3"]}

    audio_src_type = "audio_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "moonshotai/Kimi-Audio-7B-Instruct"
    prompt = """
    你是一名专业的音频分析助手。请全面理解输入音频，并按以下维度进行有条理的说明：
    内容概括：简要描述音频的主要内容、主题或事件。
    说话人信息：识别有多少个说话人，并简要描述他们的说话风格、性别或情绪。
    情绪与语气：总结音频中情绪状态及其变化（如平静、激动、紧张、愉快）。
    关键信息提取：提取音频中的核心信息点或指令，按时间顺序简要列出。
    背景环境：分析是否存在背景音乐、噪音或特殊声效，并描述其特征。
    请使用结构化自然语言输出，既简洁又完整，避免遗漏重要信息。
    """
    text_temperature = 0.0
    text_top_k = 5
    text_repetition_penalty = 1.0
    text_repetition_window_size = 16
    batch_size = 1
    rank = None

    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_analysis",
        las_udf(
            KimiAudioUnderstanding,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "prompt": prompt,
                "text_temperature": text_temperature,
                "text_top_k": text_top_k,
                "text_repetition_penalty": text_repetition_penalty,
                "text_repetition_window_size": text_repetition_window_size,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ audio_analysis                                              │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/kimi_audio_u… ┆ 内容概括：这段音频是一位老人在讲述自己的经历和感受，内容涉及…         │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
