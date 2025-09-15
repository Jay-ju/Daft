from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.qwen_omni_audio_understanding import QwenOmniAudioUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/qwen_omni_audio_understanding/sample.mp3"]}

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "Qwen/Qwen2.5-Omni-7B"
    dtype = "bfloat16"
    use_flash_attention_2 = True
    prompt = "请直接将这个音频转换成文字，不要做任何解释。"
    max_caption_length = 256
    batch_size = 1
    rank = 0
    num_gpus = int(os.getenv("NUM_GPUS", 1))

    ds = daft.from_pydict(samples)

    ds = ds.with_column(
        "caption",
        las_udf(
            QwenOmniAudioUnderstanding,
            construct_args={
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    print(ds.to_pandas()["caption"][0])

    ds.show()

    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ audio_path                     ┆ caption                                                     │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/qwen_omni_au… ┆ 人我保住了，经我取到了。俺老孙啥功名不要，只求回到这花果山…           │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
