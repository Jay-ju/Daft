from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_video_understanding import QwenVLVideoUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"video_path": [f"tos://{TOS_TEST_DIR}/qwen_vl_video_understanding/eating_56.mp4"]}

    video_src_type = "video_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = "bfloat16"
    use_flash_attention_2 = True
    prompt = "请给出该视频的详细描述。"
    max_caption_length = 256
    min_pixels = 320 * 160
    max_pixels = 320 * 160
    fps = 1
    batch_size = 2
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLVideoUnderstanding,
            construct_args={
                "video_src_type": video_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=2,
        )(col("video_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ caption                                                     │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/qwen_vl_vide… ┆ 这是一段动画片段，画面中有一个卡通青蛙角色，它被设计成一个… │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
