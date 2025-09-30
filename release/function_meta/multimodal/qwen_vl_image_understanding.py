from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_image_understanding import QwenVLImageUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "image_path": [f"tos://{TOS_TEST_DIR}/qwen_vl_image_understanding/cat_ip_adapter.jpeg"],
        "prompt": ["请给出图片的类型。"],
    }

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    dtype = "bfloat16"
    use_flash_attention_2 = True
    default_prompt = None
    max_caption_length = 256
    resized_height = None
    resized_width = None
    batch_size = 2
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLImageUnderstanding,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": default_prompt,
                "max_caption_length": max_caption_length,
                "resized_height": resized_height,
                "resized_width": resized_width,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=2,
            concurrency=1,
        )(col("image_path"), col("prompt")),
    )

    ds.show()

    # ╭────────────────────────────────┬────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ image_path                     ┆ prompt             ┆ caption                                                     │
    # │ ---                            ┆ ---                ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8               ┆ Utf8                                                        │
    # ╞════════════════════════════════╪════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/qwen_vl_imag… ┆ 请给出图片的类型。    ┆ 这是一张卡通风格的图片，描绘了一只穿着人类服装的猫。猫站在…           │
    # ╰────────────────────────────────┴────────────────────┴─────────────────────────────────────────────────────────────╯
