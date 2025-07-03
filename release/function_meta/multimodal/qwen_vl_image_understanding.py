from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_image_understanding import QwenVLImageUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"image_path": [f"tos://{TOS_TEST_DIR}/qwen_vl_image_understanding/cat_ip_adapter.jpeg"]}

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = "bfloat16"
    use_flash_attention_2 = True
    prompt = "请给出这张图片的详细描述。"
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
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "resized_height": resized_height,
                "resized_width": resized_width,
                "batch_size": batch_size,
                "rank": 0,
            },
            num_gpus=1,
            batch_size=2,
        )(col("image_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬─────────────────────────────────────────────────────────────╮
    # │ image_path                     ┆ caption                                                     │
    # │ ---                            ┆ ---                                                         │
    # │ Utf8                           ┆ Utf8                                                        │
    # ╞════════════════════════════════╪═════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/qwen_vl_imag… ┆ 这张图片展示了一只拟人化的猫，它穿着一套复古风格的服装，包…           │
    # ╰────────────────────────────────┴─────────────────────────────────────────────────────────────╯
