from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.llava_one_vision_image_understanding import LlavaOneVisionImageUnderstanding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "image_path": [f"tos://{TOS_TEST_DIR}/llava_one_vision_image_understanding/cat_ip_adapter.jpeg"],
        "prompt": ["请给出图片的类型。"],
    }

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = os.getenv("MODEL_NAME", "LLaVA-OneVision-1.5-8B-Instruct")
    dtype = "bfloat16"
    use_flash_attention_2 = True
    prompt = None
    max_caption_length = 256
    resized_height = None
    resized_width = None
    batch_size = 2
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "caption",
        las_udf(
            LlavaOneVisionImageUnderstanding,
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
                "rank": rank,
            },
            num_gpus=1,
            batch_size=2,
            concurrency=1,
        )(col("image_path"), col("prompt")),
    )

    ds.collect()
    ds.show()
    print(ds.to_pandas()["caption"][0])

    # ╭────────────────────────────────┬────────────────────┬──────────────────────────────────────────────────────────╮
    # │ image_path                     ┆ prompt             ┆ caption                                                  │
    # │ ---                            ┆ ---                ┆ ---                                                      │
    # │ Utf8                           ┆ Utf8               ┆ Utf8                                                     │
    # ╞════════════════════════════════╪════════════════════╪══════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/llava_one_vi… ┆ 请给出图片的类型。    ┆ 这幅图片属于动画类型，具体来说是CGI（计算机生成图像）动…          │
    # ╰────────────────────────────────┴────────────────────┴──────────────────────────────────────────────────────────╯
