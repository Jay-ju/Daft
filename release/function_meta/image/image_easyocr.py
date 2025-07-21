from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.image_easyocr import ImageEasyOcr
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"image": [f"tos://{TOS_TEST_DIR}/image_easyocr/通用场景图片.jpeg"]}

    image_src_type = "image_url"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "EasyOCR"
    quantize = True
    lang_list = ["en", "ch_sim"]
    batch_size = 16
    num_gpus = 1

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "ocr_result",
        las_udf(
            ImageEasyOcr,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "quantize": quantize,
                "lang_list": lang_list,
                "batch_size": batch_size,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image")),
    )
    ds.show()
    df = ds.to_pandas()

    # ╭────────────────────────────────┬───────────────────╮
    # │ image                          ┆ ocr_result        │
    # │ ---                            ┆ ---               │
    # │ Utf8                           ┆ Utf8              │
    # ╞════════════════════════════════╪═══════════════════╡
    # │ tos://tos_bucket/image_easyoc… ┆ 不论结局           │
    # │                                ┆ 我己经很感谢相遇…    │
    # ╰────────────────────────────────┴───────────────────╯
