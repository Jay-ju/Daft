from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.multimodal.embedding.clip_embedding import ClipEmbedding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {"text": ["皮卡丘", "小狗", "小猫", None]}
    content_type = "text"
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "iic/multi-modal_clip-vit-base-patch16_zh"
    model_version = "v1.0.1"
    embedding_col_name = "embedding"
    batch_size = 2
    rank = 0
    num_gpus = 1

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ClipEmbedding,
            construct_args={
                "content_type": content_type,
                "model_path": model_path,
                "model_name": model_name,
                "model_version": model_version,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    ds.show()

    # ╭────────┬────────────────────────────────╮
    # │ text   ┆ embedding                      │
    # │ ---    ┆ ---                            │
    # │ Utf8   ┆ List[Float32]                  │
    # ╞════════╪════════════════════════════════╡
    # │ 皮卡丘  ┆ [0.12005615, -0.009140015, -0… │
    # ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 小狗    ┆ [0.12963867, 0.00039935112, 0… │
    # ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 小猫    ┆ [0.12670898, 0.015533447, 0.0… │
    # ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None   ┆ None                           │
    # ╰────────┴────────────────────────────────╯

    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"image_path": [f"tos://{TOS_TEST_DIR}/clip_embedding/cat_ip_adapter.jpeg"]}
    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ClipEmbedding,
            construct_args={
                "content_type": "image_url",
                "model_path": model_path,
                "model_name": model_name,
                "model_version": model_version,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬────────────────────────────────╮
    # │ image_path                     ┆ embedding                      │
    # │ ---                            ┆ ---                            │
    # │ Utf8                           ┆ List[Float32]                  │
    # ╞════════════════════════════════╪════════════════════════════════╡
    # │ tos://tos_bucket/clip_embeddi… ┆ [0.04598999, -0.090148926, -0… │
    # ╰────────────────────────────────┴────────────────────────────────╯
