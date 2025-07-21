from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.embedding.image_vit_embedding import ImageViTEmbedding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"image": [f"tos://{TOS_TEST_DIR}/image_vit_embedding/cat_ip_adapter.png"]}

    image_src_type = "image_url"
    batch_size = 64
    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "google/vit-base-patch16-224-in21k"
    dtype = "float32"
    use_cls_token_embedding = True
    rank = 0
    num_gpus = 1

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ImageViTEmbedding,
            construct_args={
                "image_src_type": image_src_type,
                "batch_size": batch_size,
                "model_path": model_path,
                "model_name": model_name,
                "dtype": dtype,
                "use_cls_token_embedding": use_cls_token_embedding,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image")),
    )

    ds.show()
    df = ds.to_pandas()

    # ╭────────────────────────────────┬────────────────────────────────╮
    # │ image                          ┆ embedding                      │
    # │ ---                            ┆ ---                            │
    # │ Utf8                           ┆ List[Float32]                  │
    # ╞════════════════════════════════╪════════════════════════════════╡
    # │ tos://tos_bucket/image_vit_em… ┆ [-0.010721662, -0.018624008, … │
    # ╰────────────────────────────────┴────────────────────────────────╯
