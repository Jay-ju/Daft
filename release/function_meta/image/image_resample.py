from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.image_resample import ImageResample
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"image": [f"tos://{TOS_TEST_DIR}/image_resample/cat_ip_adapter.png"]}

    image_suffix = ".jpg"
    tos_dir = f"tos://{TOS_TEST_DIR}/image_resample/"
    image_src_type = "image_url"
    target_size = (200, 200)
    target_dpi = (72, 72)
    method = "lanczos"
    local_output = ""
    num_gpus = 1

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "image_resample",
        las_udf(
            ImageResample,
            construct_args={
                "image_suffix": image_suffix,
                "tos_dir": tos_dir,
                "local_output": local_output,
                "image_src_type": image_src_type,
                "target_size": target_size,
                "target_dpi": target_dpi,
                "method": method,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image")),
    )

    ds.show()
    df = ds.to_pandas()

    # ╭────────────────────────────────┬────────────────────────────────────────╮
    # │ image                          ┆ image_resample                         │
    # │ ---                            ┆ ---                                    │
    # │ Utf8                           ┆ Struct[base64: Utf8, image_path: Utf8] │
    # ╞════════════════════════════════╪════════════════════════════════════════╡
    # │ tos://tos_bucket/image_resamp… ┆ {base64: iVBORw0KGgoAAAANSUhE…         │
    # ╰────────────────────────────────┴────────────────────────────────────────╯
