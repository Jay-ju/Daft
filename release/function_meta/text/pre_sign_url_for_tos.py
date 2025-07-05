from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.pre_sign_url_for_tos import PreSignUrlForTos
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 样例数据
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"url": [f"tos://{TOS_TEST_DIR}/sample.mp4"]}

    # 使用提供的算子生成签名URL
    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "signed_url",
        las_udf(
            PreSignUrlForTos,
            construct_args={
                "expires": 3600,
            },
        )(col("url")),
    )

    ds.show()
    # 输出内容，以实际情况为准
    # ╭─────────────────────────────┬────────────────────────────────╮
    # │ url                         ┆ signed_url                     │
    # │ ---                         ┆ ---                            │
    # │ Utf8                        ┆ Utf8                           │
    # ╞═════════════════════════════╪════════════════════════════════╡
    # │ tos://tos_bucket/sample.mp4 ┆ https://tos_bucket.tos-cn-bei… │
    # ╰─────────────────────────────┴────────────────────────────────╯
