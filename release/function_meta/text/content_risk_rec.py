from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text import ContentRiskRec
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # Please set the following environment variables before running
    app_id = os.getenv("APP_ID")
    biztype = os.getenv("BIZTYPE")

    if not app_id or not biztype:
        raise ValueError("APP_ID and BIZTYPE environment variables must be set.")

    samples = {
        "id": [1, 2],
        "text": [
            "今天天气真好，我们一起去公园玩吧！",
            "出售枪支，联系电话123456789",
        ],
        "account_id": ["user1", "user2"],
        "nickname": ["用户1", "用户2"],
    }
    df = daft.from_pydict(samples)

    constructor_kwargs = {"app_id": int(app_id), "biztype": biztype}

    # Use Daft for distributed processing
    df = df.with_column(
        "parsed_result",
        las_udf(ContentRiskRec, construct_args=constructor_kwargs, concurrency=1)(
            col("text"), account_id_col=col("account_id"), nick_name_col=col("nickname")
        ),
    )
    df = df.with_column("FinalLabel", col("parsed_result").struct.get("FinalLabel"))
    df = df.with_column("Decision", col("parsed_result").struct.get("Decision"))

    df.select("text", "account_id", "nickname", "FinalLabel", "Decision").show()

    # ╭─────────────────────────────────────┬────────────┬──────────┬────────────┬──────────╮
    # │ text                                ┆ account_id ┆ nickname ┆ FinalLabel ┆ Decision │
    # │ ---                                 ┆ ---        ┆ ---      ┆ ---        ┆ ---      │
    # │ Utf8                                ┆ Utf8       ┆ Utf8     ┆ Utf8       ┆ Utf8     │
    # ╞═════════════════════════════════════╪════════════╪══════════╪════════════╪══════════╡
    # │ 今天天气真好，我们一起去公园玩吧！…       ┆ user1      ┆ 用户1    ┆            ┆ PASS     │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
    # │ 出售枪支，联系电话123456789…           ┆ user2      ┆ 用户2     ┆ 106        ┆ BLOCK    │
    # ╰─────────────────────────────────────┴────────────┴──────────┴────────────┴──────────╯
