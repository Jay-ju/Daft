from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio import AudioRiskRec
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 请在运行前设置好环境变量 APP_ID、BIZTYPE、TOS_TEST_DIR
    app_id = os.getenv("APP_ID")
    biztype = os.getenv("BIZTYPE")
    TOS_TEST_DIR_URL = os.getenv("TOS_TEST_DIR_URL", "tos_bucket")

    if not app_id or not biztype:
        raise ValueError("APP_ID and BIZTYPE environment variables must be set.")

    samples = {
        "audio_id": ["1"],
        "audio_path": [f"https://{TOS_TEST_DIR_URL}/audio_risk_rec/sample.mp3"],
    }
    df = daft.from_pydict(samples)

    constructor_kwargs = {"app_id": int(app_id), "biztype": biztype}

    df = df.with_column(
        "parsed_result",
        las_udf(AudioRiskRec, construct_args=constructor_kwargs, concurrency=1)(col("audio_id"), col("audio_path")),
    )
    df = df.with_column("Decision", col("parsed_result").struct.get("Decision"))
    df = df.with_column("Message", col("parsed_result").struct.get("Message"))
    df = df.with_column("risk_result", col("parsed_result").struct.get("risk_result"))

    df.select("audio_id", "audio_path", "Decision", "Message").show()
    # ╭──────────┬────────────────────────────────┬──────────┬─────────╮
    # │ audio_id ┆ audio_path                     ┆ Decision ┆ Message │
    # │ ---      ┆ ---                            ┆ ---      ┆ ---     │
    # │ Utf8     ┆ Utf8                           ┆ Utf8     ┆ Utf8    │
    # ╞══════════╪════════════════════════════════╪══════════╪═════════╡
    # │ 1        ┆ https://tos_bucket/audio_risk… ┆ PASS     ┆ success │
    # ╰──────────┴────────────────────────────────┴──────────┴─────────╯
