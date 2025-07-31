from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_speaker_verification_eres2net import AudioSpeakerVerificationEres2net
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {
        "speaker_a": [f"tos://{TOS_TEST_DIR}/audio_speaker_verification_eres2net/参观八达岭长城。.wav"],
        "speaker_b": [f"tos://{TOS_TEST_DIR}/audio_speaker_verification_eres2net/参观八达岭长城。.wav"],
    }

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "iic/speech_eres2net_sv_zh-cn_16k-common"
    audio_src_type = "audio_url"
    rank = 0

    df = daft.from_pydict(samples)
    df = df.with_column(
        "speaker_verification_result",
        las_udf(
            AudioSpeakerVerificationEres2net,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("speaker_a"), col("speaker_b")),
    )

    df.show()

    # ╭────────────────────────────────┬────────────────────────────────┬─────────────────────────────╮
    # │ speaker_a                      ┆ speaker_b                      ┆ speaker_verification_result │
    # │ ---                            ┆ ---                            ┆ ---                         │
    # │ Utf8                           ┆ Utf8                           ┆ Float32                     │
    # ╞════════════════════════════════╪════════════════════════════════╪═════════════════════════════╡
    # │ tos://tos_bucket/audio_speake… ┆ tos://tos_bucket/audio_speake… ┆ 1                           │
    # ╰────────────────────────────────┴────────────────────────────────┴─────────────────────────────╯
