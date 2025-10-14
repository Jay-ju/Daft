from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_vad_silero import AudioVadSilero
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_vad_silero/参观八达岭长城。.wav"]}

    model_path = os.getenv("MODEL_PATH", "./models")
    model_name = "silero-vad"
    audio_src_type = "audio_url"
    use_onnx_model = True
    onnx_model_revision = 16

    df = daft.from_pydict(samples)
    df = df.with_column(
        "audio_vad_result",
        las_udf(
            AudioVadSilero,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": model_path,
                "model_name": model_name,
                "use_onnx_model": use_onnx_model,
                "onnx_model_revision": onnx_model_revision,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )
    df.show()

    # ╭────────────────────────────────┬─────────────────────╮
    # │ audio_path                     ┆ audio_vad_result    │
    # │ ---                            ┆ ---                 │
    # │ Utf8                           ┆ List[List[Float32]] │
    # ╞════════════════════════════════╪═════════════════════╡
    # │tos://tos_bucket/audio_vad_sil… ┆ [[0.8, 2.4]]        │
    # ╰────────────────────────────────┴─────────────────────╯
