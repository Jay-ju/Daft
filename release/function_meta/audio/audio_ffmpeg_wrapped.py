from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.audio.audio_ffmpeg_wrapped import AudioFFMPEGWrapped
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"audio_path": [f"tos://{TOS_TEST_DIR}/audio_ffmpeg_wrapped/sample.mp3"]}

    ds = daft.from_pydict(samples)

    splitter = las_udf(
        AudioFFMPEGWrapped,
        construct_args={
            "filter_name": "volume",
            "filter_kwargs": {"volume": 1.5},
            "global_args": ["-loglevel", "error", "-hide_banner", "-y"],
            "output_tos_dir": f"tos://{TOS_TEST_DIR}/audio_ffmpeg_wrapped",
            "output_audio_binary": False,
            "output_audio_format": "wav",
        },
    )

    ds = ds.with_column("results", splitter(col("audio_path")))

    ds.show()
    # ╭─────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    # │ audio_path                              ┆ results                                                                                                                            │
    # │ ---                                     ┆ ---                                                                                                                                │
    # │ Utf8                                    ┆ Struct[processed_audio_path: Utf8, processed_audio_binary: Binary]                                                                 │
    # ╞═════════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
    # │ tos://tos_bucket/audio_ffmpeg_wrapped/… ┆ {processed_audio_path: "tos://tos_bucket/audio_ffmpeg_wrapped/sample_processed.wav", processed_audio_binary: null}                 │
    # ╰─────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
