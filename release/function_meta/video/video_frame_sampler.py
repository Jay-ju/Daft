from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video.video_frame_sampler import VideoFrameSampler

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    samples = {"video_path": [f"tos://{TOS_TEST_DIR}/video_frame_sampler/sample.mp4"]}

    ds = daft.from_pydict(samples)

    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_interval_frames",
            "interval_frames": 100,
            "output_tos_dir": f"tos://{TOS_TEST_DIR}/video_frame_sampler",
            "img_type": ".jpg",
        },
    )

    ds = ds.with_column("results", sampler(col("video_path")))

    ds = ds.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
        col("results").struct.get("tos_paths").alias("tos_paths"),
    )

    ds.show()
