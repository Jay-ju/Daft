from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video.video_nsfw_detect import VideoNsfwDetect

if __name__ == "__main__":
    if os.getenv("DAFT_RUNNER", "native") == "ray":
        import logging

        import ray

        def configure_logging():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S.%s".format(),
            )
            logging.getLogger("tracing.span").setLevel(logging.WARNING)
            logging.getLogger("daft_io.stats").setLevel(logging.WARNING)
            logging.getLogger("DaftStatisticsManager").setLevel(logging.WARNING)
            logging.getLogger("DaftFlotillaScheduler").setLevel(logging.WARNING)
            logging.getLogger("DaftFlotillaDispatcher").setLevel(logging.WARNING)

        ray.init(dashboard_host="0.0.0.0", runtime_env={"worker_process_setup_hook": configure_logging})
        daft.context.set_runner_ray()

    daft.set_execution_config(actor_udf_ready_timeout=600)
    daft.set_execution_config(min_cpu_per_task=0)

    samples = {
        "video_path": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_video_dataset/music_sample.mp4"
        ]
    }

    video_src_type = "video_url"
    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = "Falconsai/nsfw_image_detection"
    num_gpus = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            VideoNsfwDetect,
            construct_args={
                "model_path": model_path,
                "video_src_type": video_src_type,
                "sample_mode": "by_count_uniform",
                "count_k": 3,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬──────────╮
    # │ video_path                     ┆ nsfw     │
    # │ ---                            ┆ ---      │
    # │ Utf8                           ┆ Float64  │
    # ╞════════════════════════════════╪══════════╡
    # │ https://las-ai-qa-online.tos-… ┆ 0.000428 │
    # ╰────────────────────────────────┴──────────╯
