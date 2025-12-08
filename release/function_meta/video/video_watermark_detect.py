from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoWatermarkDetect

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
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_video_dataset/watermark_sample.mp4"
        ],
    }
    ds = daft.from_pydict(samples)

    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    constructor_kwargs = {
        "model_path": model_path,
        "model_name": "PP-OCRv4/ch_det",
        "sample_count": 15,
        "consistency_threshold": 0.8,
        "position_tolerance": 15,
    }

    ds = ds.with_column(
        "results",
        las_udf(
            VideoWatermarkDetect,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ results                                                                                   │
    # │ ---                            ┆ ---                                                                                       │
    # │ Utf8                           ┆ Struct[watermark_regions: List[Struct[ymin: Int32, ymax: Int32, xmin: Int32, xmax: Int32, │
    # │                                ┆ confidence: Float32]], video_resolution: List[Int32], total_frames: Int64]                │
    # ╞════════════════════════════════╪═══════════════════════════════════════════════════════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ {watermark_regions: [{ymin: 5…                                                            │
    # ╰────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────╯
