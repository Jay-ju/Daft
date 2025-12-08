from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSttnInpaint

if __name__ == "__main__":
    TOS_TEST_DIR = os.getenv("TOS_TEST_DIR", "tos_bucket")
    output_tos_dir = f"tos://{TOS_TEST_DIR}/video_sttn_inpaint_test"

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
        ]
    }

    ds = daft.from_pydict(samples)

    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "model_path": model_path,
        "model_name": "researchmm/STTN",
        "neighbor_stride": 5,
        "reference_length": 10,
        "max_load_num": 50,
    }

    ds = ds.with_column(
        "results",
        las_udf(
            VideoSttnInpaint,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    ds.show()

    # ╭────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────╮
    # │ video_path                     ┆ results                                                                               │
    # │ ---                            ┆ ---                                                                                   │
    # │ Utf8                           ┆ Struct[output_path: Utf8, processed_frames: Int32, processed_resolution: List[Int32]] │
    # ╞════════════════════════════════╪═══════════════════════════════════════════════════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ {output_path: tos://las-ai-qa…                                                        │
    # ╰────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────╯
