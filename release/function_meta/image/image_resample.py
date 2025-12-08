from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.image_resample import ImageResample
from daft.las.functions.udf import las_udf

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
        "image": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_image_dataset/cat_ip_adapter.jpeg"
        ],
        "image_name": ["cat_ip_adapter"],
    }

    image_suffix = ".jpg"
    image_src_type = "image_url"
    target_size = (200, 200)
    target_dpi = (72, 72)
    method = "lanczos"
    local_dir = ""
    tos_dir = ""
    num_gpus = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "image_resample",
        las_udf(
            ImageResample,
            construct_args={
                "image_suffix": image_suffix,
                "tos_dir": tos_dir,
                "local_dir": local_dir,
                "image_src_type": image_src_type,
                "target_size": target_size,
                "target_dpi": target_dpi,
                "method": method,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image"), col("image_name")),
    )

    ds.show()

    # ╭────────────────────────────────┬────────────────┬────────────────────────────────────────╮
    # │ image                          ┆ image_name     ┆ image_resample                         │
    # │ ---                            ┆ ---            ┆ ---                                    │
    # │ Utf8                           ┆ Utf8           ┆ Struct[base64: Utf8, image_path: Utf8] │
    # ╞════════════════════════════════╪════════════════╪════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ cat_ip_adapter ┆ {base64: iVBORw0KGgoAAAANSUhE…         │
    # ╰────────────────────────────────┴────────────────┴────────────────────────────────────────╯
