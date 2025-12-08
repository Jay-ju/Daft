from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.image.image_hash import ImageHash
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
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "image_hash",
        las_udf(
            ImageHash,
            construct_args={
                "image_src_type": "image_url",
                "method": "phash",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("image")),
    )

    ds.show()

    # ╭────────────────────────────────┬────────────────────────────────────────╮
    # │ image                          ┆ image_hash                             │
    # │ ---                            ┆ ---                                    │
    # │ Utf8                           ┆ Struct[hash_hex: Utf8, hash_bin: Utf8] │
    # ╞════════════════════════════════╪════════════════════════════════════════╡
    # │ https://las-ai-qa-online.tos-… ┆ {hash_hex: 8d3986a636e768ad,           │
    # │                                ┆ …                                      │
    # ╰────────────────────────────────┴────────────────────────────────────────╯
