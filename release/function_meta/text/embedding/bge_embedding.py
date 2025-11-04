from __future__ import annotations

import logging
import os

import ray

import daft
from daft import col
from daft.las.functions.text.embedding.bge_embedding import BgeEmbedding
from daft.las.functions.udf import las_udf

if __name__ == "__main__":

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

    samples = {"text": ["Hello World!", None]}
    dtype = "float16"
    batch_size = 512
    model_path = os.getenv("MODEL_PATH", "/opt/las/models")
    model_name = "BAAI/bge-m3"
    rank = 0

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "embeddings",
        las_udf(
            BgeEmbedding,
            construct_args={
                "dtype": dtype,
                "batch_size": batch_size,
                "model_path": model_path,
                "model_name": model_name,
                "rank": rank,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    df = ds.to_pandas()
    print(df[0]["embeddings"])
    ds.show()

    # ╭──────────────┬────────────────────────────────╮
    # │ text         ┆ embeddings                     │
    # │ ---          ┆ ---                            │
    # │ Utf8         ┆ List[Float32]                  │
    # ╞══════════════╪════════════════════════════════╡
    # │ Hello World! ┆ [-0.042053223, 0.02178955, -0… │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None         ┆ None                           │
    # ╰──────────────┴────────────────────────────────╯
