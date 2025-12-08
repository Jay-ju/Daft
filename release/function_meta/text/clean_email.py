from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.clean_email import CleanEmail
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
        "text": [
            "lihua@163.com This is a test content.",
            "This is a test content.",
            None,
        ]
    }

    repl = "****"
    ds = daft.from_pydict(samples)

    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            CleanEmail,
            construct_args={"repl": repl},
        )(col("text")),
    )
    ds.show()

    # ╭────────────────────────────────┬──────────────────────────────╮
    # │ text                           ┆ cleaned_text                 │
    # │ ---                            ┆ ---                          │
    # │ Utf8                           ┆ Utf8                         │
    # ╞════════════════════════════════╪══════════════════════════════╡
    # │ lihua@163.com This is a test … ┆ **** This is a test content. │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ This is a test content.        ┆ This is a test content.      │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                           ┆ None                         │
    # ╰────────────────────────────────┴──────────────────────────────╯
