from __future__ import annotations

import logging

import ray

import daft
from daft import col
from daft.las.functions.text.remove_links import RemoveLinks
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

    samples = {
        "text": [
            "prefix https://example.com/path suffix",
            None,
        ]
    }

    pattern = ""
    repl = "[LINK]"

    ds = daft.from_pydict(samples)

    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            RemoveLinks,
            construct_args={"pattern": pattern, "repl": repl},
        )(col("text")),
    )
    ds.show()

    # ╭────────────────────────────────┬──────────────────────╮
    # │ text                           ┆ cleaned_text         │
    # │ ---                            ┆ ---                  │
    # │ Utf8                           ┆ Utf8                 │
    # ╞════════════════════════════════╪══════════════════════╡
    # │ prefix https://example.com/pa… ┆ prefix [LINK] suffix │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                           ┆ None                 │
    # ╰────────────────────────────────┴──────────────────────╯
