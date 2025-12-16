# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.doc import DocConvert
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
        "docx": [
            "https://las-ai-qa-online.tos-cn-beijing.volces.com/operator_cards_serving/public/qa/shared_doc_dataset/sample.docx"
        ],
    }
    df = daft.from_pydict(samples)

    constructor_kwargs = {"target_format": "pdf", "output_dir": "/tmp/doc_convert"}

    # 使用 Daft 进行分布式处理
    df = df.with_column(
        "pdf",
        las_udf(DocConvert, construct_args=constructor_kwargs, concurrency=1)(col("docx")),
    )

    df.show()
    # ╭──────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ docx                                         ┆ pdf                                          │
    # │ ---                                          ┆ ---                                          │
    # │ Utf8                                         ┆ Utf8                                         │
    # ╞══════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ tos://tos_bucket/doc_convert/sample.docx     ┆ /tmp/doc_convert/sample.pdf                  │
    # ╰──────────────────────────────────────────────┴──────────────────────────────────────────────╯
