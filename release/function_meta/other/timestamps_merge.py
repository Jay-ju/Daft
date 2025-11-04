from __future__ import annotations

import daft
from daft import col
from daft.las.functions.other.timestamps_merge import TimestampsMerge
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {"timestamps": [[[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]]]}
    ds = daft.from_pydict(samples)

    start_time = 0.0
    pre_merge_gap_seconds = 0.5
    max_span_seconds = 10.0
    enforce_chunking = True
    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": start_time,
                "pre_merge_gap_seconds": pre_merge_gap_seconds,
                "max_span_seconds": max_span_seconds,
                "enforce_chunking": enforce_chunking,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    ds.show()
    df = ds.to_pandas()

    # ╭────────────────────────────────┬───────────────────────────╮
    # │ timestamps                     ┆ timestamps_merged         │
    # │ ---                            ┆ ---                       │
    # │ List[List[Float64]]            ┆ List[List[Float32]]       │
    # ╞════════════════════════════════╪═══════════════════════════╡
    # │ [[0, 4.34], [5.5, 7.12], [8.1… ┆ [[0, 4.34], [5.5, 10.12]] │
    # ╰────────────────────────────────┴───────────────────────────╯
