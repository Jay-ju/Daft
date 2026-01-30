from __future__ import annotations

import os
import time

import numpy as np
import pyarrow as pa

import daft
from daft.series import Series

"""
source /home/wangzheyan/las-Daft/.venv/bin/activate
export DAFT_RUNNER=ray
pytest -q -s tests/microbenchmarks/test_series_mod_vs_numpy_mod.py
"""
def test_series_mod_vs_numpy_mod_print_timings() -> None:
    num_rows = int(os.getenv("DAFT_BENCH_NUM_ROWS", "2000000"))
    num_buckets = int(os.getenv("DAFT_BENCH_NUM_BUCKETS", "257"))

    keys = [str(i) for i in range(num_rows)]
    s = daft.Series.from_pylist(keys, name="k")
    h = s.hash()

    start = time.perf_counter()
    hash_arr = h.to_arrow()
    t1 = time.perf_counter()
    hash_np = hash_arr.to_numpy(zero_copy_only=False)
    t2 = time.perf_counter()
    hash_np_u64 = hash_np.astype(np.uint64, copy=False)
    t3 = time.perf_counter()
    bucket_ids_a = (hash_np_u64 % np.uint64(num_buckets)).astype(np.int64, copy=False)
    t4 = time.perf_counter()

    start_b = time.perf_counter()
    nb = Series.from_pylist([num_buckets], name="nb")
    tb1 = time.perf_counter()
    bucket_series = h % nb
    tb2 = time.perf_counter()
    bucket_arr = bucket_series.to_arrow()
    tb3 = time.perf_counter()
    bucket_ids_b = bucket_arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    tb4 = time.perf_counter()

    assert bucket_ids_a.shape == bucket_ids_b.shape
    assert np.array_equal(bucket_ids_a, bucket_ids_b)

    print("pyarrow:", pa.__version__)
    print("num_rows:", num_rows, "num_buckets:", num_buckets)
    print(
        "A hash.to_arrow:",
        t1 - start,
        "to_numpy:",
        t2 - t1,
        "astype_u64:",
        t3 - t2,
        "numpy_mod:",
        t4 - t3,
        "total:",
        t4 - start,
    )
    print(
        "B make_scalar_series:",
        tb1 - start_b,
        "series_mod:",
        tb2 - tb1,
        "to_arrow:",
        tb3 - tb2,
        "to_numpy:",
        tb4 - tb3,
        "total:",
        tb4 - start_b,
    )
