from __future__ import annotations

import pytest

import daft
from daft import DataType, Series, col
from daft.udf import func
from tests.conftest import get_tests_daft_runner_name

"""
source /home/wangzheyan/las-Daft/.venv/bin/activate
export DAFT_RUNNER=ray
pytest -q -s tests/dataframe/test_checkpoint_ingest_udf_function.py
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray Runner to be in use")
def test_checkpoint_ingest_keys_function_udf_routes_to_correct_actor():
    import numpy as np
    import pyarrow as pa
    import ray

    num_buckets = 4
    keys = [f"k{i % 97}" for i in range(2000)]
    df_keys = daft.from_pydict({"key": keys})

    @ray.remote
    class KeySetActor:
        def __init__(self) -> None:
            self.keys: set[object] = set()

        def add_keys(self, input_keys: list[object]) -> None:
            self.keys.update(input_keys)

        def get_keys(self) -> set[object]:
            return self.keys

    actors_by_bucket: dict[int, ray.actor.ActorHandle] = {i: KeySetActor.remote() for i in range(num_buckets)}

    @func.batch(return_dtype=DataType.null())
    def ingest_keys(
        input: Series,
        *,
        actors_by_bucket: dict[int, ray.actor.ActorHandle] = actors_by_bucket,
        num_buckets: int = num_buckets,
    ) -> Series:
        num_rows = len(input)
        if num_rows == 0:
            return Series.from_arrow(pa.nulls(0))

        keys = input.to_pylist()
        keys_np = np.asarray(keys, dtype=object)

        hash_arr = input.hash().to_arrow()
        hash_np = hash_arr.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        bucket_ids = (hash_np % np.uint64(num_buckets)).astype(np.int64, copy=False)

        row_order = np.argsort(bucket_ids, kind="stable")
        bucket_sorted = bucket_ids[row_order]
        run_starts = np.flatnonzero(np.r_[True, bucket_sorted[1:] != bucket_sorted[:-1]])
        run_ends = np.r_[run_starts[1:], len(bucket_sorted)]
        buckets_present = bucket_sorted[run_starts]

        futures = []
        for bucket, start, end in zip(buckets_present, run_starts, run_ends):
            actor = actors_by_bucket[int(bucket)]
            subset = keys_np[row_order[int(start) : int(end)]].tolist()
            futures.append(actor.add_keys.remote(subset))

        if futures:
            ray.get(futures, timeout=300)

        return Series.from_arrow(pa.nulls(num_rows))

    df_keys.select(ingest_keys(col("key"))).collect()

    expected_sets: dict[int, set[object]] = {i: set() for i in range(num_buckets)}
    base_hash = (
        Series.from_pylist(keys, name="key")
        .hash()
        .to_arrow()
        .to_numpy(zero_copy_only=False)
        .astype(np.uint64, copy=False)
    )
    base_bucket_ids = (base_hash % np.uint64(num_buckets)).astype(np.int64, copy=False)
    for k, b in zip(keys, base_bucket_ids):
        expected_sets[int(b)].add(k)

    got_sets = ray.get([actors_by_bucket[i].get_keys.remote() for i in range(num_buckets)])
    got_by_bucket = {i: got_sets[i] for i in range(num_buckets)}

    print("num_buckets:", num_buckets)
    print("expected sizes:", {i: len(expected_sets[i]) for i in range(num_buckets)})
    print("actual sizes:", {i: len(got_by_bucket[i]) for i in range(num_buckets)})

    for i in range(num_buckets):
        got_keys_list = sorted(list(got_by_bucket[i]))
        if not got_keys_list:
            print(f"bucket {i}: empty")
            continue

        h = (
            Series.from_pylist(got_keys_list, name="key")
            .hash()
            .to_arrow()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        mods = (h % np.uint64(num_buckets)).astype(np.int64, copy=False)
        uniq_mods = np.unique(mods)

        sample_n = min(30, len(got_keys_list))
        sample_keys = got_keys_list[:sample_n]
        sample_hash = h[:sample_n]
        sample_mods = mods[:sample_n]

        print(f"bucket {i}: uniq(hash%{num_buckets})={uniq_mods.tolist()}")
        print(f"bucket {i} sample keys:", sample_keys)
        print(f"bucket {i} sample hash:", [int(x) for x in sample_hash])
        print(f"bucket {i} sample mod :", sample_mods.tolist())

        assert (mods == i).all(), f"bucket {i} has wrong hash%{num_buckets}: {uniq_mods}"

    assert got_by_bucket == expected_sets

