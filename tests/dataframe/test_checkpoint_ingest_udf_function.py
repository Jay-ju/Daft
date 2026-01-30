from __future__ import annotations

import pytest

import daft
from daft import DataType, Series, col
from daft.udf import func
from tests.conftest import get_tests_daft_runner_name


@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray Runner to be in use")
def test_checkpoint_ingest_keys_function_udf_print_hash_mod_sample():
    import numpy as np
    import pyarrow as pa
    import ray

    num_buckets = 4
    sample_n = 2

    keys = [f"k{i % 97}" for i in range(2_000_000)]
    df_keys = daft.from_pydict({"key": keys})

    @ray.remote
    class KeySetActor:
        def __init__(self) -> None:
            self.keys: set[object] = set()

        def add_keys(self, input_keys: list[object]) -> None:
            self.keys.update(input_keys)

        def size(self) -> int:
            return len(self.keys)

        def sample_keys(self, n: int) -> list[object]:
            out = []
            for k in self.keys:
                out.append(k)
                if len(out) >= n:
                    break
            return out

    actors_by_bucket: dict[int, ray.actor.ActorHandle] = {i: KeySetActor.remote() for i in range(num_buckets)}

    @func.batch(return_dtype=DataType.null())
    async def ingest_keys( 
        input: Series,
        *,
        actors_by_bucket: dict[int, ray.actor.ActorHandle] = actors_by_bucket,
        num_buckets: int = num_buckets,
    ) -> Series:
        import asyncio

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
            await asyncio.wait_for(asyncio.gather(*futures), timeout=300)

        return Series.from_arrow(pa.nulls(num_rows))

    df_keys.select(ingest_keys(col("key"))).collect()

    sizes = ray.get([actors_by_bucket[i].size.remote() for i in range(num_buckets)])
    samples = ray.get([actors_by_bucket[i].sample_keys.remote(sample_n) for i in range(num_buckets)])

    print("num_buckets:", num_buckets)
    print("actor sizes:", {i: sizes[i] for i in range(num_buckets)})

    for i in range(num_buckets):
        sample_keys = samples[i]
        if not sample_keys:
            print(f"bucket {i}: empty sample")
            continue

        h = (
            Series.from_pylist(sample_keys, name="key")
            .hash()
            .to_arrow()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        mods = (h % np.uint64(num_buckets)).astype(np.int64, copy=False)
        uniq_mods = np.unique(mods)

        print(f"bucket {i}: uniq(hash%{num_buckets})={uniq_mods.tolist()}")
        print(f"bucket {i} sample keys:", sample_keys[:10])
        print(f"bucket {i} sample hash:", [int(x) for x in h[:10]])
        print(f"bucket {i} sample mod :", mods[:10].tolist())

        assert (mods == i).all(), f"bucket {i} has wrong hash%{num_buckets}: {uniq_mods}"
