from __future__ import annotations

import pytest

import daft
from tests.conftest import get_tests_daft_runner_name

pytestmark = pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray Runner to be in use")

"""
DAFT_RUNNER=ray pytest -s tests/dataframe/test_hash_repartition_bucket_mapping.py
"""


def test_hash_repartition_partition_index_is_bucket_id(make_df) -> None:
    num_buckets = 7
    ids = list(range(200))

    df = make_df({"id": ids}).select("id").repartition(num_buckets, "id")
    parts = list(df.iter_partitions())

    import ray

    parts = ray.get(parts)
    assert len(parts) == num_buckets
    print(f"len(parts): {len(parts)}")

    seen: set[int] = set()
    for bucket_id, mp in enumerate(parts):
        part_ids = mp.to_pydict()["id"]
        print(f"part_ids: {part_ids}")
        s = daft.Series.from_pylist(part_ids, name="id")
        hashes = s.hash().to_pylist()
        remainders = [(h % num_buckets) for h in hashes]
        print(f"remainders: {remainders}")

        for v, h in zip(part_ids, hashes):
            assert (h % num_buckets) == bucket_id
            seen.add(v)

    assert seen == set(ids)
