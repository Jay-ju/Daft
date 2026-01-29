from __future__ import annotations

import numpy as np

import daft

"""
source /home/wangzheyan/las-Daft/.venv/bin/activate
export DAFT_RUNNER=ray
pytest -q -s tests/dataframe/test_checkpoint_filter_hash_routing.py
"""


def _bucket_ids_for_series(s: daft.Series, num_buckets: int) -> np.ndarray:
    try:
        import pyarrow.compute as pc

        hash_arr = s.hash().to_arrow()
        bucket_arr = pc.modulus(hash_arr, num_buckets)
        return bucket_arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    except Exception:
        hashes = s.hash().to_pylist()
        hashes_np = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
        return (hashes_np % np.uint64(num_buckets)).astype(np.int64, copy=False)


def test_checkpoint_filter_hash_routing_series_blocks_are_consistent() -> None:
    num_buckets = 5

    keys = [
        "a",
        "b",
        "a",
        "c",
        None,
        "d",
        "e",
        "f",
        "b",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        None,
        "m",
        "n",
        "o",
        "p",
    ]

    s = daft.Series.from_pylist(keys, name="key")
    hashes = s.hash().to_pylist()
    bucket_ids = _bucket_ids_for_series(s, num_buckets)

    print("num_buckets:", num_buckets)
    print("keys:", keys)
    print("hashes:", hashes)
    print("bucket_ids:", bucket_ids.tolist())

    for bucket in range(num_buckets):
        row_indices = np.nonzero(bucket_ids == bucket)[0]
        if len(row_indices) == 0:
            continue

        block_keys = [keys[i] for i in row_indices]
        block_hashes = [hashes[i] for i in row_indices]
        block_bucket_ids = bucket_ids[row_indices]

        print("\nBUCKET", bucket, "size", len(row_indices))
        print("  indices:", row_indices.tolist())
        print("  keys:", block_keys)
        print("  hashes:", block_hashes)
        print("  bucket_ids:", block_bucket_ids.tolist())

        assert np.all(block_bucket_ids == bucket)
        for h in block_hashes:
            assert int(h % num_buckets) == bucket

    assert len(bucket_ids) == len(keys)
