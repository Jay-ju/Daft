# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.other.timestamps_merge import TimestampsMerge
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result

samples = [
    {
        "timestamps": [[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]],
    }
]
input_df = pd.DataFrame(samples)


def test_timestamps_merge_1():
    expected_samples = [
        {
            "timestamps": [[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]],
            "timestamps_merged": [[0.0, 4.34], [5.5, 10.12]],
        }
    ]
    expected_df = pd.DataFrame(expected_samples)

    ds = daft.from_pandas(input_df)

    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": 0.0,
                "pre_merge_gap_seconds": 0.5,
                "max_span_seconds": 10.0,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    actual_df = ds.to_pandas()
    assert_dataframe_result(actual_df, expected_df)


def test_timestamps_merge_2():
    expected_samples = [
        {
            "timestamps": [[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]],
            "timestamps_merged": [[1.0, 5.34], [6.5, 11.12]],
        }
    ]
    expected_df = pd.DataFrame(expected_samples)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": 1.0,
                "pre_merge_gap_seconds": 0.5,
                "max_span_seconds": 10.0,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    actual_df = ds.to_pandas()
    assert_dataframe_result(actual_df, expected_df)


def test_timestamps_merge_3():
    expected_samples = [
        {"timestamps": [[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]], "timestamps_merged": [[0.0, 10.12]]}
    ]
    expected_df = pd.DataFrame(expected_samples)

    ds = daft.from_pandas(input_df)

    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": 0.0,
                "pre_merge_gap_seconds": 2.0,
                "max_span_seconds": 10.0,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    actual_df = ds.to_pandas()
    assert_dataframe_result(actual_df, expected_df)


def test_timestamps_merge_4():
    expected_samples = [
        {
            "timestamps": [[0.0, 4.34], [5.50, 7.12], [8.10, 8.34], [8.50, 10.12]],
            "timestamps_merged": [[0.0, 3.0], [3.0, 4.34], [5.5, 7.12], [8.1, 10.12]],
        }
    ]
    expected_df = pd.DataFrame(expected_samples)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": 0.0,
                "pre_merge_gap_seconds": 0.5,
                "max_span_seconds": 3,
                "enforce_chunking": True,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    actual_df = ds.to_pandas()
    assert_dataframe_result(actual_df, expected_df)


def test_timestamps_merge_5():
    input_df = pd.DataFrame(
        [
            {
                "timestamps": [[0.0, 4.34], [5.50], [8.10, 8.34], [8.50, 10.12]],
            }
        ]
    )

    expected_samples = [{"timestamps": [[0.0, 4.34], [5.50], [8.10, 8.34], [8.50, 10.12]], "timestamps_merged": None}]
    expected_df = pd.DataFrame(expected_samples)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "timestamps_merged",
        las_udf(
            TimestampsMerge,
            construct_args={
                "start_time": 0.0,
                "pre_merge_gap_seconds": 0.5,
                "max_span_seconds": 3,
                "enforce_chunking": True,
            },
            num_cpus=1,
            batch_size=1,
            concurrency=1,
        )(col("timestamps")),
    )
    actual_df = ds.to_pandas()
    assert_dataframe_result(actual_df, expected_df)
