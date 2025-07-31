from __future__ import annotations

import pandas as pd
import pyarrow as pa

import daft
from daft import col
from daft.las.functions.types import Operator
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


class DummyUDF(Operator):
    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.int64()

    def transform(self, col_a: pa.Array, col_b: pa.Array | None = None, c: str | None = None) -> pa.Array:
        assert isinstance(col_a, pa.Array)
        return col_a


def test_las_udf():
    df = daft.from_pydict({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    expected_df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"], "dummy_col": [1, 2, 3]})

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col("a"), col("b"), "dummy"))
    assert_dataframe_result(df.to_pandas(), expected_df)

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col_a=col("a"), col_b=col("b"), c="dummy"))
    assert_dataframe_result(df.to_pandas(), expected_df)

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col("a")))
    assert_dataframe_result(df.to_pandas(), expected_df)

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col_a=col("a")))
    assert_dataframe_result(df.to_pandas(), expected_df)

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col("a"), None, None))
    assert_dataframe_result(df.to_pandas(), expected_df)

    df = df.with_column("dummy_col", las_udf(DummyUDF)(col_a=col("a"), col_b=None, c=None))
    assert_dataframe_result(df.to_pandas(), expected_df)
