# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.clean_email import CleanEmail
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "lihua@163.com This is a test content.",
        "This is a test content.",
        None,
    ]
}

repl = "****"
input_df = pd.DataFrame(samples)


def test_clean_email():
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            CleanEmail,
            construct_args={"repl": repl},
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["cleaned_text"][0] == "**** This is a test content."
    assert actual_df["cleaned_text"][2] is None
