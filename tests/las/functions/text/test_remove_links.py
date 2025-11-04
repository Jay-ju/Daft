# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.remove_links import RemoveLinks
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "prefix https://example.com/path suffix",
        None,
    ]
}

pattern = ""
repl = "[LINK]"
input_df = pd.DataFrame(samples)


def test_clean_links_operator():
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "cleaned_text",
        las_udf(
            RemoveLinks,
            construct_args={"pattern": pattern, "repl": repl},
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["cleaned_text"][1] is None
    assert actual_df["cleaned_text"][0] == "prefix [LINK] suffix"
