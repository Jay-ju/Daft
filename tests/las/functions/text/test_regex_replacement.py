# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.regex_replacement import RegexReplacer
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "<Query></Query><Title>提升党的领导力，推进国家治理体系和治理能力现代化</Title><Url>http://news.cnr.cn/native/gd/20191212/t20191212_524895559.shtml</Url>",
        None,
    ]
}
input_df = pd.DataFrame(samples)
patterns = [r"<.*?>", "国家"]
replacements = ["/replace_tag", "中国"]


def test_clean_html_tag():
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "replaced_text",
        las_udf(
            RegexReplacer,
            construct_args={"patterns": patterns, "replacements": replacements},
        )(col("text")),
    )
    actual_df = ds.to_pandas()
    assert actual_df["replaced_text"][1] is None
    assert "/replace_tag/replace_tag/replace_tag提升党的领导力" in actual_df["replaced_text"][0]
