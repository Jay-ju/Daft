# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.whitespace_normalizer import WhitespaceNormalizer
from daft.las.functions.udf import las_udf


def test_whitespace_normalizer_unicode_spaces():
    test_texts = [
        "Hello\u2000World\u2001Test",
        "Hello\u2002World\u2003Test",
        "Hello\u2004World\u2005Test",
        "Hello\u2006World\u2007Test",
        "Hello\u2008World\u2009Test",
        "Hello\u200aWorld\u200bTest",
        "Hello\u202fWorld\u205fTest",
        "Hello\u3000World\u200bTest",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 8
    for i in range(8):
        assert actual_df["normalized_text"][i] == "Hello World Test"


def test_whitespace_normalizer_mixed_unicode():
    test_texts = [
        "中文\u2000文本\u2001测试",
        "English\u2002中文\u2003混合",
        "数字\u2004123\u2005456",
        "符号\u2006@#$%\u2007&*()",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 4
    assert actual_df["normalized_text"][0] == "中文 文本 测试"
    assert actual_df["normalized_text"][1] == "English 中文 混合"
    assert actual_df["normalized_text"][2] == "数字 123 456"
    assert actual_df["normalized_text"][3] == "符号 @#$% &*()"


def test_whitespace_normalizer_consecutive_unicode():
    test_texts = [
        "Hello\u2000\u2001\u2002World",
        "Test\u2003\u2004\u2005\u2006Case",
        "Multiple\u2007\u2008\u2009\u200a\u200bSpaces",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 3
    assert actual_df["normalized_text"][0] == "Hello World"
    assert actual_df["normalized_text"][1] == "Test Case"
    assert actual_df["normalized_text"][2] == "Multiple Spaces"


def test_whitespace_normalizer_ideographic_spaces():
    test_texts = [
        "中文\u3000文本\u3000测试",
        "English\u3000中文\u3000Mixed",
        "数字\u3000123\u3000456",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 3
    assert actual_df["normalized_text"][0] == "中文 文本 测试"
    assert actual_df["normalized_text"][1] == "English 中文 Mixed"
    assert actual_df["normalized_text"][2] == "数字 123 456"


def test_whitespace_normalizer_zero_width():
    test_texts = [
        "Hello\u200bWorld\u200bTest",
        "中文\u200b文本\u200b测试",
        "Mixed\u200bEnglish\u200b中文",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "normalized_text",
        las_udf(WhitespaceNormalizer)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 3
    assert actual_df["normalized_text"][0] == "Hello World Test"
    assert actual_df["normalized_text"][1] == "中文 文本 测试"
    assert actual_df["normalized_text"][2] == "Mixed English 中文"
