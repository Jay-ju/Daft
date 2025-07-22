# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.special_characters_ratio_calculator import SpecialCharactersRatioCalculator
from daft.las.functions.udf import las_udf


def test_special_characters_ratio_calculator_all():
    test_texts = [
        "Hello world!",
        "1234567890",
        "     ",
        "Hello\tworld\n",
        "!!!@@@###",
        "这是中文文本",
        "你好 Hello 😊 123 !!!",
        "",
        "no-specials",
        "Hello, world! 123 这是中文。",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "special_ratio",
        las_udf(
            SpecialCharactersRatioCalculator,
            construct_args={"character_type": "all"},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "special_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["special_ratio"]) == 10

    assert result_list["special_ratio"][0] > 0.1
    assert result_list["special_ratio"][1] == 1.0
    assert result_list["special_ratio"][2] is None
    assert result_list["special_ratio"][3] > 0.15
    assert result_list["special_ratio"][4] == 1.0
    assert result_list["special_ratio"][5] < 0.05
    assert result_list["special_ratio"][6] > 0.3
    assert result_list["special_ratio"][7] is None
    assert result_list["special_ratio"][8] > 0.0
    assert result_list["special_ratio"][9] > 0.1


def test_special_characters_ratio_calculator_whitespace():
    test_texts = [
        "Hello world",
        "Hello\tworld\n",
        "Hello world with spaces",
        "NoSpacesHere",
        "Multiple\t\t\t\tspaces\n\n\n",
        "",
        "     ",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "whitespace_ratio",
        las_udf(
            SpecialCharactersRatioCalculator,
            construct_args={"character_type": "whitespace"},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "whitespace_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["whitespace_ratio"]) == 7

    assert result_list["whitespace_ratio"][0] > 0.0
    assert result_list["whitespace_ratio"][1] > 0.0
    assert result_list["whitespace_ratio"][2] > 0.0
    assert result_list["whitespace_ratio"][3] == 0.0
    assert result_list["whitespace_ratio"][4] > 0.0
    assert result_list["whitespace_ratio"][5] is None
    assert result_list["whitespace_ratio"][6] is None


def test_special_characters_ratio_calculator_punctuation():
    test_texts = [
        "Hello world!",
        "Hello, world.",
        "No punctuation here",
        "!!!@@@###$$$",
        "Mixed: content, with! punctuation.",
        "",
        "     ",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "punctuation_ratio",
        las_udf(
            SpecialCharactersRatioCalculator,
            construct_args={"character_type": "punctuation"},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "punctuation_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["punctuation_ratio"]) == 7

    assert result_list["punctuation_ratio"][0] > 0.0
    assert result_list["punctuation_ratio"][1] > 0.0
    assert result_list["punctuation_ratio"][2] == 0.0
    assert result_list["punctuation_ratio"][3] > 0.0
    assert result_list["punctuation_ratio"][4] > 0.0
    assert result_list["punctuation_ratio"][5] is None
    assert result_list["punctuation_ratio"][6] is None


def test_special_characters_ratio_calculator_digits():
    test_texts = [
        "Hello world 123",
        "No digits here",
        "123456789",
        "Mixed content with 123 numbers",
        "HelloWorld123",
        "",
        "     ",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "digits_ratio",
        las_udf(
            SpecialCharactersRatioCalculator,
            construct_args={"character_type": "digits"},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "digits_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["digits_ratio"]) == 7

    assert result_list["digits_ratio"][0] > 0.0
    assert result_list["digits_ratio"][1] == 0.0
    assert result_list["digits_ratio"][2] > 0.0
    assert result_list["digits_ratio"][3] > 0.0
    assert result_list["digits_ratio"][4] > 0.0
    assert result_list["digits_ratio"][5] is None
    assert result_list["digits_ratio"][6] is None
