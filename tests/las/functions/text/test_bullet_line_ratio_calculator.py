# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.bullet_line_ratio_calculator import BulletLineRatioCalculator
from daft.las.functions.udf import las_udf


def test_bullet_line_ratio_calculator_basic():
    test_texts = [
        "第一行内容\n• 项目符号行1\n- 项目符号行2\n第三行内容\n* 项目符号行3",
        "这是普通文本\n没有任何项目符号\n都是正常的内容",
        "• 只有项目符号\n- 全是项目符号\n* 没有普通文本",
        "",
        None,
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "bullet_ratio",
        las_udf(BulletLineRatioCalculator)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 5
    assert actual_df["bullet_ratio"][0] == 0.6
    assert actual_df["bullet_ratio"][1] == 0.0
    assert actual_df["bullet_ratio"][2] == 1.0
    assert pd.isna(actual_df["bullet_ratio"][3])
    assert pd.isna(actual_df["bullet_ratio"][4])


def test_bullet_line_ratio_calculator_different_bullets():
    test_texts = [
        "普通文本\n● 圆形项目符号\n▪ 方形项目符号\n→ 箭头项目符号\n★ 星形项目符号",
        "混合内容\n· 点项目符号\n➤ 箭头项目符号\n普通文本\n✔ 勾选项目符号",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "bullet_ratio",
        las_udf(BulletLineRatioCalculator)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 2
    assert actual_df["bullet_ratio"][0] == 0.8
    assert actual_df["bullet_ratio"][1] == 0.6


def test_bullet_line_ratio_calculator_whitespace():
    test_texts = [
        "   • 前面有空格的项目符号\n\t- 前面有制表符的项目符号\n普通文本",
        "• 正常项目符号\n  普通文本\n  * 前面有空格的项目符号",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "bullet_ratio",
        las_udf(BulletLineRatioCalculator)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 2
    assert 0.66 <= actual_df["bullet_ratio"][0] <= 0.67
    assert 0.66 <= actual_df["bullet_ratio"][1] <= 0.67


def test_bullet_line_ratio_calculator_empty_lines():
    test_texts = [
        "\n\n• 项目符号\n\n- 项目符号\n\n普通文本\n\n",
        "普通文本\n\n\n• 项目符号\n\n\n",
    ]

    test_df = pd.DataFrame({"text": test_texts})
    ds = daft.from_pandas(test_df)
    ds = ds.with_column(
        "bullet_ratio",
        las_udf(BulletLineRatioCalculator)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 2
    assert 0.66 <= actual_df["bullet_ratio"][0] <= 0.67
    assert actual_df["bullet_ratio"][1] == 0.5


def test_bullet_line_ratio_calculator_large_text():
    large_text = "普通文本\n" + "\n".join([f"• 项目符号行{i}" for i in range(50)]) + "\n普通文本"

    large_samples = {"text": [large_text]}
    large_df = pd.DataFrame(large_samples)

    ds = daft.from_pandas(large_df)
    ds = ds.with_column(
        "bullet_ratio",
        las_udf(BulletLineRatioCalculator)(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 1
    assert 0.96 <= actual_df["bullet_ratio"][0] <= 0.97
