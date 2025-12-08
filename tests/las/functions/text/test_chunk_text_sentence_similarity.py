# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.chunk_text_sentence_similarity import ChunkTextSentenceSimilarity
from daft.las.functions.udf import las_udf

chunk_size = 128
chunk_overlap = 32

samples = {
    "text": [
        """2023年，中国乘用车市场零售总量恢复至疫情前的水平，显示出市场的强劲复苏。新能源乘用车行业同年实现了高质量增长，全年销量超过770万辆，平均每三辆新售乘用车中就有一辆为新能源车型。在政策支持和市场供需的共同推动下，新能源车市场展现出更多市场化特征。预计2024年，这一增长趋势将持续。随着新能源技术和商业模式的不断成熟，乘用车产品正逐步迈入“体验型商品”阶段，消费者在购车时更加关注整体体验，而不仅仅是车辆的社会属性或耐用品属性。
    与此同时，经过多年的快速发展和普及，2023年我国新车金融渗透率出现了首次回落，整体渗透率为56%，较上一年下降2%，新能源车的金融渗透率略低于整体水平。市场竞争日益激烈，加之宏观金融环境的影响，汽车金融领域的价格战愈发激烈。此外，产品同质化、需求多样化满足不足以及渠道模式固化等问题依然存在。研究认为，围绕用户综合体验进行汽车金融服务转型，将成为行业突破瓶颈、实现差异化发展的关键方向。""",
        None,
        "",
    ]
}
txt_input_df = pd.DataFrame(samples)


def test_chunk_text_sentence_similarity_text(local_models_dir):
    ds = daft.from_pandas(txt_input_df)
    ds = ds.with_column(
        "chunks",
        las_udf(
            ChunkTextSentenceSimilarity,
            construct_args={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model_path": local_models_dir,
                "embedding_model_name": "BAAI/bge-m3",
                "breakpoint_percentile_threshold": 80,
            },
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["chunks"][0]) == 2
    assert "消费者在购车时更加关注整体体验" in actual_df["chunks"][0][1]
    assert actual_df["chunks"][1] is None or len(actual_df["chunks"][1]) == 0
