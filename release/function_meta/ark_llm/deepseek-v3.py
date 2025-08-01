from __future__ import annotations

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_text_generate import ArkLLMTextGenerate
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需配置环境变量 LAS_API_KEY ： LAS_API_KEY 通过在 LAS 服务页面上创建获取
    queries = {
        "query": [
            "中国的首都在哪里",
            "十字花科植物有哪些",
        ]
    }

    ds = daft.from_pydict(queries)
    ds = ds.with_column(
        "llm_result",
        las_udf(
            ArkLLMTextGenerate,
            construct_args={
                "model": "deepseek-v3",
                "inference_type": "online",
            },
        )(col("query")),
    )
    ds.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────┬─────────────────────────────────────────────────────────╮
    # │ query              ┆ llm_result                                              │
    # │ ---                ┆ ---                                                     │
    # │ Utf8               ┆ Utf8                                                    │
    # ╞════════════════════╪═════════════════════════════════════════════════════════╡
    # │ 中国的首都在哪里   ┆ 中国的首都是**北京**。北京是中国的政治、文化、国际交往… │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 十字花科植物有哪些 ┆ 十字花科（Brassicaceae或Cruciferae）…                   │
    # ╰────────────────────┴─────────────────────────────────────────────────────────╯
