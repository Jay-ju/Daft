from __future__ import annotations

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_1_5_lite_32k import Doubao15Lite32k
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 需提前配置环境变量 LAS_ACCESS_KEY 和 LAS_ACCOUND_ID ： LAS_ACCESS_KEY 是账号的 AK ， LAS_ACCOUND_ID 是账号 id
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
            Doubao15Lite32k,
            construct_args={
                "version": "250115",
            },
        )(col("query")),
    )
    ds.show()

    #  输出(每次大模型推理结果可能不同)
    # ╭────────────────────┬───────────────────────────────────────╮
    # │ query              ┆ llm_result                            │
    # │ ---                ┆ ---                                   │
    # │ Utf8               ┆ Utf8                                  │
    # ╞════════════════════╪═══════════════════════════════════════╡
    # │ 中国的首都在哪里      ┆ 中国的首都是北京。                       │
    # │                    ┆                                       │
    # │                    ┆ 北京是中国的政治中心、文化中心、国际…       │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ 十字花科植物有哪些    ┆ 十字花科植物种类繁多，常见的有：           │
    # │                    ┆ 1. **蔬菜类**                          │
    # │                    ┆  …                                    │
    # ╰────────────────────┴───────────────────────────────────────╯
