from __future__ import annotations

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_1_5_lite_32k import Doubao15Lite32k
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 配置模型的访问信息，包括ak和账号id，请根据实际情况修改
    access_key_id = "you_access_key_id"
    accound_id = "your_account_id"
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
                "access_key": access_key_id,
                "account_id": accound_id,
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
