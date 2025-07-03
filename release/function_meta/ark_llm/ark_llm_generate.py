from __future__ import annotations

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_generate import ArkLLMGenerate
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    # 配置模型的访问信息，包括ak和账号id，请根据实际情况修改
    access_key_id = "you_access_key_id"
    accound_id = "your_account_id"
    queries = ["中国的首都在哪里", "十字花科植物有哪些"]
    messages = {"messages": [[{"role": "user", "content": query}] for query in queries]}

    ds = daft.from_pydict(messages)
    ds = ds.with_column(
        "llm_result",
        las_udf(
            ArkLLMGenerate,
            construct_args={
                "model": "doubao-1.5-pro-32k",
                "version": "250115",
                "inference_type": "online",
                "access_key": access_key_id,
                "account_id": accound_id,
            },
        )(col("messages")),
    )
    ds.show()

    #  输出(每次大模型推理结果可能不同)
# ╭─────────────────────────────────────────┬─────────────────────────────────────────────────────────────╮
# │ messages                                ┆ llm_result                                                  │
# │ ---                                     ┆ ---                                                         │
# │ List[Struct[content: Utf8, role: Utf8]] ┆ Utf8                                                        │
# ╞═════════════════════════════════════════╪═════════════════════════════════════════════════════════════╡
# │ [{content: 中国的首都在哪里,            ┆ 中国的首都是北京。                                          │
# │ role: us…                               ┆                                                             │
# │                                         ┆ 北京是中华人民共和国的政治中心、文化…                       │
# ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
# │ [{content: 十字花科植物有哪些,          ┆ 十字花科是一个经济价值较大的科，拥有许多常见的植物，以下为… │
# │ role: u…                                ┆                                                             │
# ╰─────────────────────────────────────────┴─────────────────────────────────────────────────────────────╯
