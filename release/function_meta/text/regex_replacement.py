from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.regex_replacement import RegexReplacer
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "<Query></Query><Title>提升党的领导力，推进国家治理体系和治理能力现代化</Title><Url>http://news.cnr.cn/native/gd/20191212/t20191212_524895559.shtml</Url>",
            None,
        ]
    }

    patterns = [r"<.*?>", "国家"]
    replacements = ["/replace_tag", "中国"]
    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "replaced_text",
        las_udf(
            RegexReplacer,
            construct_args={"patterns": patterns, "replacements": replacements},
        )(col("text")),
    )
    ds.show()

    # ╭───────────────────────────────────────┬────────────────────────────────╮
    # │ text                                  ┆ replaced_text                  │
    # │ ---                                   ┆ ---                            │
    # │ Utf8                                  ┆ Utf8                           │
    # ╞═══════════════════════════════════════╪════════════════════════════════╡
    # │ <Query></Query><Title>提升党的领导力…    ┆ /replace_tag/replace_tag/repl… │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    # │ None                                  ┆ None                           │
    # ╰───────────────────────────────────────┴────────────────────────────────╯
