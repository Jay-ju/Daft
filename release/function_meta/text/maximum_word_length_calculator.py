from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.maximum_word_length_calculator import MaximumWordLengthCalculator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "Hello world 你好世界",
            "Python编程 is fun",
            "这是一个中文句子",
            "The quick brown fox jumps over the lazy dog",
            "supercalifragilisticexpialidocious is a very long word",
            None,
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "max_word_length",
        las_udf(
            MaximumWordLengthCalculator,
            construct_args={},
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────╮
    # │ text                                                                         ┆ max_word_length                              │
    # │ ---                                                                          ┆ ---                                          │
    # │ Utf8                                                                         ┆ Int64                                        │
    # ╞══════════════════════════════════════════════════════════════════════════════╪══════════════════════════════════════════════╡
    # │ Hello world 你好世界                                                          ┆ 5                                            │
    # │ Python编程 is fun                                                             ┆ 6                                            │
    # │ 这是一个中文句子                                                               ┆ 0                                            │
    # │ The quick brown fox jumps over the lazy dog                                 ┆ 5                                            │
    # │ supercalifragilisticexpialidocious is a very long word                      ┆ 34                                           │
    # │ None                                                                         ┆ 0                                            │
    # ╰──────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────╯
