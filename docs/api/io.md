# I/O

Daft offers a variety of approaches to creating a DataFrame from reading various data sources (in-memory data, files, data catalogs, and integrations) and writing to various data sources. See more about other [Connectors](../connectors/index.md) in Daft User Guide.

## Input

<!-- from_ -->

::: daft.from_arrow
    options:
        heading_level: 3

::: daft.from_dask_dataframe
    options:
        heading_level: 3

::: daft.from_glob_path
    options:
        heading_level: 3

::: daft.from_pandas
    options:
        heading_level: 3

::: daft.from_pydict
    options:
        heading_level: 3

::: daft.from_pylist
    options:
        heading_level: 3

::: daft.from_ray_dataset
    options:
        heading_level: 3

<!-- read_ -->

::: daft.read_csv
    options:
        heading_level: 3

::: daft.read_deltalake
    options:
        heading_level: 3

::: daft.read_hudi
    options:
        heading_level: 3

::: daft.read_iceberg
    options:
        heading_level: 3

::: daft.read_json
    options:
        heading_level: 3

::: daft.read_lance
    options:
        heading_level: 3

::: daft.read_parquet
    options:
        heading_level: 3

::: daft.read_sql
    options:
        heading_level: 3

::: daft.read_video_frames
    options:
        heading_level: 3

::: daft.read_warc
    options:
        heading_level: 3

::: daft.read_huggingface
    options:
        heading_level: 3

::: daft.sql.sql.sql
    options:
        heading_level: 3

## Output

<!-- write_ -->

::: daft.dataframe.DataFrame.write_csv
    options:
        heading_level: 3

::: daft.dataframe.DataFrame.write_deltalake
    options:
        heading_level: 3

::: daft.dataframe.DataFrame.write_iceberg
    options:
        heading_level: 3

::: daft.dataframe.DataFrame.write_lance
    options:
        heading_level: 3

::: daft.dataframe.DataFrame.write_parquet
    options:
        heading_level: 3

## FilenameProvider

Daft 允许通过 [`daft.io.FilenameProvider`][daft.io.FilenameProvider] 来自定义写入时生成的文件名。

在以下场景中可以使用 `filename_provider`：

- `DataFrame.write_parquet(..., filename_provider=...)`
- `DataFrame.write_csv(..., filename_provider=...)`
- `daft.functions.url.upload(..., filename_provider=...)`

`FilenameProvider` 会在一次逻辑写入开始时收到一个 `write_uuid`，同一次写入产生的所有文件都会复用这个 UUID。根据不同的写入模式，Daft 会调用：

- `get_filename_for_block(write_uuid, task_index, block_index, file_idx, ext)`：块级写入，例如 `write_parquet` / `write_csv`。
- `get_filename_for_row(row, write_uuid, task_index, block_index, row_index, ext)`：行级写入，例如 `functions.url.upload` 在单目录上传时。

Daft 内置的默认实现 [`_DefaultFilenameProvider`][daft.io.filename_provider._DefaultFilenameProvider] 使用如下模式生成文件名：

```text
<write_uuid>_<task_index>_<block_index>_<row_or_file_index>.<ext>
```

### 示例：自定义 Parquet 文件名

```python
import daft
from daft.io import FilenameProvider


class LabelledParquetProvider(FilenameProvider):
    def get_filename_for_block(self, write_uuid, task_index, block_index, file_idx, ext):
        return f"part-{task_index}-{block_index}-{file_idx}.{ext}"

    def get_filename_for_row(self, row, write_uuid, task_index, block_index, row_index, ext):
        # 不用于块级写入，可以选择直接抛错
        raise NotImplementedError


df = daft.from_pydict({"x": [1, 2, 3]})
# 生成文件名类似 "part-0-0-0.parquet"
df.write_parquet("output_dir", filename_provider=LabelledParquetProvider())
```

### 示例：upload 单目录按行命名

```python
import os

import daft
from daft.io import FilenameProvider


class ImageUploadProvider(FilenameProvider):
    def get_filename_for_block(self, *args, **kwargs):  # pragma: no cover - 未使用
        raise NotImplementedError

    def get_filename_for_row(self, row, write_uuid, task_index, block_index, row_index, ext):
        return f"image-{row_index}.bin"


df = daft.from_pydict({"bytes": [b"a", b"b", b"c"]})
folder = "./uploads"

# 最终写出的本地路径形如 "uploads/image-0.bin"、"uploads/image-1.bin" ...
df = df.with_column("path", df["bytes"].upload(folder, filename_provider=ImageUploadProvider()))
result = df.collect()
print(result.to_pydict()["path"])
```

## User-Defined

Daft supports diverse input sources and output sinks, this section covers lower-level APIs which we are evolving for more advanced usage.

!!! warning "Warning"

    These APIs are considered experimental.

::: daft.io.source.DataSource
    options:
        filters: ["!^_"]
        heading_level: 3

::: daft.io.source.DataSourceTask
    options:
        filters: ["!^_"]
        heading_level: 3

::: daft.io.sink.DataSink
    options:
        filters: ["!^_"]
        heading_level: 3

::: daft.io.sink.WriteResult
    options:
        filters: ["!^_"]
        heading_level: 3

## Pushdowns

Daft supports predicate, projection, and limit pushdowns.

::: daft.io.pushdowns.Pushdowns
    options:
        filters: ["!^_"]
        heading_level: 3

::: daft.io.scan.ScanOperator
    options:
        filters: ["!^_"]
