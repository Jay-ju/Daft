from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.api_annotations import PublicAPI
from daft.dataframe import DataFrame
from daft.io._csv import read_csv
from daft.io._json import read_json
from daft.io._parquet import read_parquet
from daft.io.lance._lance import read_lance
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

    from daft import DataFrame
    from daft.daft import IOConfig
    from daft.datatype import DataType, TimeUnit


@dataclass
class ReadOptions(ABC):
    io_config: IOConfig


@dataclass
class CsvReadOptions(ReadOptions):
    infer_schema: bool = True
    schema: dict[str, DataType] | None = None
    has_headers: bool = True
    delimiter: str | None = None
    double_quote: bool = True
    quote: str | None = None
    escape_char: str | None = None
    comment: str | None = None
    allow_variable_columns: bool = False
    file_path_column: str | None = None
    hive_partitioning: bool = False
    schema_hints: dict[str, DataType] | None = None
    _buffer_size: int | None = None
    _chunk_size: int | None = None


@dataclass
class JsonReadOptions(ReadOptions):
    infer_schema: bool = True
    schema: dict[str, DataType] | None = None
    file_path_column: str | None = None
    hive_partitioning: bool = False
    schema_hints: dict[str, DataType] | None = None
    _buffer_size: int | None = None
    _chunk_size: int | None = None


@dataclass
class ParquetReadOptions(ReadOptions):
    row_groups: list[list[int]] | None = None
    infer_schema: bool = True
    schema: dict[str, DataType] | None = None
    file_path_column: str | None = None
    hive_partitioning: bool = False
    coerce_int96_timestamp_unit: str | TimeUnit | None = None
    schema_hints: dict[str, DataType] | None = None
    _multithreaded_io: bool | None = None
    _chunk_size: int | None = None


@dataclass
class IcebergReadOptions(ReadOptions):
    table: str | PyIcebergTable
    snapshot_id: int | None = None


@dataclass
class LanceReadOptions(ReadOptions):
    version: str | int | None = None
    asof: str | None = None
    block_size: int | None = None
    commit_lock: object | None = None
    index_cache_size: int | None = None
    default_scan_options: dict[str, str] | None = None
    metadata_cache_size_bytes: int | None = None


def _check_read_options(format: str | None = None, read_options: ReadOptions | None = None) -> None:
    if format is None or read_options is None:
        return
    if (
        (format == "csv" and not isinstance(read_options, CsvReadOptions))
        or (format == "parquet" and not isinstance(read_options, ParquetReadOptions))
        or (format == "iceberg" and not isinstance(read_options, IcebergReadOptions))
        or (format == "lance" and not isinstance(read_options, LanceReadOptions))
        or ((format == "json" or format == "jsonl") and not isinstance(read_options, JsonReadOptions))
    ):
        raise ValueError(
            f"Miss match format and read_options: format: {format} but with write_options: {type(read_options)}"
        )


@PublicAPI
def read_las_dataset(name: str, read_options: ReadOptions | None = None) -> DataFrame:
    """Create a DataFrame from a las dataset.

    Args:
        name: Name of the dataset.
        io_config: IOConfig for reading the storage
        **kwargs: Additional format-specific arguments

    Returns:
        DataFrame: a DataFrame with the schema converted from the dataset.
    """
    io_config = None if read_options is None else read_options.io_config
    config = LasDatasetConfig.from_io_config(io_config)
    client = LasDatasetClient(config)

    if not (client.dataset_exist(name=name)):
        raise ValueError(f"Dataset {name} not exist!")

    dataset_info = client.get_dataset(name=name)

    format = dataset_info.format.lower()  # type: ignore
    assert dataset_info.data_path is not None
    _check_read_options(format, read_options)

    path = dataset_info.data_path
    if path.startswith("tos://"):
        path = path.replace("tos://", "s3://")

    if format == "csv":
        return read_csv(path=path, **vars(read_options))
    elif format == "parquet":
        return read_parquet(path=path, **vars(read_options))
    elif format == "lance":
        return read_lance(url=path, **vars(read_options))
    elif format == "iceberg":
        raise NotImplementedError()
    elif format == "json" or format == "jsonl":
        return read_json(path=path, **vars(read_options))
    else:
        raise ValueError(f"Unsupported data format: {format}")
