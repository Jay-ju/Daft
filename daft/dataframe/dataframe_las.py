# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from daft.api_annotations import DataframePublicAPI
from daft.daft import IOConfig
from daft.las.infra.las_dataset import (
    LasDatasetClient,
    LasDatasetConfig,
    LasDatasetInfo,
    las_dataset_format,
    las_dataset_privacy,
)
from daft.las.io import TOSConfig
from daft.las.io.factory import rm

if TYPE_CHECKING:
    import pathlib

    import pyiceberg

    from daft.dataframe.dataframe import DataFrame
    from daft.schema import Schema
    from daft.utils import ColumnInputType


@dataclass
class CreateLasDatasetOptions:
    nick_name: str | None = None
    privacy: str = "Private"
    description: str = ""


@dataclass
class WriteOptions(ABC):
    io_config: IOConfig


@dataclass
class CsvWriteOptions(WriteOptions):
    root_dir: str | pathlib.Path | None = None
    write_mode: Literal["append", "overwrite", "overwrite-partitions"] = "append"
    partition_cols: list[ColumnInputType] | None = None


@dataclass
class JsonWriteOptions(WriteOptions):
    root_dir: str | pathlib.Path | None = None
    write_mode: Literal["append", "overwrite", "overwrite-partitions"] = "append"
    partition_cols: list[ColumnInputType] | None = None


@dataclass
class ParquetWriteOptions(WriteOptions):
    root_dir: str | pathlib.Path | None = None
    compression: str = "snappy"
    write_mode: Literal["append", "overwrite", "overwrite-partitions"] = "append"
    partition_cols: list[ColumnInputType] | None = None


@dataclass
class IcebergWriteOptions(WriteOptions):
    table: pyiceberg.table.Table | None = None
    mode: str = "append"


@dataclass
class LanceWriteOptions(WriteOptions):
    uri: str | pathlib.Path | None = None
    mode: Literal["create", "append", "overwrite"] = "create"
    schema: Schema | None = None


def _check_format_write_options(
    format: str, expected: WriteOptions, actual: WriteOptions | None = None
) -> WriteOptions:
    if actual is None:
        return expected
    if not isinstance(actual, expected.__class__):
        raise ValueError(f"Miss match format and write_options: format {format} but with write_options: {type(actual)}")
    return actual


def _check_write_options(format: str | None = None, write_options: WriteOptions | None = None) -> WriteOptions:
    if format == "csv":
        return _check_format_write_options(
            format="csv",
            expected=CsvWriteOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config())),
            actual=write_options,
        )
    if format == "parquet":
        return _check_format_write_options(
            format="parquet",
            expected=ParquetWriteOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config())),
            actual=write_options,
        )
    if format == "iceberg":
        return _check_format_write_options(
            format="iceberg",
            expected=IcebergWriteOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config())),
            actual=write_options,
        )
    if format == "lance":
        return _check_format_write_options(
            format="lance",
            expected=LanceWriteOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config())),
            actual=write_options,
        )
    if format == "json" or format == "jsonl":
        return _check_format_write_options(
            format="json",
            expected=JsonWriteOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config())),
            actual=write_options,
        )

    raise ValueError(f"Not supported format: {format}")


def _extract_privacy(create_ds_options: CreateLasDatasetOptions | None = None) -> str:
    if create_ds_options is None:
        return "private"
    if create_ds_options.privacy not in las_dataset_privacy.keys():
        raise ValueError("Invalid privacy value, only 'Private' or 'Public' allowed.")
    return create_ds_options.privacy.lower()


def _extract_format(format: str | None) -> str | None:
    if format is None:
        return None
    format = format.lower()
    if format not in las_dataset_format.keys():
        raise ValueError(f"Invalid format value, {las_dataset_format.keys()} are allowed.")
    return format


def _extract_root_dir(write_options: WriteOptions | None) -> str | None:
    if write_options is None:
        root_dir = None
    else:
        if isinstance(write_options, LanceWriteOptions):
            root_dir = write_options.uri
        elif isinstance(write_options, (ParquetWriteOptions, CsvWriteOptions, JsonWriteOptions)):
            root_dir = write_options.root_dir
        else:
            root_dir = None
    if root_dir is not None:
        root_dir = str(root_dir)
        if not root_dir.startswith("tos://") and not root_dir.startswith("s3://"):
            raise ValueError("Only tos or s3 path is supported")
        root_dir = root_dir.replace("s3://", "tos://")
    return root_dir


def _extract_mode(write_options: WriteOptions) -> str | None:
    import os

    # Overwrite is a non-safe operation for certain formats like csv, parquet, json.
    enable_overwrite_default = False

    if not (hasattr(write_options, "mode") or hasattr(write_options, "write_mode")):
        return None

    if isinstance(write_options, LanceWriteOptions):
        mode = write_options.mode
        if mode not in ["create", "append", "overwrite"]:
            raise ValueError(f"Invalid mode: {mode}, 'create', 'append' or 'overwrite' are allowed.")
        # Overwrite lance is safe
        enable_overwrite_default = True

    elif isinstance(write_options, IcebergWriteOptions):
        mode = write_options.mode  # type: ignore[assignment]
        if mode != "append":
            raise ValueError(f"Invalid mode: {mode}, only 'append' is allowed.")

    elif isinstance(write_options, (CsvWriteOptions, ParquetWriteOptions, JsonWriteOptions)):
        mode = write_options.write_mode  # type: ignore[assignment]
        if mode not in ["append", "overwrite", "overwrite-partitions"]:
            raise ValueError(f"Invalid mode: {mode}, 'append' 'overwrite' or 'overwrite-partitions' are allowed.")

    else:
        raise ValueError("Invalid write_options, only 'csv', 'json', 'iceberg', 'lance', 'parquet' are allowed.")

    enable_overwrite = os.environ.get("ENABLE_OVERWRITE_LAS_DATASET", enable_overwrite_default)
    if (not enable_overwrite) and (mode == "overwrite" or mode == "overwrite-partitions"):  # type: ignore[comparison-overlap]
        raise ValueError("Overwrite is not supported now")

    return mode


def _normalize(format: str) -> str:
    if format in ("json", "jsonl"):
        return "json"
    return format


@DataframePublicAPI
def write_las_dataset(  # type: ignore[no-untyped-def]
    self,
    name: str,
    format: str | None = None,
    write_options: WriteOptions | None = None,
    create_ds_options: CreateLasDatasetOptions | None = None,
) -> DataFrame:
    """Writes the DataFrame to LAS dataset, returning a new DataFrame with paths to the files that were written.

    The dataset will be created if it was not exist.
    Files may be written to `<root_dir>/*` with randomly generated UUIDs as the file names.

    Args:
        name: Name of the dataset to be created.
        format: Format of the dataset (CSV, Parquet, Lance, Jsonl or Iceberg).
        write_options: The format specified write options. These options will be passed to correspond writing functions.
        create_ds_options: The options used to create the dataset.

    Returns:
        DataFrame: A new DataFrame containing paths to the written files.

    Raises:
        ValueError: If the format is unsupported or if required arguments are missing.
        Exception: If dataset creation fails in the LAS service.

    Examples:
        >>> df.write_las_dataset(
        ...     name="my_dataset",
        ...     format="CSV",
        ...     write_options=CsvWriteOptions(root_dir="tos://path/to/my/csv/"),
        ... )

    Note:
        For iceberg format, the 'table' parameter must be provided in kwargs.
        The method automatically registers the dataset with the LAS service after writing files.
    """
    write_options = _check_write_options(format, write_options)

    # Extract and check the parameters.
    privacy = _extract_privacy(create_ds_options)
    format = _extract_format(format=format)
    root_dir = _extract_root_dir(write_options=write_options)
    mode = _extract_mode(write_options=write_options)

    # Create a las dataset client.
    config = LasDatasetConfig.from_io_config(write_options.io_config)
    client = LasDatasetClient(config)

    # Check if the dataset exists.
    dataset_exists = False
    if client.dataset_exist(name=name):
        dataset_exists = True

        dataset_info = client.get_dataset(name=name)
        format_from_las = dataset_info.format.lower()  # type: ignore
        root_dir_from_las = dataset_info.data_path

        if format is None:
            format = format_from_las
        if root_dir is None:
            root_dir = root_dir_from_las

        if _normalize(format) != _normalize(format_from_las):
            raise ValueError(
                f"The dataset {name} already exists, but the data format it records: "
                f"{format_from_las} is not consistent with that you specified: {format}"
            )
        if root_dir != root_dir_from_las:
            raise ValueError(
                f"The dataset {name} already exists, but the data path it records: "
                f"{root_dir_from_las} is not consistent with the 'root_dir' you specified: {root_dir}"
            )
        if mode == "create" and format == "lance":
            raise ValueError("'create' mode is not allowed for existing dataset with lance format")
    else:
        if create_ds_options is None:
            raise ValueError("Dataset not exist, and arg 'create_ds_options' is not provided")
        if root_dir is None:
            raise ValueError("You must specify arg 'root_dir'/'url' for writing data")
        if format is None:
            format = "lance"

    assert root_dir is not None
    root_dir = root_dir.replace("tos://", "s3://")

    if hasattr(write_options, "root_dir"):
        write_options.root_dir = root_dir
    elif hasattr(write_options, "uri"):
        write_options.uri = root_dir

    if format == "csv":
        result_df = self.write_csv(**vars(write_options))
    elif format == "parquet":
        result_df = self.write_parquet(**vars(write_options))
    elif format == "lance":
        result_df = self.write_lance(**vars(write_options))
    elif format == "iceberg":
        raise NotImplementedError()
    elif format == "json" or format == "jsonl":
        result_df = self.write_json(**vars(write_options))
    else:
        raise ValueError(f"Unsupported format: {format}")

    # If the dataset already exists, just return
    if dataset_exists:
        return result_df

    # Create new las dataset
    dataset = LasDatasetInfo(
        name=name,
        format=las_dataset_format[format],
        nick_name=create_ds_options.nick_name,  # type: ignore[union-attr]
        storage="TOS",
        data_path=root_dir.replace("s3://", "tos://"),
        privacy=las_dataset_privacy[privacy],
        description=create_ds_options.description,  # type: ignore[union-attr]
    )
    try:
        client.create_dataset(dataset=dataset)
    except Exception:
        rm(root_dir)
        raise

    return result_df
