# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Literal

import daft
from daft import col
from daft.api_annotations import PublicAPI
from daft.daft import IOConfig
from daft.dataframe import DataFrame
from daft.io._csv import read_csv
from daft.io._json import read_json
from daft.io._parquet import read_parquet
from daft.io.lance._lance import read_lance
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig
from daft.las.io import TOSConfig, exists

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

    from daft import DataFrame
    from daft.datatype import DataType, TimeUnit


@dataclass
class ReadOptions:
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
    table: str | PyIcebergTable | None = None
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


@dataclass
class FolderReadOptions(ReadOptions):
    # May read metadata.csv/jsonl/parquet
    metadata_options: ReadOptions = field(
        default_factory=lambda: ReadOptions(io_config=IOConfig(s3=TOSConfig.from_env().to_s3_config()))
    )
    file_types: list[str] = field(default_factory=list)
    read_type: Literal["url", "binary", "base64", "ndarray"] = "url"


@dataclass
class ImageFolderReadOptions(FolderReadOptions):
    # TODO: add more read options, like image related features
    file_types: list[str] = field(
        default_factory=lambda: [
            ".blp",
            ".dib",
            ".bmp",
            ".bufr",
            ".cur",
            ".pcx",
            ".dcx",
            ".dds",
            ".ps",
            ".eps",
            ".fit",
            ".fits",
            ".fli",
            ".flc",
            ".ftc",
            ".ftu",
            ".gbr",
            ".gif",
            ".grib",
            # ".h5",   # may contain zero or several images
            # ".hdf",  # may contain zero or several images
            ".png",
            ".apng",
            ".jp2",
            ".j2k",
            ".jpc",
            ".jpf",
            ".jpx",
            ".j2c",
            ".icns",
            ".ico",
            ".im",
            ".iim",
            ".tif",
            ".tiff",
            ".jfif",
            ".jpe",
            ".jpg",
            ".jpeg",
            ".mpg",
            ".mpeg",
            ".msp",
            ".pcd",
            ".pxr",
            ".pbm",
            ".pgm",
            ".ppm",
            ".pnm",
            ".psd",
            ".bw",
            ".rgb",
            ".rgba",
            ".sgi",
            ".ras",
            ".tga",
            ".icb",
            ".vda",
            ".vst",
            ".webp",
            ".wmf",
            ".emf",
            ".xbm",
            ".xpm",
        ]
    )


@dataclass
class AudioFolderReadOptions(FolderReadOptions):
    # TODO: add more read options
    file_types: list[str] = field(
        default_factory=lambda: [
            ".aiff",
            ".au",
            ".avr",
            ".caf",
            ".flac",
            ".htk",
            ".svx",
            ".mat4",
            ".mat5",
            ".mpc2k",
            ".m4a",
            ".ogg",
            ".paf",
            ".pvf",
            ".raw",
            ".rf64",
            ".sd2",
            ".sds",
            ".ircam",
            ".voc",
            ".w64",
            ".wav",
            ".nist",
            ".wavex",
            ".wve",
            ".xi",
            ".mp3",
            ".opus",
        ]
    )


@dataclass
class VideoFolderReadOptions(FolderReadOptions):
    # TODO: add more read options
    file_types: list[str] = field(
        default_factory=lambda: [
            ".mkv",
            ".mp4",
            ".avi",
            ".mpeg",
            ".mov",
        ]
    )


def _check_read_options(format: str | None = None, read_options: ReadOptions | None = None) -> ReadOptions:
    if read_options is None and format in ["csv", "parquet", "iceberg", "lance", "json"]:
        return ReadOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config()))

    if read_options is None and format in ["image", "audio", "video"]:
        return FolderReadOptions(IOConfig(s3=TOSConfig.from_env().to_s3_config()))

    if (
        (format == "csv" and not isinstance(read_options, CsvReadOptions))
        or (format == "parquet" and not isinstance(read_options, ParquetReadOptions))
        or (format == "iceberg" and not isinstance(read_options, IcebergReadOptions))
        or (format == "lance" and not isinstance(read_options, LanceReadOptions))
        or ((format == "json" or format == "jsonl") and not isinstance(read_options, JsonReadOptions))
        or (format == "image" and not isinstance(read_options, ImageFolderReadOptions))
        or (format == "audio" and not isinstance(read_options, AudioFolderReadOptions))
        or (format == "video" and not isinstance(read_options, VideoFolderReadOptions))
    ):
        raise ValueError(
            f"Miss match format and read_options: format: {format} but with write_options: {type(read_options)}"
        )

    return read_options  # type: ignore[return-value]


@PublicAPI
def read_las_dataset(name: str, read_options: ReadOptions | None = None) -> DataFrame:
    """Create a DataFrame from a las dataset.

    Args:
        name: Name of the dataset.
        read_options: The format specified read options. These options will be passed to correspond read functions.

    Returns:
        DataFrame: a DataFrame with the schema converted from the dataset.
            For audio/image/video folder, if there's a metadata.csv/metadata.jsonl/metadata.parquet,
            this will return the content the metadata file as dataframe, otherwise, this will return
            a dataframe whose content is the file list, with the following schema:

            1. bytes/base64: the content of the file, bytes or base64 according to the reading options,
                and if only read url, this column does not exist.
            2. path: the path to the file/directory
            3. size: size of the object in bytes
            4. rows: number of rows if the file is parquet
    """
    io_config = None if read_options is None else read_options.io_config
    config = LasDatasetConfig.from_io_config(io_config)
    client = LasDatasetClient(config)

    if not (client.dataset_exist(name=name)):
        raise ValueError(f"Dataset {name} not exist!")

    dataset_info = client.get_dataset(name=name)

    format = dataset_info.format.lower()  # type: ignore
    assert dataset_info.data_path is not None
    read_options = _check_read_options(format, read_options)

    path = dataset_info.data_path.replace("tos://", "s3://").rstrip("/")

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
    elif format == "image":
        return _read_image_folder(path=path, read_options=read_options)  # type: ignore[arg-type]
    elif format == "audio":
        return _read_audio_folder(path=path, read_options=read_options)  # type: ignore[arg-type]
    elif format == "video":
        return _read_video_folder(path=path, read_options=read_options)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported data format: {format}")


def _read_folder(path: str, format: str, read_options: FolderReadOptions) -> DataFrame:
    # Read the image/audio/video folder. The behavior is like huggingface datasets:
    # if there's metadata.csv/json/jsonl/parquet, read it as the folder dataframe,
    # otherwise read the files list.
    from daft.las.functions import las_udf
    from daft.las.functions.file.download import LoadFileBase64, LoadFileBytes

    # 1. Read metadata if there's metadata file
    metadata_csv = path + "/metadata.csv"
    metadata_json = path + "/metadata.json"
    metadata_jsonl = path + "/metadata.jsonl"
    metadata_parquet = path + "/metadata.parquet"

    if exists(metadata_csv):
        return daft.read_csv(metadata_csv, **vars(read_options.metadata_options))
    elif exists(metadata_json):
        return daft.read_json(metadata_json, **vars(read_options.metadata_options))
    elif exists(metadata_jsonl):
        return daft.read_json(metadata_jsonl, **vars(read_options.metadata_options))
    elif exists(metadata_parquet):
        return daft.read_parquet(metadata_parquet, **vars(read_options.metadata_options))

    # 2. Read file list if there isn't metadata file
    # TODO: support ndarray
    if read_options.read_type in ["ndarray"]:
        raise ValueError(f"{read_options.read_type} is not supported now")

    if read_options.read_type not in ["url", "binary", "base64", "ndarray"]:
        raise ValueError(
            f"Unsupported read_type: {read_options.read_type}, 'url', 'binary', 'base64', 'ndarray' are allowed."
        )

    path = path + "/**"
    io_config = None if read_options is None else read_options.io_config
    dataframe = daft.from_glob_path(path=path, io_config=io_config)

    suffixes = None if read_options is None else read_options.file_types
    if suffixes:
        condition = reduce(
            lambda acc, suffix: acc | col("path").str.endswith(suffix),
            suffixes[1:],
            col("path").str.endswith(suffixes[0]),
        )
        dataframe = dataframe.where(condition)
    else:
        dataframe = dataframe

    if read_options.read_type == "url":
        return dataframe.with_column_renamed("path", format)
    elif read_options.read_type == "binary":
        return dataframe.with_column("bytes", las_udf(operator=LoadFileBytes)(col("path")))
    elif read_options.read_type == "base64":
        return dataframe.with_column("base64", las_udf(operator=LoadFileBase64)(col("path")))
    else:
        raise ValueError(f"Unsupported read_type: {read_options.read_type}")


def _read_image_folder(path: str, read_options: ImageFolderReadOptions) -> DataFrame:
    return _read_folder(path=path, format="image", read_options=read_options)


def _read_audio_folder(path: str, read_options: AudioFolderReadOptions) -> DataFrame:
    return _read_folder(path=path, format="audio", read_options=read_options)


def _read_video_folder(path: str, read_options: VideoFolderReadOptions) -> DataFrame:
    return _read_folder(path=path, format="video", read_options=read_options)
