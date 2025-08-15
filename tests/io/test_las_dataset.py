from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import daft
from daft.io import (
    CreateLasDatasetOptions,
    CsvReadOptions,
    CsvWriteOptions,
    IcebergReadOptions,
    IcebergWriteOptions,
    IOConfig,
    JsonReadOptions,
    JsonWriteOptions,
    LanceReadOptions,
    LanceWriteOptions,
    ParquetReadOptions,
    ParquetWriteOptions,
    ReadOptions,
    WriteOptions,
)
from daft.io._las_dataset import AudioFolderReadOptions
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig
from daft.las.io import TOSConfig

# currently "json" is not supported by distributed runner
formats = ["csv", "parquet", "lance"]

data = {"name": ["Bush", "Obama", "Trump"], "age": [79, 64, 79]}
dataframe: daft.DataFrame = daft.from_pydict(data)


def _format_specified_options(
    format: str, root_dir: str
) -> tuple[ReadOptions, WriteOptions, WriteOptions, WriteOptions, WriteOptions]:
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())
    if format == "csv":
        read_options = CsvReadOptions(io_config=io_config)
        write_options = CsvWriteOptions(root_dir=root_dir, io_config=io_config)
        write_options_root_dir = dataclasses.replace(write_options, root_dir="tos://another/dir")
        write_options_append = dataclasses.replace(write_options)
        write_options_overwrite = dataclasses.replace(write_options, write_mode="overwrite")
    elif format == "parquet":
        read_options = ParquetReadOptions(io_config=io_config)
        write_options = ParquetWriteOptions(root_dir=root_dir, io_config=io_config)
        write_options_root_dir = dataclasses.replace(write_options, root_dir="tos://another/dir")
        write_options_append = dataclasses.replace(write_options)
        write_options_overwrite = dataclasses.replace(write_options, write_mode="overwrite")
    elif format == "lance":
        read_options = LanceReadOptions(io_config=io_config)
        write_options = LanceWriteOptions(uri=root_dir, io_config=io_config)
        write_options_root_dir = dataclasses.replace(write_options, uri="tos://another/dir")
        write_options_append = dataclasses.replace(write_options, mode="append")
        write_options_overwrite = dataclasses.replace(write_options, mode="overwrite")
    elif format == "json":
        read_options = JsonReadOptions(io_config=io_config)
        write_options = JsonWriteOptions(root_dir=root_dir, io_config=io_config)
        write_options_root_dir = dataclasses.replace(write_options, root_dir="tos://another/dir")
        write_options_append = dataclasses.replace(write_options)
        write_options_overwrite = dataclasses.replace(write_options, write_mode="overwrite")
    elif format == "iceberg":
        # TODO: iceberg is not supported now
        read_options = IcebergReadOptions(io_config=io_config, table="")
        write_options = IcebergWriteOptions(io_config=io_config, table="")
        write_options_root_dir = dataclasses.replace(write_options)
        write_options_append = dataclasses.replace(write_options)
        write_options_overwrite = dataclasses.replace(write_options)
    else:
        raise ValueError(f"Unsupported format: {format}")
    return read_options, write_options, write_options_root_dir, write_options_append, write_options_overwrite


@pytest.mark.parametrize("format", formats)
def test_las_dataset(format, uuid_short, object_store_test_dir, monkeypatch):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

    client = LasDatasetClient(LasDatasetConfig())

    dataset_name = "dataset_" + uuid_short
    dataset_nickname = dataset_name
    root_dir = f"{object_store_test_dir}/{dataset_name}"
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())

    read_options, write_options, write_options_root_dir, write_options_append, write_options_overwrite = (
        _format_specified_options(format=format, root_dir=root_dir)
    )

    create_ds_options = CreateLasDatasetOptions(
        nick_name=dataset_nickname, privacy="private", description="This is test dataset"
    )

    # 1. confirm the dataset does not exist
    assert client.dataset_exist(dataset_name) is False

    # 2. write non-exist dataset, but didn't provide create_ds_options
    with pytest.raises(ValueError, match=r"Dataset not exist, and arg 'create_ds_options' is not provided"):
        dataframe.write_las_dataset(name=dataset_name, format=format)

    # 3. the write options mismatch with the format
    with pytest.raises(ValueError, match=r"Miss match format and write_options.*"):
        dataframe.write_las_dataset(name=dataset_name, format=format, write_options=WriteOptions(io_config=io_config))

    # 4. write the dataset and make sure it has been created
    dataframe.write_las_dataset(
        name=dataset_name, format=format, write_options=write_options, create_ds_options=create_ds_options
    )
    assert client.dataset_exist(dataset_name) is True

    # 5. read the dataset and check the result
    actual = daft.read_las_dataset(name=dataset_name, read_options=read_options)
    pd.testing.assert_frame_equal(dataframe.to_pandas(), actual.to_pandas())

    # 6. write the existing dataset, but provide different root_dir
    with pytest.raises(ValueError, match=r".*already exists, but the data path it records.*"):
        dataframe.write_las_dataset(
            name=dataset_name,
            format=format,
            write_options=write_options_root_dir,
        )

    # 7. write to the existing dataset (append)
    dataframe.write_las_dataset(name=dataset_name, format=format, write_options=write_options_append)

    # 8. read the dataset and check the result
    actual = daft.read_las_dataset(name=dataset_name, read_options=read_options)
    expected = pd.concat([dataframe.to_pandas(), dataframe.to_pandas()])
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.to_pandas())

    # 9. overwrite is not supported for certain formats
    if format in ["csv", "parquet", "json"]:
        with pytest.raises(ValueError, match=r"Overwrite is not supported now.*"):
            dataframe.write_las_dataset(
                name=dataset_name,
                format=format,
                write_options=write_options_overwrite,
            )


def test_read_folder(monkeypatch):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

    expected_meta = {
        "file_name": [
            "02 - Sad But True.uncompressed_NotWorking.flac",
            "When I Grow Up.flac",
            "figaro.flac",
            "file_doesnt_work.m4a",
        ],
        "size": [1000, 1000, 1000, 1000],
    }

    # 1. test read metadata
    dataset_name = "daft_test_audio_folder"
    df = daft.read_las_dataset(name=dataset_name)
    assert df.to_pydict() == expected_meta

    # 2. no metadata, read the file list, and read file as bytes
    expected_meta = {
        "bytes": [b"This is a mock audio"],
        "num_rows": [None],
        "path": ["s3://las-ci/daft/dataset/audio_without_meta/mock-audio.mp3"],
        "size": [20],
    }
    dataset_name = "daft_test_audio_folder_without_meta"
    read_options = AudioFolderReadOptions(
        read_type="binary", io_config=IOConfig(s3=TOSConfig.from_env().to_s3_config())
    )
    df = daft.read_las_dataset(name=dataset_name, read_options=read_options)
    assert df.to_pydict() == expected_meta

    # 3. no metadata, read the file list, and read file as base64
    expected_meta = {
        "base64": ["VGhpcyBpcyBhIG1vY2sgYXVkaW8="],
        "num_rows": [None],
        "path": ["s3://las-ci/daft/dataset/audio_without_meta/mock-audio.mp3"],
        "size": [20],
    }
    dataset_name = "daft_test_audio_folder_without_meta"
    read_options = AudioFolderReadOptions(
        read_type="base64", io_config=IOConfig(s3=TOSConfig.from_env().to_s3_config())
    )
    df = daft.read_las_dataset(name=dataset_name, read_options=read_options)
    assert df.to_pydict() == expected_meta
