from __future__ import annotations

import dataclasses
import os

import pandas as pd
import pytest

import daft
from daft.dataframe.dataframe_las import FolderWriteOptions
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
from daft.las.io.factory import rm

daft.set_execution_config(actor_udf_ready_timeout=600)

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

    # 2. write non-exist dataset, but didn't provide 'root_dir/url'
    with pytest.raises(ValueError, match=r"You must specify arg 'root_dir'/'url' for writing data"):
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
    monkeypatch.setenv("ENABLE_OVERWRITE_LAS_DATASET", "True")
    write_options.write_mode = "overwrite"
    if format in ["csv", "parquet", "json"]:
        with pytest.raises(ValueError, match=r"Overwrite is not supported now.*"):
            dataframe.write_las_dataset(
                name=dataset_name,
                format=format,
                write_options=write_options_overwrite,
            )

    # 10. remove the las dataset
    try:
        client.delete_dataset(name=dataset_name)
    except:  # noqa: E722
        pass
    assert client.dataset_exist(name=dataset_name) is False


def test_read_folder():
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
        "path": [f"s3://{os.getenv('TEST_OBJECT_BUCKET', 'daft-ci')}/las-dataset/audio_without_meta/mock-audio.mp3"],
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
        "path": [f"s3://{os.getenv('TEST_OBJECT_BUCKET', 'daft-ci')}/las-dataset/audio_without_meta/mock-audio.mp3"],
        "size": [20],
    }
    dataset_name = "daft_test_audio_folder_without_meta"
    read_options = AudioFolderReadOptions(
        read_type="base64", io_config=IOConfig(s3=TOSConfig.from_env().to_s3_config())
    )
    df = daft.read_las_dataset(name=dataset_name, read_options=read_options)
    assert df.to_pydict() == expected_meta


@pytest.mark.skip("Temporarily skip")
def test_write_folder(uuid_short, monkeypatch):
    metadata = [
        {"file_name": "02 - Sad But True.uncompressed_NotWorking.flac", "size": 1000},
        {"file_name": "When I Grow Up.flac", "size": 1000},
        {"file_name": "figaro.flac", "size": 1000},
        {"file_name": "file_doesnt_work.m4a", "size": 1000},
    ]
    df = daft.from_pylist(metadata)

    client = LasDatasetClient(LasDatasetConfig())

    dataset_name = "daft_test_write_audio_folder" + uuid_short
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())
    root_dir = f"tos://{os.getenv('TEST_OBJECT_BUCKET', 'daft-ci')}/las-dataset/audio_without_meta_for_write_test"
    write_options = FolderWriteOptions(io_config=io_config)

    # ensure there isn't metadata
    rm(
        f"tos://{os.getenv('TEST_OBJECT_BUCKET', 'daft-ci')}/las-dataset/audio_without_meta_for_write_test/metadata.jsonl"
    )

    # 1. write without root_dir
    with pytest.raises(ValueError, match=r"You must specify arg 'root_dir'/'url' for writing data*"):
        df.write_las_dataset(name="daft_write_folder_test", format="audio", write_options=write_options)

    # 2. test write metadata and create audio dataset
    write_options = FolderWriteOptions(io_config=io_config, root_dir=root_dir)
    df.write_las_dataset(name=dataset_name, format="audio", write_options=write_options)

    assert client.dataset_exist(dataset_name) is True

    # 3. read the dataset
    df = daft.read_las_dataset(name=dataset_name)
    assert df.to_pylist() == metadata

    # 4. check the dataset
    assert client.dataset_exist(name=dataset_name) is True

    # 5. overwrite is not supported
    monkeypatch.setenv("ENABLE_OVERWRITE_LAS_DATASET", "True")
    write_options.write_mode = "overwrite"
    with pytest.raises(ValueError, match=r"Overwrite is not supported now.*"):
        df.write_las_dataset(
            name=dataset_name,
            format="audio",
            write_options=write_options,
        )

    # 6. clear the metadata and dataset created above
    rm(
        f"tos://{os.getenv('TEST_OBJECT_BUCKET', 'daft-ci')}/las-dataset/audio_without_meta_for_write_test/metadata.jsonl"
    )
    try:
        client.delete_dataset(name=dataset_name)
    except:  # noqa: E722
        pass
    assert client.dataset_exist(name=dataset_name) is False
