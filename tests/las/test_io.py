# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import daft
from daft.daft import IOConfig
from daft.las.io import download_file, exists, file_size, rm, upload_file
from daft.las.io.factory import LasIOFactory
from daft.las.io.http import HttpIO
from daft.las.io.tos import TOSConfig
from daft.las.io.utils import generate_temp_file, normalize_local_path


def test_generate_temp_filename():
    path = "/a/b/c"
    temp_file = generate_temp_file(path)
    assert temp_file == f'/a/b/.c.temp-{datetime.now(UTC).strftime("%Y%m%d%H%M%S")}'

    temp_file = generate_temp_file(path, suffix="temp-suffix")
    assert temp_file == "/a/b/.c.temp-suffix"

    path = "file:///a/b/c"
    temp_file = generate_temp_file(path, suffix="temp-suffix")
    assert temp_file == "file:///a/b/.c.temp-suffix"

    path = "/a/b/"
    temp_file = generate_temp_file(path, suffix="temp-suffix")
    assert temp_file == "/a/.b.temp-suffix"

    path = "s3://bucket/a/b/c"
    temp_file = generate_temp_file(path, suffix="temp-suffix")
    assert temp_file == "s3://bucket/a/b/.c.temp-suffix"


def test_basic_io(tmpdir, object_store_test_dir):
    work_dir = str(tmpdir)
    rand_dir = uuid.uuid4()

    # 1. test on local filesystem
    remote_path = f"{work_dir}/{rand_dir}/test_basic_io.txt"
    _test_basic_io(remote_path, work_dir)

    # 2. test on local filesystem with fill path
    remote_path = f"file://{work_dir}/{rand_dir}/test_basic_io_full_path.txt"
    full_work_dir = f"file://{work_dir}"
    _test_basic_io(remote_path, full_work_dir)

    # 3. test on object store.
    remote_path = f"{object_store_test_dir}/test_basic_io.txt"
    _test_basic_io(remote_path, work_dir)

    full_work_dir = f"file://{work_dir}"
    _test_basic_io(remote_path, full_work_dir)


def test_with_daft_s3_config(tmpdir, object_store_test_dir, monkeypatch):
    tos_config = TOSConfig.from_env()

    monkeypatch.delenv("TOS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TOS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACCESS_KEY_ID", raising=False)

    io_config = IOConfig(s3=tos_config.to_s3_config())
    daft.set_planning_config(default_io_config=io_config)

    remote_path = f"{object_store_test_dir}/test_basic_io.txt"
    _test_basic_io(remote_path, str(tmpdir))


def test_with_s3_uri(tmpdir, object_store_test_dir, monkeypatch):
    tos_config = TOSConfig.from_env()

    monkeypatch.delenv("TOS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TOS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACCESS_KEY_ID", raising=False)

    io_config = IOConfig(s3=tos_config.to_s3_config())
    daft.set_planning_config(default_io_config=io_config)

    s3_dir = object_store_test_dir.replace("tos://", "s3://")
    remote_path = f"{s3_dir}/test_basic_io.txt"
    _test_basic_io(remote_path, str(tmpdir))


def _test_basic_io(remote_path: str, work_dir: str):
    # Prepare data
    raw_file = normalize_local_path(f"{work_dir}/raw_file")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    data = "hello world"
    with raw_file.open(mode="w+") as f:
        f.write(data)

    # 1. Failed to upload file since source file doesn't exist.
    try:
        upload_file(f"{work_dir}/{uuid.uuid4()}-raw_file", remote_path)
    except FileNotFoundError:
        assert True

    # 2. Failed to upload file since source file is a dir.
    try:
        source = f"{work_dir}/{uuid.uuid4()}"
        normalize_local_path(source).mkdir(parents=True, exist_ok=True)
        upload_file(source, remote_path)
    except FileNotFoundError:
        assert True

    # 3. Upload file
    upload_file(str(raw_file), remote_path)
    assert exists(remote_path)
    assert file_size(remote_path) == len(data)

    # 4. Upload exist file
    upload_file(str(raw_file), remote_path)
    assert file_size(remote_path) == len(data)

    # 5. Failed to upload file with non-overwrite mode since target file already exist
    try:
        upload_file(str(raw_file), remote_path, overwrite=False)
    except FileExistsError:
        assert True

    # 6. Download file
    local_file = f"{work_dir}/{Path(remote_path).name}"
    download_file(remote_path, local_file)
    assert normalize_local_path(local_file).stat().st_size == len(data)

    # 7. Download file again with overwrite=True
    download_file(remote_path, local_file)
    assert normalize_local_path(local_file).stat().st_size == len(data)

    # 8. Download file again with overwrite=False
    try:
        download_file(remote_path, local_file, overwrite=False)
    except FileExistsError:
        assert True

    # 9. Remove local file
    rm(local_file)
    assert not exists(local_file)


def test_io_cache_cache():
    client = LasIOFactory.get().get_client("/a/b/c")
    assert LasIOFactory.get().get_client("/a/b/c") == client

def test_http_io():
    client = LasIOFactory.get().get_client("http://a/b/c", headers={"Authorization": "Bearer 123456"}, max_retries=3, backoff=1)
    assert isinstance(client, HttpIO)
    assert client.headers == {"Authorization": "Bearer 123456"}
    assert client.max_retries == 3
    assert client.backoff == 1


def test_daft_io_config_consistency(object_store_test_dir):
    tos_config = TOSConfig.from_env()
    io_config = IOConfig(s3=tos_config.to_s3_config())
    daft.set_planning_config(default_io_config=io_config)
    s3_dir = object_store_test_dir.replace("tos://", "s3://")

    expected = daft.from_pydict({"a": [1, 2, 3, 4], "b": [2, 4, 3, 1]})

    parquet_path = f"{s3_dir}/test.parquet"
    expected.write_parquet(parquet_path, io_config=io_config)
    actual = daft.read_parquet(parquet_path, io_config=io_config)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected.to_pandas())

    csv_path = f"{s3_dir}/test.csv"
    expected.write_csv(csv_path, io_config=io_config)
    actual = daft.read_csv(csv_path, io_config=io_config)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected.to_pandas())

    lance_path = f"{s3_dir}/test.lance"
    expected.write_lance(lance_path, io_config=io_config, mode="overwrite")
    actual = daft.read_lance(lance_path, io_config=io_config)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected.to_pandas())
