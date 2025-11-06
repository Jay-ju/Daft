# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import daft
from daft.daft import IOConfig
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig, LasDatasetInfo
from daft.las.io.tos import TOSConfig


def test_las_dataset_client(uuid_short, monkeypatch, object_store_test_dir):
    config: LasDatasetConfig = LasDatasetConfig()
    client: LasDatasetClient = LasDatasetClient(config)

    data_path = f"{object_store_test_dir}/las_dataset"
    uri = data_path.replace("tos://", "s3://")
    dataframe = daft.from_pydict({"name": ["bush", "obama", "trump"], "age": [1, 2, 3]})
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())

    dataframe.write_csv(root_dir=uri, io_config=io_config)

    ds_name = "test_dataset_" + uuid_short
    expected = LasDatasetInfo(
        name=ds_name,
        format="CSV",
        nick_name=ds_name,
        data_path=data_path,
        description="A dataset for test",
    )

    # Set up local mount path environment variable
    local_mount_path = "/local/path/to/dataset"
    monkeypatch.setenv(f"LAS_CONTROLLED_DATASET_MOUNT_PATH_{ds_name}", local_mount_path)

    assert client.dataset_exist(name=ds_name) is False

    client.create_dataset(dataset=expected)
    actual = client.get_dataset(ds_name)

    assert actual.name == expected.name
    assert actual.format == expected.format
    assert actual.nick_name == expected.nick_name
    assert actual.description == expected.description
    assert actual.data_path == "/local/path/to/dataset"

    # test delete dataset
    client.delete_dataset(ds_name)
    assert client.dataset_exist(name=ds_name) is False
