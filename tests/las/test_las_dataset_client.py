# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pytest
from httpx import HTTPStatusError

import daft
from daft.daft import IOConfig
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig, LasDatasetInfo
from daft.las.io.tos import TOSConfig


def test_las_dataset_client(uuid_short, monkeypatch, object_store_test_dir):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

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

    assert client.dataset_exist(name=ds_name) is False

    client.create_dataset(dataset=expected)
    actual = client.get_dataset(ds_name)

    assert actual == expected

    # test delete dataset
    client.delete_dataset(ds_name)
    assert client.dataset_exist(name=ds_name) is False

    # test non-exist dataset, it would give an exception
    with pytest.raises(HTTPStatusError, match=r".*Dataset.* is not exist"):
        client.delete_dataset("Non-existent-dataset")
