from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft.daft import IOConfig
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig
from daft.las.io.tos import TOSConfig

formats = ["csv", "parquet", "lance"]

data = {"name": ["Bush", "Obama", "Trump"], "age": [79, 64, 79]}

dataframe = daft.from_pydict(data)


@pytest.mark.parametrize("format", formats)
def test_las_dataset_basic(format, uuid_short, object_store_test_dir, monkeypatch):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

    client = LasDatasetClient(LasDatasetConfig())

    dataset_name = "dataset_" + uuid_short
    dataset_nickname = dataset_name
    root_dir = f"{object_store_test_dir}/{dataset_name}"
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())

    assert client.dataset_exist(dataset_name) is False

    dataframe.write_las_dataset(
        name=dataset_name,
        root_dir=root_dir,
        io_config=io_config,
        format=format,
        nick_name=dataset_nickname,
        labels=["label1", "label2"],
        privacy="PRIVATE",
        description="This is test dataset",
    )
    assert client.dataset_exist(dataset_name) is True

    actual = daft.read_las_dataset(name=dataset_name, io_config=io_config)
    pd.testing.assert_frame_equal(dataframe.to_pandas(), actual.to_pandas())


def test_append_data(uuid_short, object_store_test_dir, monkeypatch):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

    client = LasDatasetClient(LasDatasetConfig())

    dataset_name = "dataset_" + uuid_short
    dataset_nickname = dataset_name
    root_dir = f"{object_store_test_dir}/{dataset_name}"
    io_config = IOConfig(s3=TOSConfig.from_env().to_s3_config())

    # 1. confirm the dataset does not exist
    assert client.dataset_exist(dataset_name) is False

    # 2. write dataset, but didn't provide root_dir
    with pytest.raises(ValueError, match=r"You must specify the 'root_dir'*"):
        dataframe.write_las_dataset(name=dataset_name)

    # 3. write the dataset and make sure it has been created
    dataframe.write_las_dataset(
        name=dataset_name,
        root_dir=root_dir,
        io_config=io_config,
        format="csv",
        nick_name=dataset_nickname,
        labels=["label1", "label2"],
        privacy="PRIVATE",
        description="This is test dataset",
    )
    assert client.dataset_exist(dataset_name) is True

    # 4. read the dataset and check the result
    actual = daft.read_las_dataset(name=dataset_name, io_config=io_config)
    pd.testing.assert_frame_equal(dataframe.to_pandas(), actual.to_pandas())

    # 5. write the existing dataset, but provide different root_dir
    with pytest.raises(ValueError, match="already exists, but the data path it records"):
        dataframe.write_las_dataset(name=dataset_name, root_dir="tos://another/non/exist/root_dir")

    # 6. write to the existing dataset
    dataframe.write_las_dataset(name=dataset_name, io_config=io_config)

    # 7. read the dataset and check the result
    actual = daft.read_las_dataset(name=dataset_name, io_config=io_config)
    expected = pd.concat([dataframe.to_pandas(), dataframe.to_pandas()])
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.to_pandas())
