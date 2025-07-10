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
def test_las_dataset_csv(format, uuid_short, object_store_test_dir, monkeypatch):
    monkeypatch.setenv("LAS_SERVICE_NAME", "las_ai_qa")

    client = LasDatasetClient(LasDatasetConfig.from_env())

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
