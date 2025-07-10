from __future__ import annotations

from typing import TYPE_CHECKING, Any

from daft.api_annotations import PublicAPI
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig

from ._csv import read_csv
from ._iceberg import read_iceberg
from ._lance import read_lance
from ._parquet import read_parquet

if TYPE_CHECKING:
    from daft import DataFrame
    from daft.daft import IOConfig, S3Config


@PublicAPI
def read_las_dataset(name: str, io_config: IOConfig | None, **kwargs: Any) -> DataFrame:
    """Create a DataFrame from a las dataset.

    Args:
        name: Name of the dataset.
        io_config: IOConfig for reading the storage
        **kwargs: Additional format-specific arguments

    Returns:
        DataFrame: a DataFrame with the schema converted from the dataset.
    """
    if io_config is not None and io_config.s3 is not None:
        s3_config: S3Config = io_config.s3
        region = s3_config.region_name
        access_key = s3_config.key_id
        secret_key = s3_config.access_key
        session_token = s3_config.session_token
        config = LasDatasetConfig(
            region=region, access_key=access_key, secret_key=secret_key, session_token=session_token
        )
    else:
        config = LasDatasetConfig.from_env()
    client = LasDatasetClient(config)

    if not (client.dataset_exist(name=name)):
        raise ValueError(f"Dataset {name} not exist!")

    dataset_info = client.get_dataset(name=name)

    format = dataset_info.format
    assert dataset_info.data_path is not None

    path = dataset_info.data_path
    if path.startswith("tos://"):
        path = path.replace("tos://", "s3://")

    if format.name == "CSV":
        kwargs.pop("path", None)
        return read_csv(path=path, io_config=io_config, **kwargs)
    elif format.name == "PARQUET":
        kwargs.pop("path", None)
        return read_parquet(path=path, io_config=io_config, **kwargs)
    elif format.name == "LANCE":
        kwargs.pop("url", None)
        return read_lance(url=path, io_config=io_config, **kwargs)
    elif format.name == "ICEBERG":
        kwargs.pop("table", None)
        table = dataset_info.table
        assert table is not None
        return read_iceberg(table=table, io_config=io_config, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {format.name}")
