from __future__ import annotations

from typing import TYPE_CHECKING, Any

from daft.api_annotations import PublicAPI
from daft.las.infra.las_dataset import LasDatasetClient, LasDatasetConfig

from ._csv import read_csv
from ._parquet import read_parquet
from .iceberg._iceberg import read_iceberg
from .lance._lance import read_lance

if TYPE_CHECKING:
    from daft import DataFrame
    from daft.daft import IOConfig


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
    config = LasDatasetConfig.from_io_config(io_config)
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
