# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

from daft.daft import IOConfig
from daft.las.infra.credentials import UrlCredentialsProvider
from daft.las.infra.open_api import OpenAPIClient
from daft.las.io import TOSConfig
from daft.las.utils import (
    get_ak_sk,
    get_credentials_provider_url,
    get_region,
    get_session_token,
    is_static_credential,
    not_blank,
)

logger = logging.getLogger(__name__)


class LasDatasetFormat(Enum):
    CSV = 1
    JSON = 2
    PARQUET = 3
    LANCE = 4
    LANCE_TABLE = 5
    ICEBERG = 6
    IMAGE_FOLDER = 7
    VIDEO_FOLDER = 8
    AUDIO_FOLDER = 9
    TEXT = 10


las_dataset_privacy = {
    "public": "Public",
    "private": "Private",
}


las_dataset_format = {
    "csv": "CSV",
    "json": "JSONL",
    "jsonl": "JSONL",
    "parquet": "Parquet",
    "lance": "Lance",
    "iceberg": "Iceberg",
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
    "text": "Text",
    "webdataset": "WebDataset",
}


las_dataset_storage = {"tos": "TOS", "vepfs": "vePFS"}


@dataclass
class LasDatasetInfo:
    name: str
    format: str | None = None
    nick_name: str | None = None
    storage: str = "TOS"
    data_path: str | None = None
    tags: list[str] | None = None
    privacy: str = "Private"
    description: str | None = None
    table: str | None = None


class LasDatasetConfig:
    """Configs for the las client."""

    def __init__(
        self,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        credentials_provider_url: str | None = None,
    ):
        access_key_env, secret_key_env = get_ak_sk("las")

        self.region = region if not_blank(region) else get_region("las")
        self.access_key = access_key if not_blank(access_key) else access_key_env
        self.secret_key = secret_key if not_blank(secret_key) else secret_key_env
        self.session_token = session_token if not_blank(session_token) else get_session_token("las")
        self.credentials_provider_url = (
            credentials_provider_url if not_blank(credentials_provider_url) else get_credentials_provider_url("las")
        )

        if not_blank(self.credentials_provider_url):
            credentials_provider = UrlCredentialsProvider(self.credentials_provider_url)  # type: ignore[arg-type]
            self.session_token = credentials_provider.get_credentials().session_token
            self.access_key = credentials_provider.get_credentials().access_key
            self.secret_key = credentials_provider.get_credentials().secret_key

        if not not_blank(self.region):
            raise ValueError("'region' is not configured")
        if not is_static_credential(self.access_key, self.secret_key) and not not_blank(self.credentials_provider_url):
            raise ValueError("Cannot found credentials or credential provider.")

    @staticmethod
    def from_io_config(config: IOConfig | None) -> LasDatasetConfig:
        config = config if config is not None else IOConfig(s3=TOSConfig.from_env().to_s3_config())
        return LasDatasetConfig(
            region=config.s3.region_name,
            access_key=config.s3.key_id,
            secret_key=config.s3.access_key,
            session_token=config.s3.session_token,
        )


class LasDatasetClient:
    """Client that /create/get las datasets."""

    def __init__(self, config: LasDatasetConfig):
        service = os.environ.get("LAS_SERVICE_NAME", "las")
        self.api_client = OpenAPIClient(
            service=service,
            region=config.region,  # type: ignore[arg-type]
            access_key=config.access_key,  # type: ignore[arg-type]
            secret_key=config.secret_key,  # type: ignore[arg-type]
            session_token=config.session_token,
        )

    def create_dataset(self, dataset: LasDatasetInfo) -> None:
        body = {
            "DatasetName": dataset.name,
            "Nickname": dataset.nick_name,
            "Format": dataset.format,
            "Storage": dataset.storage,
            "DataPath": dataset.data_path,
            "Tags": dataset.tags,
            "Privacy": dataset.privacy,
            "Description": dataset.description,
        }
        self.api_client.call_api(
            method="POST",
            params={},
            headers={},
            action="CreateDataset",
            body=body,
        )

    def dataset_exist(self, name: str) -> bool:
        body = {"DatasetName": name}
        response = self.api_client.call_api(
            method="POST",
            params={},
            headers={},
            action="CheckDatasetExists",
            body=body,
        )
        return bool(response.json()["Result"]["Exists"])

    def get_dataset(self, name: str) -> LasDatasetInfo:
        body = {"DatasetName": name}
        response = self.api_client.call_api(
            method="POST",
            params={},
            headers={},
            action="GetDataset",
            body=body,
        )
        result = response.json()["Result"]

        table: str | None = None
        if result.get("Catalog", None) is not None:
            catalog = result["Catalog"]
            catalog_name = catalog["CatalogName"]
            schema_name = catalog["SchemaName"]
            table_name = catalog["TableName"]
            table = f"{catalog_name}.{schema_name}.{table_name}"

        res = LasDatasetInfo(
            name=result.get("DatasetName", None),
            nick_name=result.get("Nickname", None),
            tags=result.get("Labels", None),
            privacy=result.get("Privacy", None),
            description=result.get("Description", None),
            format=result.get("Format", None),
            storage=result.get("Storage", None),
            data_path=result.get("DataPath", None),
            table=table,
        )

        dataset_local_path = os.environ.get(f"LAS_CONTROLLED_DATASET_MOUNT_PATH_{name}")
        if dataset_local_path is not None:
            res.data_path = dataset_local_path

        return res

    def delete_dataset(self, name: str, delete_data: bool = False) -> None:
        body = {"DatasetName": name, "DeleteData": delete_data}
        self.api_client.call_api(
            method="POST",
            params={},
            headers={},
            action="DeleteDataset",
            body=body,
        )

    def refresh_dataset(self, name: str) -> None:
        body = {"DatasetName": name}
        self.api_client.call_api(
            method="POST",
            params={},
            headers={},
            action="RefreshDatasetMetadata",
            body=body,
        )
