# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

from daft.las.infra.credentials import UrlCredentialsProvider
from daft.las.infra.open_api import OpenAPIClient
from daft.las.utils import get_ak_sk, get_credentials_provider_url, get_region, get_session_token, not_blank

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


class Privacy(Enum):
    PUBLIC = 1
    PRIVATE = 2


class Storage(Enum):
    TOS = 1
    VEPFS = 2


@dataclass
class LasDatasetInfo:
    name: str
    format: LasDatasetFormat
    nick_name: str | None = None
    storage: Storage = Storage.TOS
    data_path: str | None = None
    labels: list[str] | None = None
    privacy: Privacy = Privacy.PUBLIC
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
        assert region is not None
        self.region = region

        if not_blank(credentials_provider_url):
            credentials_provider = UrlCredentialsProvider(credentials_provider_url)  # type: ignore[arg-type]
            self.session_token = credentials_provider.get_credentials().session_token
            self.access_key = credentials_provider.get_credentials().access_key
            self.secret_key = credentials_provider.get_credentials().secret_key
        else:
            assert access_key is not None
            assert secret_key is not None
            self.access_key = access_key
            self.secret_key = secret_key
            self.session_token = session_token

    @staticmethod
    def from_env() -> LasDatasetConfig:
        load_dotenv()

        region = get_region("las")
        access_key, secret_key = get_ak_sk("las")
        return LasDatasetConfig(
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            session_token=get_session_token("las"),
            credentials_provider_url=get_credentials_provider_url("las"),
        )


class LasDatasetClient:
    """Client that /create/get las datasets."""

    def __init__(self, config: LasDatasetConfig):
        # In debug mode, set LAS_SERVICE_NAME to las_ai_qa
        service = os.environ.get("LAS_SERVICE_NAME", "las")

        self.api_client = OpenAPIClient(
            service=service,
            region=config.region,
            access_key=config.access_key,
            secret_key=config.secret_key,
            session_token=config.session_token,
        )

    def create_dataset(self, dataset: LasDatasetInfo) -> None:
        body = {
            "DatasetName": dataset.name,
            "Nickname": dataset.nick_name,
            "Format": dataset.format.name,
            "Storage": dataset.storage.name,
            "DataPath": dataset.data_path,
            "Labels": dataset.labels,
            "Privacy": dataset.privacy.name,
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
            action="ExistsDataset",
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
        if result["Catalog"] is not None:
            catalog = result["Catalog"]
            catalog_name = catalog["CatalogName"]
            schema_name = catalog["SchemaName"]
            table_name = catalog_name["TableName"]
            table = f"{catalog_name}.{schema_name}.{table_name}"

        return LasDatasetInfo(
            name=result["DatasetName"],
            nick_name=result["Nickname"],
            labels=result["Labels"],
            privacy=Privacy[result["Privacy"]],
            description=result["Description"],
            format=LasDatasetFormat[result["Format"].upper()],
            storage=Storage[result["Storage"].upper()],
            data_path=result["DataPath"],
            table=table,
        )
