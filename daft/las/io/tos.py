# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from tos.credential import Credentials
from tosfs import TosFileSystem
from tosfs.certification import NoLockUrlCredentialsProvider
from tosfs.exceptions import TosfsError

import daft.daft
from daft.daft import S3Config, S3Credentials
from daft.las.io.factory import LasIO, register_io_client
from daft.las.io.utils import generate_temp_file, normalize_local_path
from daft.las.utils import (
    get_ak_sk,
    get_credentials_provider_url,
    get_env,
    get_region,
    get_session_token,
    is_static_credential,
    not_blank,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

logger = logging.getLogger(__name__)


class TosCredentials:
    access_key: str
    secret_key: str
    session_token: str | None
    expiry: datetime | None

    def __init__(
        self, access_key: str, secret_key: str, session_token: str | None = None, expiry: datetime | None = None
    ):
        if not access_key or access_key.isspace():
            raise ValueError("access_key is cannot be empty")

        if not secret_key or secret_key.isspace():
            raise ValueError("secret_key is cannot be empty")

        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.expiry = expiry

    @staticmethod
    def from_s3_credentials(cred: S3Credentials) -> TosCredentials:
        return TosCredentials(
            access_key=cred.key_id,
            secret_key=cred.access_key,
            session_token=cred.session_token,
            expiry=cred.expiry,
        )

    def to_s3_credentials(self) -> S3Credentials:
        return S3Credentials(
            key_id=self.access_key,
            access_key=self.secret_key,
            session_token=self.session_token,
            expiry=self.expiry,
        )

    def to_tosfs_credentials(self) -> Credentials:
        return Credentials(self.access_key, self.secret_key, self.session_token)


class TOSConfig:
    """I/O configuration for accessing tos object store."""

    region: str | None
    endpoint: str | None
    access_key: str | None
    secret_key: str | None
    credentials_provider: Callable[[], TosCredentials] | None
    credentials_provider_url: str | None
    max_retry_num: int
    max_connections: int
    connection_timeout: int
    socket_timeout: int

    def __init__(
        self,
        endpoint: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        credentials_provider: Callable[[], TosCredentials] | None = None,
        credentials_provider_url: str | None = None,
        max_retry_num: int = 20,
        max_connections: int = 1024,
        connection_timeout: int = 180,
        socket_timeout: int = 180,
    ):
        self.endpoint, self.region = self._parse_endpoint(endpoint, region)
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.credentials_provider = credentials_provider
        self.credentials_provider_url = credentials_provider_url

        self._check_credential_info()

        self.max_retry_num = max_retry_num
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.socket_timeout = socket_timeout

    def _check_credential_info(self) -> None:
        if (
            not is_static_credential(self.access_key, self.secret_key)
            and not not_blank(self.credentials_provider_url)
            and not self.credentials_provider
        ):
            raise ValueError("Cannot found credentials or credential provider.")

    def _parse_endpoint(self, endpoint: str | None, region: str | None) -> tuple[str, str]:
        if endpoint is None:
            raise ValueError("endpoint must be specified")

        if region is None:
            region = self._extract_region_regex(endpoint)
            if region is None:
                raise ValueError("region must be specified")

        return endpoint, region

    @classmethod
    def _extract_region_regex(cls, endpoint: str) -> str | None:
        match = re.search(r"tos-([a-z]+[-a-z0-9]*)\.(?:iv|v)olces\.com", endpoint)
        return match.group(1) if match else None

    @staticmethod
    def from_env() -> TOSConfig:
        load_dotenv()

        access_key, secret_key = get_ak_sk("tos")
        return TOSConfig(
            endpoint=os.getenv("TOS_ENDPOINT"),
            region=get_region("tos"),
            access_key=access_key,
            secret_key=secret_key,
            session_token=get_session_token("tos"),
            credentials_provider_url=get_credentials_provider_url("tos"),
            max_retry_num=int(get_env("TOS_MAX_RETRY_NUM", 20)),  # type: ignore[arg-type]
            max_connections=int(get_env("TOS_MAX_CONNECTIONS", 1024)),  # type: ignore[arg-type]
            connection_timeout=int(get_env("TOS_CONNECTION_TIMEOUT", 180)),  # type: ignore[arg-type]
            socket_timeout=int(get_env("TOS_SOCKET_TIMEOUT", 180)),  # type: ignore[arg-type]
        )

    @staticmethod
    def from_s3_config(s3_config: S3Config) -> TOSConfig:
        if not s3_config.endpoint_url or "tos-s3" not in s3_config.endpoint_url:
            raise ValueError(
                f"s3 endpoint '{s3_config.endpoint_url}' is not standard endpoint for tos, "
                f"cannot convert it. Please using other approach to create TosConfig."
            )

        config = TOSConfig(
            endpoint=s3_config.endpoint_url.replace("tos-s3", "tos") if s3_config.endpoint_url else None,
            region=s3_config.region_name,
            access_key=s3_config.key_id,
            secret_key=s3_config.access_key,
            session_token=s3_config.session_token,
            connection_timeout=int(s3_config.connect_timeout_ms / 1000),
            socket_timeout=int(s3_config.read_timeout_ms / 1000),
            max_retry_num=s3_config.num_tries,
        )

        credentials_provider = s3_config.credentials_provider
        if credentials_provider:
            config.credentials_provider = lambda: TosCredentials.from_s3_credentials(credentials_provider())

        return config

    def to_s3_config(self) -> S3Config:
        endpoint_url = self.endpoint.replace("tos-", "tos-s3-") if self.endpoint else None

        def provider() -> S3Credentials:
            assert self.credentials_provider is not None
            return self.credentials_provider().to_s3_credentials()

        return S3Config(
            endpoint_url=endpoint_url,
            region_name=self.region,
            key_id=self.access_key,
            access_key=self.secret_key,
            session_token=self.session_token,
            credentials_provider=provider if self.credentials_provider else None,
            force_virtual_addressing=True,
            connect_timeout_ms=self.connection_timeout * 1000,
            read_timeout_ms=self.socket_timeout * 1000,
            num_tries=self.max_retry_num,
        )

    def virtual_host_endpoint(self, bucket: str) -> str:
        if bucket is None:
            raise ValueError("bucket must be specified")

        if self.endpoint is None:
            raise ValueError("endpoint is not found.")

        if self.endpoint.startswith("https://"):
            endpoint = self.endpoint[8:]
            scheme = "https"
        elif self.endpoint.startswith("https://"):
            endpoint = self.endpoint[7:]
            scheme = "http"
        else:
            endpoint = self.endpoint
            scheme = "https"

        return f"{scheme}://{bucket}.{endpoint}"


@register_io_client(scheme="tos")
class TosIO(LasIO):
    """The TosIO provides uploading and downloading file to/from tos object store."""

    def __init__(self, config: TOSConfig | None = None) -> None:
        if config is None:
            try:
                config = TOSConfig.from_env()
                logger.info("Fetch config from env.")
            except ValueError:
                ctx = daft.daft.get_context()
                s3_config = ctx._daft_planning_config.default_io_config.s3
                if s3_config:
                    config = TOSConfig.from_s3_config(s3_config)
                    logger.info("Fetch config from s3 config of daft context.")

        self._fs = self._create_tosfs(config)

    @staticmethod
    def _create_tosfs(config: TOSConfig | None) -> TosFileSystem:
        if config is None:
            raise ValueError("TOS config is not provided")

        if is_static_credential(config.access_key, config.secret_key):
            return TosFileSystem(
                endpoint=config.endpoint,
                region=config.region,
                key=config.access_key,
                secret=config.secret_key,
                session_token=config.session_token,
            )

        if config.credentials_provider_url:
            if config.credentials_provider_url.isspace():
                raise ValueError("credentials_provider_url cannot be empty")

            return TosFileSystem(
                endpoint=config.endpoint,
                region=config.region,
                credentials_provider=NoLockUrlCredentialsProvider(config.credentials_provider_url),
            )

        if config.credentials_provider:
            return TosFileSystem(
                endpoint=config.endpoint,
                region=config.region,
                credentials_provider=lambda: config.credentials_provider().to_tosfs_credentials(),
            )

        raise ValueError("Cannot found credentials or credential provider.")

    @classmethod
    def scheme(cls) -> str:
        return "tos"

    def mkdirs(self, path: str) -> None:
        # Note: no need to check whether a file existed, because general bucket hasn't the concept
        # of dir or file, and directory bucket doesn't support overwrite a file via dir.
        self._fs.makedirs(path, exist_ok=True)

    def rm(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._fs.rm(path, recursive=True)

    def exists(self, path: str) -> bool:
        return self._fs.exists(path)

    def file_size(self, path: str) -> int:
        return self._fs.info(path)["size"]

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        local_file = normalize_local_path(local)
        if local_file.exists():
            if not overwrite:
                raise FileExistsError(f"The local file: {local} already exists.")
        else:
            parent = local_file.parent
            parent.mkdir(parents=True, exist_ok=True)

        temp_file = Path(generate_temp_file(str(local_file)))
        try:
            self._fs.get_file(remote, str(temp_file))

            # delete the existing file or dir after downloading file instead of deleting at
            # first to avoid loss the existing data as much as possible.
            if local_file.exists():
                if local_file.is_dir():
                    shutil.rmtree(local)
                else:
                    local_file.unlink()

            temp_file.rename(local_file)
        except FileNotFoundError:
            logger.exception("The source file: %s is not found.", local)
        except TosfsError:
            logger.exception("Failed to download file from %s to %s.", remote, local)
        finally:
            temp_file.unlink(missing_ok=True)

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        local_file = normalize_local_path(local)
        if not local_file.exists():
            raise FileNotFoundError(f"The source file: {local_file} is not found.")

        if local_file.is_dir():
            raise FileNotFoundError(f"The source file: {local_file} is not a file.")

        # Note: The most general object store bucket supports overwrite the existing
        # object, but the directory bucket doesn't allow overwrote dir via a file.
        # We check the existence here only when overwrite is False instead of checking
        # existence all the time to reduce head QPS.
        if not overwrite and self._fs.exists(remote):
            raise FileExistsError(f"The destination file: {remote} already exists.")

        try:
            super(self._fs.__class__, self._fs).put_file(local, remote)
        except (ValueError, TosfsError):
            if self._fs.isdir(remote):
                # The culprit might be the directory bucket doesn't allow overwrote
                # a dir via a file.
                self._fs.rm(remote, recursive=True)
                self._fs.put_file(local, remote)

            logger.exception("Failed to upload file from %s to %s.", local, remote)
