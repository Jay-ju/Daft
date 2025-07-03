# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

from volcengine.visual.VisualService import VisualService

from daft.las.infra.credentials import Credentials, CredentialsProvider, UrlCredentialsProvider
from daft.las.utils import get_ak_sk, is_static_credential, not_blank


class VisualServiceConfig:
    """Config for the volcengine visual service."""

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        credentials_provider: CredentialsProvider | None = None,
        credentials_provider_url: str | None = None,
        host: str = "visual.volcengineapi.com",
        scheme: str = "http",
        connect_timeout: int = 30,
        socket_timeout: int = 30,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.credentials_provider = credentials_provider
        self.credentials_provider_url = credentials_provider_url
        self.host = host
        self.scheme = scheme
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout

        self._check_credential_info()

    def _check_credential_info(self) -> None:
        if (
            not is_static_credential(self.access_key, self.secret_key)
            and not not_blank(self.credentials_provider_url)
            and not self.credentials_provider
        ):
            raise ValueError("Cannot found credentials or credential provider.")

    @staticmethod
    def from_env() -> VisualServiceConfig:
        access_key, secret_key = get_ak_sk("visual_service")
        return VisualServiceConfig(
            access_key=access_key,
            secret_key=secret_key,
            session_token=os.getenv("VISUAL_SERVICE_SESSION_TOKEN"),
            credentials_provider_url=os.getenv("VISUAL_SERVICE_CREDENTIAL_PROVIDER_URL"),
            host=os.getenv("VISUAL_SERVICE_HOST", "visual.volcengineapi.com"),
            scheme=os.getenv("VISUAL_SERVICE_SCHEME", "http"),
            connect_timeout=int(os.getenv("VISUAL_SERVICE_CONNECT_TIMEOUT", 30)),
            socket_timeout=int(os.getenv("VISUAL_SERVICE_SOCKET_TIMEOUT", 30)),
        )


def get_visual_service(config: VisualServiceConfig) -> VisualService:
    """Get visual service by the default settings."""
    visual_service = VisualService()

    ak: str | None = None
    sk: str | None = None
    token: str | None = None
    credentials: Credentials | None = None

    if config.access_key and config.secret_key:
        ak = config.access_key
        sk = config.secret_key
    elif config.credentials_provider:
        credentials = config.credentials_provider()
    elif config.credentials_provider_url:
        credentials = UrlCredentialsProvider(config.credentials_provider_url).get_credentials()
    else:
        raise ValueError("Missing credentials.")
    if credentials is not None:
        ak = credentials.access_key
        sk = credentials.secret_key
        token = credentials.session_token

    visual_service.set_ak(ak)
    visual_service.set_sk(sk)
    if not_blank(token):
        visual_service.set_session_token(token)
    visual_service.set_host(config.host)
    visual_service.set_scheme(config.scheme)
    visual_service.set_connection_timeout(config.connect_timeout)
    visual_service.set_socket_timeout(config.socket_timeout)

    return visual_service
