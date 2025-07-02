# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
from typing import Callable

from volcengine.visual.VisualService import VisualService

from daft.las.infra.credentials import Credentials, UrlCredentialsProvider
from daft.las.utils import get_ak_sk, is_static_credential, not_blank


class VisualServiceConfig:
    """Config for the volcengine visual service."""

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        credential_provider: Callable[[], Credentials] | None = None,
        credential_provider_url: str | None = None,
        host: str = "visual.volcengineapi.com",
        scheme: str = "http",
        connect_timeout: int = 30,
        socket_timeout: int = 30,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.credential_provider = credential_provider
        self.credential_provider_url = credential_provider_url
        self.host = host
        self.scheme = scheme
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout

        self._check_credential_info()

    def _check_credential_info(self) -> None:
        if (
            not is_static_credential(self.access_key, self.secret_key)
            and not not_blank(self.credential_provider_url)
            and not self.credential_provider
        ):
            raise ValueError("Cannot found credentials or credential provider.")

    @staticmethod
    def from_env() -> VisualServiceConfig:
        access_key, secret_key = get_ak_sk("visualservice")
        return VisualServiceConfig(
            access_key=access_key,
            secret_key=secret_key,
            session_token=os.getenv("VISUALSERVICE_SESSION_TOKEN"),
            credential_provider_url=os.getenv("VISUALSERVICE_CREDENTIAL_PROVIDER_URL"),
            host=os.getenv("VISUALSERVICE_HOST", "visual.volcengineapi.com"),
            scheme=os.getenv("VISUALSERVICE_SCHEME", "http"),
            connect_timeout=int(os.getenv("VISUALSERVICE_CONNECT_TIMEOUT", 30)),
            socket_timeout=int(os.getenv("VISUALSERVICE_SOCKET_TIMEOUT", 30)),
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
    elif config.credential_provider:
        credentials = config.credential_provider()
    elif config.credential_provider_url:
        credentials = UrlCredentialsProvider(config.credential_provider_url).get_credentials()
    else:
        raise ValueError("Missing credentials.")
    if credentials is not None:
        ak = credentials.get_ak()
        sk = credentials.get_sk()
        token = credentials.get_session_token()

    visual_service.set_ak(ak)
    visual_service.set_sk(sk)
    if not_blank(token):
        visual_service.set_session_token(token)
    visual_service.set_host(config.host)
    visual_service.set_scheme(config.scheme)
    visual_service.set_connection_timeout(config.connect_timeout)
    visual_service.set_socket_timeout(config.socket_timeout)

    return visual_service
