# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from httpx import Auth, Request, Response

from daft.las.infra.credentials import CredentialsProvider, StaticCredentialsProvider
from daft.las.infra.http.singer import Singer
from daft.las.utils import not_blank

if TYPE_CHECKING:
    from collections.abc import Generator


class FunctionAuth(Auth):  # type: ignore[misc]
    def __init__(self, func: Callable[[Request], Request]) -> None:
        self._func = func

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        yield self._func(request)


class AuthProvider:
    def __call__(self, req: Request) -> Request:
        return self.authentication(req)

    def authentication(self, req: Request) -> Request:
        raise NotImplementedError()


class AnonymousProvider(AuthProvider):
    def authentication(self, req: Request) -> Request:
        return req


class ApiKeyAuthProvider(AuthProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def authentication(self, req: Request) -> Request:
        req.headers.update({"Authorization": f"ApiKey {self.api_key}"})
        return req


class VolcOpenApiAuthProvider(AuthProvider):
    def __init__(
        self,
        service: str,
        region: str,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        credential_provider: CredentialsProvider | None = None,
        signing_algorithm: str = "HMAC-SHA256",
        date_key: str = "x-date",
        content_hash_key: str = "x-content-sha256",
    ):
        self.logger = logging.getLogger("VolcOpenApiAuthProvider")
        if not_blank(access_key) and not_blank(secret_key):
            if credential_provider:
                self.logger.warning(
                    "Both credential_provider and access_key/secret_key are provided, credential_provider will be ignored."
                )

            credential_provider = StaticCredentialsProvider(
                access_key=access_key,  # type: ignore[arg-type]
                secret_key=secret_key,  # type: ignore[arg-type]
                session_token=session_token,
            )

        if credential_provider is None:
            raise ValueError("Either credential_provider or access_key/secret_key must be provided.")

        self.singer = Singer(service, region, credential_provider, signing_algorithm, date_key, content_hash_key)

    def authentication(self, req: Request) -> Request:
        req = self.singer.sign_request(req)

        self.logger.debug("The authorization is: %s", req.headers.get("Authorization"))
        return req
