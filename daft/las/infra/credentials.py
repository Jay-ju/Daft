# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
from httpx import Response
from tenacity import retry, stop_after_attempt, wait_exponential

from daft.las.utils import not_blank

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class Credentials:
    """Class that holds the user's credentials that includes ak/sk and token."""

    access_key: str
    secret_key: str
    session_token: str
    expire_time: datetime


class CredentialsProvider:
    def get_credentials(self) -> Credentials | None:
        raise NotImplementedError


class UrlCredentialsProvider(CredentialsProvider):
    """The class provides the credentials from an url."""

    credential_url: str
    credentials: Credentials
    expire_duration: timedelta = timedelta(minutes=10)

    def __init__(self, credential_url: str):
        if not_blank(credential_url):
            raise ValueError("The credential_url param must not be empty.")
        self._lock = threading.Lock()
        self.credential_url = credential_url

    def get_credentials(self) -> Credentials:
        res = self._try_get_credentials()
        if res is not None:
            return res
        with self._lock:
            res = self._try_get_credentials()
            if res is not None:
                return res

            resp = self._exec_call_with_retry()
            res_body = resp.json()
            self.credentials = Credentials(
                access_key=res_body.get("AccessKeyId"),
                secret_key=res_body.get("SecretAccessKey"),
                session_token=res_body.get("SessionToken"),
                expire_time=datetime.strptime(res_body.get("ExpiredTime"), DATE_FORMAT),
            )
            return self.credentials

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))  # type: ignore[misc]
    def _exec_call_with_retry(self) -> Response:
        return httpx.get(self.credential_url, timeout=30)

    def get_account_info(self) -> dict[str, Any]:
        credentials = self.get_credentials()
        return {
            "AccessKeyId": credentials.access_key,
            "SecretAccessKey": credentials.secret_key,
            "SessionToken": credentials.session_token,
            "ExpiredTime": credentials.expire_time,
        }

    def _try_get_credentials(self) -> Credentials | None:
        if self.credentials is None:
            return None
        expire_threshold: datetime = self.credentials.expire_time - self.expire_duration
        return None if (datetime.now().timestamp() > expire_threshold.timestamp()) else self.credentials
