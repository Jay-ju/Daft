# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
from typing import Any


def not_blank(string: str | None) -> bool:
    return string is not None and not len(string.strip()) == 0


def is_static_credential(access_key: str | None, secret_key: str | None) -> bool:
    return not_blank(access_key) and not_blank(secret_key)


def get_ak_sk(service: str) -> tuple[str | None, str | None]:
    """Get the access key and secret key from env.

    ACCESS_KEY/SECRET_KEY are short form of ACCESS_KEY_ID/SECRET_ACCESS_KEY.

    Priority (tos as an example):
    LAS_TOS_ACCESS_KEY > TOS_ACCESS_KEY > TOS_ACCESS_KEY_ID > ACCESS_KEY > ACCESS_KEY_ID
    LAS_TOS_SECRET_KEY > TOS_SECRET_KEY > TOS_SECRET_ACCESS_KEY > SECRET_KEY > SECRET_ACCESS_KEY
    """
    prefix = service.upper()
    service_ak_short = prefix + "_ACCESS_KEY"
    service_ak_long = prefix + "_ACCESS_KEY_ID"
    service_sk_short = prefix + "_SECRET_KEY"
    service_sk_long = prefix + "_SECRET_ACCESS_KEY"

    access_key = (
        os.getenv(f"LAS_{service_ak_short}")
        or os.getenv(service_ak_short)
        or os.getenv(service_ak_long)
        or os.getenv("ACCESS_KEY")
        or os.getenv("ACCESS_KEY_ID")
    )
    secret_key = (
        os.getenv(f"LAS_{service_sk_short}")
        or os.getenv(service_sk_short)
        or os.getenv(service_sk_long)
        or os.getenv("SECRET_KEY")
        or os.getenv("SECRET_ACCESS_KEY")
    )

    return access_key, secret_key


def get_session_token(service: str) -> str | None:
    return os.getenv(f"{service.upper()}_SESSION_TOKEN") or os.getenv("SESSION_TOKEN")


def get_region(service: str) -> str | None:
    return os.getenv(f"{service.upper()}_REGION") or os.getenv("REGION")


def get_env(env: str, default: Any | None = None) -> Any | None:
    return os.getenv(env) or default


def get_credentials_provider_url(service: str) -> str | None:
    return os.getenv(f"{service.upper()}_CREDENTIALS_PROVIDER_URL") or os.getenv("CREDENTIALS_PROVIDER_URL")
