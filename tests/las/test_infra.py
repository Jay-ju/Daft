# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import pytest

from daft.las.infra.ark import ArkConfig, get_ark_client


def test_ark_client():
    ark_client = None
    try:
        config = ArkConfig.from_env()
        ark_client = get_ark_client(config)
        assert not ark_client.is_closed()
    finally:
        if ark_client is not None:
            ark_client.close()


def test_missing_base_url(monkeypatch):
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url must be specified"):
        ArkConfig.from_env()


def test_missing_credential(monkeypatch):
    monkeypatch.delenv("ARK_ACCESS_KEY", raising=False)
    monkeypatch.delenv("ARK_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACCESS_KEY_ID", raising=False)

    with pytest.raises(ValueError, match="Cannot found credentials or credential provider"):
        get_ark_client(ArkConfig.from_env())
