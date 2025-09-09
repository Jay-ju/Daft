# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pytest

from daft.las.infra.ark import ArkConfig, get_ark_client
from daft.las.infra.open_api import OpenAPIClient
from daft.las.infra.tos_client import TosClient
from daft.las.utils import get_ak_sk

VERSION = "2024-06-30"
HOST = "open.volcengineapi.com"
PATH = "/"
SERVICES = ["TOS", "CONTENT_SECURITY", "VISUAL_SERVICE"]


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


@pytest.mark.parametrize("service", SERVICES)
def test_get_ak_sk(service, monkeypatch):
    test_las_service_access_key = "test_las_service_access_key"
    test_las_service_secret_key = "test_las_service_secret_key"
    test_service_access_key = "test_service_access_key"
    test_service_secret_key = "test_service_secret_key"
    test_service_access_key_id = "test_service_access_key_id"
    test_service_secret_access_key = "test_service_secret_access_key"
    test_access_key = "test_access_key"
    test_secret_key = "test_secret_key"
    test_access_key_id = "test_access_key_id"
    test_secret_access_key = "test_secret_access_key"

    monkeypatch.delenv(f"LAS_{service}_ACCESS_KEY", raising=False)
    monkeypatch.delenv(f"{service}_ACCESS_KEY", raising=False)
    monkeypatch.delenv(f"LAS_{service}_SECRET_KEY", raising=False)
    monkeypatch.delenv(f"{service}_SECRET_KEY", raising=False)
    monkeypatch.delenv(f"{service}_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv(f"{service}_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACCESS_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("SECRET_ACCESS_KEY", raising=False)

    ak, sk = get_ak_sk(service)
    assert ak is None
    assert sk is None

    monkeypatch.setenv(f"LAS_{service}_ACCESS_KEY", test_las_service_access_key)
    monkeypatch.setenv(f"{service}_ACCESS_KEY", test_service_access_key)
    monkeypatch.setenv(f"{service}_SECRET_KEY", test_service_secret_key)
    monkeypatch.setenv(f"LAS_{service}_SECRET_KEY", test_las_service_secret_key)
    monkeypatch.setenv(f"{service}_ACCESS_KEY_ID", test_service_access_key_id)
    monkeypatch.setenv(f"{service}_SECRET_ACCESS_KEY", test_service_secret_access_key)
    monkeypatch.setenv("ACCESS_KEY", test_access_key)
    monkeypatch.setenv("SECRET_KEY", test_secret_key)
    monkeypatch.setenv("ACCESS_KEY_ID", test_access_key_id)
    monkeypatch.setenv("SECRET_ACCESS_KEY", test_secret_access_key)

    ak, sk = get_ak_sk(service)
    assert ak == test_las_service_access_key
    assert sk == test_las_service_secret_key

    monkeypatch.delenv(f"LAS_{service}_ACCESS_KEY")
    monkeypatch.delenv(f"LAS_{service}_SECRET_KEY")
    ak, sk = get_ak_sk(service)
    assert ak == test_service_access_key
    assert sk == test_service_secret_key

    monkeypatch.delenv(f"{service}_ACCESS_KEY")
    monkeypatch.delenv(f"{service}_SECRET_KEY")
    ak, sk = get_ak_sk(service)
    assert ak == test_service_access_key_id
    assert sk == test_service_secret_access_key

    monkeypatch.delenv(f"{service}_ACCESS_KEY_ID")
    monkeypatch.delenv(f"{service}_SECRET_ACCESS_KEY")
    ak, sk = get_ak_sk(service)
    assert ak == test_access_key
    assert sk == test_secret_key

    monkeypatch.delenv("ACCESS_KEY")
    monkeypatch.delenv("SECRET_KEY")
    ak, sk = get_ak_sk(service)
    assert ak == test_access_key_id
    assert sk == test_secret_access_key

    monkeypatch.delenv("ACCESS_KEY_ID")
    monkeypatch.delenv("SECRET_ACCESS_KEY")
    ak, sk = get_ak_sk(service)
    assert ak is None
    assert sk is None


def test_openapi():
    access_key, secret_key = get_ak_sk("las")
    client = OpenAPIClient(service="las_ai_qa", region="cn-beijing", access_key=access_key, secret_key=secret_key)
    response = client.call_api(
        method="POST",
        params={},
        headers={
            "Content-Type": "application/json",
        },
        action="CheckDatasetExists",
        body={"DatasetName": "non_exist_ds"},
    )
    assert response.json()["Result"]["Exists"] is False


def test_tos_client():
    tos_client = TosClient()
    assert tos_client.tos_client is not None
    url = tos_client.pre_sign_url("tos://test_bucket/object_for_test")
    assert url is not None and "test_bucket" in url and url.startswith("https")

    instance2 = TosClient()
    assert id(tos_client) == id(instance2)
