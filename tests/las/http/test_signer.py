from __future__ import annotations

import httpx

from daft.las.infra.credentials import StaticCredentialsProvider
from daft.las.infra.http.singer import Singer


def test_singer():
    signer = Singer(
        service="billing",
        region="cn-beijing",
        credential_provider=StaticCredentialsProvider(access_key="test_ak", secret_key="test_sk"),
        date="20250329",
    )

    client = httpx.Client(base_url="https://billing.volcengineapi.com")
    headers = {"x-date": "20250329T180937Z"}
    req = client.build_request(
        method="GET",
        url="/",
        params={"Action": "QueryBalanceAcct", "Version": "2022-01-01"},
        headers=headers,
    )

    req = signer.sign_request(req)
    signature = req.headers["Authorization"]
    assert (
        signature
        == "HMAC-SHA256 Credential=test_ak/20250329/cn-beijing/billing/request, SignedHeaders=host;x-content-sha256;x-date, Signature=de757de0aa3f5fc0634678be4ee6b796e858fbe0fc3a2133b46bae23408f824a"
    )


def test_parse_date():
    signer = Singer(
        service="billing",
        region="cn-beijing",
        credential_provider=StaticCredentialsProvider(access_key="test_ak", secret_key="test_sk"),
    )

    assert signer.parse_date("Mon, 09 Sep 2011 23:36:00 GMT") == "20110909"
    assert signer.parse_date("Sunday, 06-Nov-94 08:49:37 GMT") == "20941106"
    assert signer.parse_date("Wed Dec 4 00:00:00 2002") == "20021204"
    assert signer.parse_date("20100325T010101Z") == "20100325"
    assert signer.parse_date("2009-03-25T10:11:12.13-01:00") == "20090325"
    assert signer.parse_date("20100325 010101") is None
