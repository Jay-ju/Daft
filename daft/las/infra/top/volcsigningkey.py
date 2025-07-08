# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime


class VolcSigningKey:
    def __init__(
        self, secret_key: str, region: str, service: str, date: str | None = None, store_secret_key: bool = True
    ) -> None:
        self.region = region
        self.service = service
        self.date = date or datetime.utcnow().strftime("%Y%m%d")
        self.scope = f"{self.date}/{self.region}/{self.service}/request"
        self.store_secret_key = store_secret_key
        self.secret_key = secret_key if self.store_secret_key else None
        self.key = self.generate_key(secret_key, self.region, self.service, self.date)

    @classmethod
    def generate_key(cls, secret_key: str, region: str, service: str, date: str) -> bytes:
        init_key = (secret_key).encode("utf-8")
        date_key = cls.sign_sha256(init_key, date)
        region_key = cls.sign_sha256(date_key, region)
        service_key = cls.sign_sha256(region_key, service)
        return cls.sign_sha256(service_key, "request")

    @staticmethod
    def sign_sha256(key: bytes, msg: bytes | str) -> bytes:
        if isinstance(msg, str):
            msg = msg.encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).digest()
