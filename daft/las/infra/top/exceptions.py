# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations


class RequestsVolcAuthException(Exception):
    pass


class DateMismatchError(RequestsVolcAuthException):
    pass


class NoSecretKeyError(RequestsVolcAuthException):
    pass


class DateFormatError(RequestsVolcAuthException):
    pass
