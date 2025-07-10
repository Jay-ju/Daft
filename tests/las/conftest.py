# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pytest


@pytest.fixture
def tos_test_data_dir():
    scheme = os.getenv("TEST_OBJECT_SCHEME", "tos")
    return os.getenv("TEST_TOS_DATA_PATH", f"{scheme}://las-ai-cn-beijing/qa/test-data")


@pytest.fixture
def local_test_data_dir():
    return os.getenv("TEST_LOCAL_DATA_PATH", "/data00/tiger/las/test-data")


@pytest.fixture
def http_test_data_dir():
    return os.getenv("TEST_HTTP_DATA_PATH", "https://las-ai-cn-beijing.tos-cn-beijing.volces.com/qa/test-data")


@pytest.fixture
def local_models_dir():
    return os.getenv("TEST_MODELS_PATH", "/data00/tiger/las/models")
