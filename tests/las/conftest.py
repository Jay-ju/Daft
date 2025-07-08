# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import uuid

import pytest

from daft.las.io import exists, rm


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


@pytest.fixture
def uuid_short():
    return str(uuid.uuid4()).split("-")[0]


@pytest.fixture
def object_store_test_dir(request, uuid_short):
    scheme = os.getenv("TEST_OBJECT_SCHEME", "tos")
    bucket = os.getenv("TEST_OBJECT_BUCKET", "las-ci")
    temp_dir = f"{scheme}://{bucket}/object-store/test-{uuid_short}"

    def clean_temp_data():
        if exists(temp_dir):
            rm(temp_dir)

    request.addfinalizer(clean_temp_data)

    return temp_dir
