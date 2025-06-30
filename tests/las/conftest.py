from __future__ import annotations

import os
import uuid

import pytest

from daft.las.io import exists, rm


@pytest.fixture
def object_store_test_dir(request):
    scheme = os.getenv("TEST_OBJECT_SCHEME", "tos")
    bucket = os.getenv("TEST_OBJECT_BUCKET", "las-ci")
    temp_dir = f"{scheme}://{bucket}/object-store/test-{uuid.uuid4().hex!s}"

    def clean_temp_data():
        if exists(temp_dir):
            rm(temp_dir)

    request.addfinalizer(clean_temp_data)

    return temp_dir
