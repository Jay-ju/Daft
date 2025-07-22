# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from daft.las.functions.utils.common_utils import (
    byte_to_base64,
    path_to_base64,
    pre_sign_url_for_tos,
    run_on_local_path,
)


def _build_text_query(raw_query: str, prompt: str | None = None) -> str:
    """Builds a text query by formatting the prompt with the raw query."""
    if prompt is None:
        return raw_query
    return prompt.format(query=raw_query) if "{query}" in prompt else prompt + raw_query


def gen_text_message(raw_query: str, prompt: str | None = None, system_content: str | None = None) -> Any:
    """Generates a message for the model based on the input row."""
    if raw_query is None or (raw_query.strip() == ""):
        return None

    new_query = _build_text_query(raw_query, prompt)
    user_message = {"role": "user", "content": new_query}
    if system_content is None:
        return [user_message]

    return [{"role": "system", "content": system_content}, user_message]


def gen_media_data(data_type: str, media_info: Any, media_type: str, source_type: str = "url") -> str:
    """Generates media data for the model based on the input row."""
    source_type = source_type.lower() if source_type else "url"
    assert source_type in ["url", "base64", "binary"]

    if source_type == "base64":
        return f"data:{data_type}/{media_type};base64,{media_info}"

    if source_type == "binary":
        if isinstance(media_info, str):
            media_info = media_info.encode()
        base64_data = byte_to_base64(media_info)
        return f"data:{data_type}/{media_type};base64,{base64_data}"

    # Now start to process url
    scheme = urlparse(media_info).scheme
    if scheme and scheme in ["http", "https"]:
        # For http data, ark llm can support http/https url, so we don't need to convert it to base64
        return media_info

    if scheme and scheme in ["tos", "s3"]:
        sign_url = pre_sign_url_for_tos(media_info)
        return sign_url

    base64_data = run_on_local_path(media_info, lambda path: path_to_base64(path))
    return f"data:{data_type}/{media_type};base64,{base64_data}"
