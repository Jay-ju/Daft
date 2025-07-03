# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import Any


def build_text_query(raw_query: str, prompt: str | None = None) -> str:
    """Builds a text query by formatting the prompt with the raw query."""
    raw_query = "" if raw_query is None else raw_query
    if prompt is None:
        return raw_query
    return prompt.format(query=raw_query) if "{query}" in prompt else prompt + raw_query


def gen_text_message(raw_query: str, prompt: str | None = None, system_content: str | None = None) -> Any:
    """Generates a message for the model based on the input row."""
    new_query = build_text_query(raw_query, prompt)
    user_message = {"role": "user", "content": new_query}
    if system_content is None:
        return [user_message]

    return [{"role": "system", "content": system_content}, user_message]
