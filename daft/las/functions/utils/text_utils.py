# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import re


def strip_markdown_images(md_content: str) -> str:
    """Remove image information from the given Markdown content.

    Args:
        md_content (str): Markdown content.

    Returns:
        str: The Markdown content with image information removed.
    """
    pattern = re.compile(r'!\[(.*?)]\((.*?)\)|<img[^>]*?src="(.*?)"[^>]*?>')
    return re.sub(pattern, "", md_content)
