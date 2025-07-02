# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import av

if TYPE_CHECKING:
    from av.container.input import InputContainer


def load_video(path: str, mode: str = "r") -> InputContainer:
    """Load video from a file path.

    Args:
        path (str): The video file path.
        mode (str, optional): The loading mode (default "r").

    Returns:
        av.container.input.InputContainer: The loaded video container.

    Raises:
        FileNotFoundError: If the file does not exist and mode is read.

    Example:
        video = load_video("video.mp4")
    """
    if not Path(path).exists() and "r" in mode:
        raise FileNotFoundError(f"Video [{path}] does not exist!")
    return av.open(path, mode)
