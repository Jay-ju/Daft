# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def get_version():
    """Get the version from pyproject.toml."""
    version = os.getenv("OP_VERSION")
    if version is not None:
        return version

    import tomllib

    with (Path(__file__).parent.parent.parent / "pyproject.toml").open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
        parts = version.split(".")
        if len(parts) <= 3:
            return version
        return ".".join(parts[:3])


OP_BUCKET = os.getenv("OP_BUCKET", "las-ai-cn-beijing")
OP_ENVIRONMENT = os.getenv("OP_ENVIRONMENT", "qa")  # baseline/qa/dev
OP_REGION = os.getenv("OP_REGION", "cn-beijing")
OP_VERSION = get_version()


class Category(Enum):
    """The categories of the functions."""

    DEFAULT = ""
    OTHER = "其他"
    IO = "输入输出"
    TABLE = "表"
    TEXT = "文本"
    DOC = "文档"
    AUDIO = "音频"
    IMAGE = "图片"
    VIDEO = "视频"
    MULTI_MODAL = "多模态"
    LLM_ONLINE_REASONING = "大模型在线推理"
    LLM_OFFLINE_REASONING = "大模型批量推理接入点"


class SubCategory(Enum):
    """The sub-categories of the functions."""

    DEFAULT = ""
    OTHER = "其他"

    # text
    TEXT_CLEAN = "文本清洗"
    TEXT_PROCESSING = "文本处理"
    TEXT_EMBEDDING = "文本向量化"
    TEXT_CLASSIFICATION = "文本分类"
    TEXT_TRANSLATION = "文本翻译"
    TEXT_SIMILARITY = "文本相似度计算"
    TEXT_SUMMARIZATION = "文本摘要"
    TEXT_GENERATION = "文本生成"
    TEXT_RANKING = "文本排序"
    TEXT_ENTITY_RECOGNITION = "文本实体识别"
    TEXT_QUALITY_ASSESSMENT = "文本质量评估"
    TEXT_CONTENT_SECURITY = "文本安全识别"

    # audio
    AUDIO_CLEAN = "音频清洗"
    AUDIO_PROCESSING = "音频处理"
    AUDIO_EMBEDDING = "音频向量化"
    AUDIO_CLASSIFICATION = "音频分类"
    AUDIO_RECOGNITION = "音频识别"
    AUDIO_GENERATION = "音频生成"
    AUDIO_QUALITY_ASSESSMENT = "音频质量评估"
    AUDIO_CONTENT_SECURITY = "音频安全识别"

    # image
    IMAGE_CLEAN = "图片清洗"
    IMAGE_PROCESSING = "图片处理"
    IMAGE_EMBEDDING = "图片向量化"
    IMAGE_CLASSIFICATION = "图片分类"
    IMAGE_SEGMENTATION = "图片分割"
    IMAGE_OCR = "图片OCR"
    IMAGE_GENERATION = "图片生成"
    IMAGE_ENTITY_RECOGNITION = "图像实体识别"
    IMAGE_QUALITY_ASSESSMENT = "图片质量评估"
    IMAGE_CONTENT_SECURITY = "图片安全识别"

    # video
    VIDEO_CLEAN = "视频清洗"
    VIDEO_PROCESSING = "视频处理"
    VIDEO_EMBEDDING = "视频向量化"
    VIDEO_CLASSIFICATION = "视频分类"
    VIDEO_GENERATION = "视频生成"
    VIDEO_QUALITY_ASSESSMENT = "视频质量评估"
    VIDEO_CONTENT_SECURITY = "视频安全识别"

    # doc
    DOC_PARSE = "文档解析"
    DOC_CONVERSION = "文档格式转换"

    # multi modal
    IMAGE_TO_TEXT = "图片理解"
    AUDIO_TO_TEXT = "音频理解"
    VIDEO_TO_TEXT = "视频理解"
    IMAGE_TO_AUDIO = "图片转音频"
    IMAGE_TO_VIDEO = "图片转视频"
    VIDEO_TO_AUDIO = "视频转音频"
    VIDEO_TO_IMAGE = "视频转图片"


class ValueType(Enum):
    """Value type that sync to the frontend to render the page."""

    Text = (1,)
    List = (2,)
    Dict = (3,)
    Picture = (4,)
    PictureWall = (5,)
    Video = (6,)
    VideoWall = (7,)
    Audio = (8,)
    AudioWall = (9,)
    File = (10,)


@dataclass
class DataItem:
    """The data time to be rendered."""

    Type: str
    Value: str
    Description: str


@dataclass
class ExtraMetaModel:
    Code: str
    CodeDescription: str
    BeforeData: list[DataItem]
    AfterData: list[DataItem]
    Published: bool = False


@dataclass
class OpMetaModel:
    """The description of the operator."""

    Name: str
    Clazz: str
    Description: str
    Category: str
    SubCategory: str
    Tags: list[str] | None = None
    Precondition: str | None = None
