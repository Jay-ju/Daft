# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import io
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.image_utils import decode_image
from daft.las.io import upload_file

logger = logging.getLogger(__name__)


class ImageResample(Operator):
    """**图像重采样处理器，支持多种插值算法和输出格式。**

    **核心功能**
    - 提供4种专业级插值算法：
        - 最近邻插值（nearest） - 速度最快，适合像素艺术
        - 双线性插值（bilinear） - 平衡速度与质量
        - 双三次插值（bicubic） - 高精度平滑处理
        - Lanczos插值（lanczos） - 抗锯齿最佳，适合照片
    - 多格式输入支持：
        - URL
        - Base64编码
        - 二进制流
    - 双输出模式：
        - Base64编码直出
        - TOS持久化存储
    """  # noqa: D415

    def __init__(
        self,
        image_suffix: str = ".jpg",
        tos_dir: str = "",
        local_dir: str = "",
        image_src_type: str = "image_url",
        target_size: list[int] = [200, 200],
        target_dpi: list[int] = [72, 72],
        method: str = "lanczos",
        **kwargs: Any,
    ) -> None:
        """图像处理器初始化方法.

        Args:
            image_suffix: 保存到 TOS 中的图像格式。
                描述: 保存到 TOS 中的图像格式，默认为'.jpg'
                可选值: [".jpg", ".png"]
                默认值: ".jpg"
            tos_dir: 保存图像的 TOS 文件夹路径。
                描述: 将重采样后的图像保存到的 TOS 路径，如果不设置，则不会将重采样后的图像保存到 TOS 中。
                默认值: ""
            local_dir: 保存图像的本地文件夹路径。
                描述: 调整后图像的本地存储目录，如果为空，则调整后的图像不保存到本地。
                默认值: ""
            image_src_type: 输入图像的格式类型，支持：
                - url地址(image_url)
                - base64编码(image_base64)
                - 二进制流(image_binary)
                描述: 输入图像的格式类型。
                可选值: ["image_url", "image_base64", "image_binary"]
                默认值: "image_url"
            target_size: 重采样后的图像尺寸。
                描述: 重采样后的图像尺寸，格式为 [width, height]，默认为 [200, 200]。
                默认值: [200, 200]
            target_dpi: 图像 DPI。
                描述: 图像 DPI，格式为 [width, height]，默认为 [72, 72]。
                默认值: [72, 72]
            method: 重采样方法。
                描述: 重采样方法，支持 nearest(最近邻插值)、bilinear(双线性插值)、bicubic(双三次插值)、lanczos(Lanczos插值)，默认为 lanczos。
                可选值: ["nearest", "bilinear", "bicubic", "lanczos"]
                默认值: "lanczos"
        """
        super().__init__(**kwargs)
        self.image_suffix = image_suffix
        self.tos_dir = tos_dir
        self.local_dir = local_dir
        self.image_src_type = image_src_type
        self.target_size = target_size
        self.target_dpi = target_dpi
        self.method = method

        if self.image_src_type not in ["image_url", "image_base64", "image_binary"]:
            logger.error("Unsupported image source type: %s", self.image_src_type)
            raise ValueError(f"Invalid image source type: {self.image_src_type}")

        self.tos_dir = self.tos_dir.strip("/") if self.tos_dir else ""

    def _resample_image(self, image: Image.Image) -> tuple[Image.Image, str]:
        method_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }

        try:
            resample_method = method_map.get(self.method.lower(), None)

            if resample_method is None:
                logger.warning("Unsupported resample method: %s, using original image", self.method)
                return image, ""

            resampled_image = image.resize(self.target_size, resample_method)

            with io.BytesIO() as buffer:
                resampled_image.save(buffer, format="png")
                img_bytes = buffer.getvalue()

            return resampled_image, base64.b64encode(img_bytes).decode("utf-8")

        except Exception:
            logger.exception(
                "Failed to resample image with method %s and size %s. Error",
                self.method,
                self.target_size,
            )
            return None, ""

    def _generate_filename_prefix(self, idx: int, original_images: list[str], original_images_name: list[str]) -> str:
        if original_images_name and len(original_images_name) == len(original_images):
            name = ".".join(original_images_name[idx].split(".")[:-1])
        elif self.image_src_type == "image_url":
            name = Path(original_images[idx]).stem
        else:
            name = f"{int(time.time())!s}_{(random.randint(1, 1000000))!s}"
            logger.info("Generated filename prefix: %s", name)
        return f"{name}_resample"

    def _save(self, image: Image.Image, prefix: str) -> str | None:
        """Save resampled images."""
        filename = f"{prefix}{self.image_suffix}"
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            if self.local_dir:
                tmp_dir = self.local_dir
            tmp_path = str(Path(tmp_dir) / filename)
            logger.info("Temporary storage path: %s", tmp_path)
            try:
                image.save(tmp_path, dpi=self.target_dpi)
                logger.info("Target dpi: %s", self.target_dpi)
                if self.tos_dir:
                    upload_file(str(tmp_path), self.tos_dir + "/" + filename, overwrite=True)
                    logger.debug("Tos path: %s/%s", self.tos_dir, filename)
                    return f"{self.tos_dir}/{filename}"
                if self.local_dir:
                    return tmp_path
            except Exception:
                logger.exception("Failed to save!")
                raise
        return None

    def transform(self, images: pa.Array, images_name: pa.Array = None) -> pa.Array:
        """批量执行图像重采样处理，支持多种输入格式和输出配置。

        Args:
            images: 包含输入图像的数组，支持URL/base64/二进制格式
            images_name: 可选参数，包含图像标识名的数组，用于生成输出文件名

        Returns:
            pyarrow.Array: 包含处理结果的字典数组，每个元素包含：
                - base64: 重采样后图像的base64编码；
                - image_path: 本地/TOS存储路径（当配置输出目录时有效）
        """  # noqa: D415
        images = [image.as_py() for image in images]

        if images_name:
            images_name = [image.as_py() for image in images_name]
        else:
            images_name = []

        images_cnt = len(images)
        base64_results = []
        save_paths_results = []

        for idx, img_data in enumerate(images):
            try:
                img = decode_image(img_data, self.image_src_type)
                resampled_img, img_base64 = self._resample_image(img)
                if not resampled_img:
                    raise ValueError("Empty resampling result")

                if self.tos_dir or self.local_dir:
                    prefix = self._generate_filename_prefix(idx, images, images_name)
                    resampled_img_path = self._save(resampled_img, prefix)

            except Exception:
                logger.exception("Image resampling failed [index: %d ].", idx)
                img_base64 = None
                resampled_img_path = None

            save_paths_results.append(resampled_img_path)
            base64_results.append(img_base64)
        logger.info("Batch processing completed. Total processed: %d images", images_cnt)

        results = []
        for i in range(images_cnt):
            record = {"base64": base64_results[i]}
            if self.tos_dir or self.local_dir:
                record["image_path"] = save_paths_results[i]
            else:
                record["image_path"] = None
            results.append(record)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [pa.field("base64", pa.string()), pa.field("image_path", pa.string())]
        return pa.struct(fields)
