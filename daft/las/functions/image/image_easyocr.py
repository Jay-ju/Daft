# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.functions.utils.image_utils import decode_image

logger = logging.getLogger(__name__)


class ImageEasyOcr(Operator):
    """**基于 EasyOCR 的多语言OCR识别组件，支持中英文混合场景下的文本检测与识别。**

    **核心功能**
    - 支持 100+ 种语言识别（需配置对应语言模型）
    - 输入格式兼容：
        - TOS URL
        - Base64编码
        - 二进制流
        - Numpy数组
    - 性能优化：
        - GPU 加速推理
        - 模型量化（默认开启）
        - 批量处理优化

    **多语言支持**
    - 简体中文(ch_sim)
    - 英文(en)
    - 日文(ja)
    - 韩文(ko)
    - 法文(fr)
    - 德文(de)
    完整语言列表参考官方文档：https://www.jaided.ai/easyocr/
    """  # noqa: D415

    def __init__(
        self,
        image_src_type: str = "image_url",
        model_path: str = "/opt/las/models",
        model_name: str = "EasyOCR",
        quantize: bool = True,
        lang_list: list[str] = ["en", "ch_sim"],
        batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        r"""OCR处理器初始化方法.

        Args:
            image_src_type: 输入图像的格式类型，支持：
                - tos/http 地址 (image_url)
                - base64 编码 (image_base64)
                - 二进制流 (image_binary)
                可选值: ["image_url", "image_base64", "image_binary"]
                默认值: "image_url"
            model_path: 模型存储路径。
                默认值: "/opt/las/models"
            model_name: 使用的模型名称，当前仅支持 "EasyOCR"。
                默认值: "EasyOCR"
            quantize: 是否启用模型量化加速推理。
                默认值: True
            lang_list: 支持识别的语言列表，详见 https://www.jaided.ai/easyocr/
                默认值: ["en", "ch_sim"]
            batch_size: GPU 推理批次大小（可根据显存调整）。
                默认值: 16
        """
        super().__init__(**kwargs)
        self.image_src_type = image_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.quantize = quantize
        self.lang_list = lang_list
        self.batch_size = batch_size

        if self.image_src_type not in ["image_url", "image_binary", "image_base64"]:
            logger.error("Invalid image source type: %s", self.image_src_type)
            raise ValueError(f"Unsupported image source type: {self.image_src_type}")

        import easyocr

        model_dir = Path(self.model_path) / self.model_name
        try:
            logger.info(
                "Initializing EasyOCR with config:\n"
                "- Model directory: %s\n"
                "- Languages: %s\n"
                "- Device: %s\n"
                "- Quantization: %s",
                model_dir,
                self.lang_list,
                "GPU" if self.use_gpu else "CPU",
                self.quantize,
            )

            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory {model_dir} not found")

            self.reader = easyocr.Reader(
                model_storage_directory=str(model_dir),
                lang_list=self.lang_list,
                recognizer=True,
                verbose=False,
                gpu=self.use_gpu,
                download_enabled=False,
                quantize=self.quantize,
            )
            logger.info("EasyOCR initialized successfully")

        except Exception:
            logger.exception(
                "Failed to initialize EasyOCR! Model config: %s\n GPU available: %s", model_dir, self.use_gpu
            )
            raise RuntimeError("OCR model not initialized!")

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _ocr(self, current_batch: list[Any]) -> list[str | None]:
        import numpy as np

        batch_ocr_results: list[str | None] = []
        logger.debug("Performing image preprocessing")
        preprocess_start = time.time()

        for _, img_data in enumerate(current_batch):
            try:
                image_array = np.array(decode_image(img_data, self.image_src_type))

                if image_array.shape[-1] == 4:
                    image_array = image_array[:, :, :3]

                logger.debug("Image array shape: %s", image_array.shape)
                ocr_result = self.reader.readtext(image=image_array, batch_size=self.batch_size)
                text = "\n".join([res[1] for res in ocr_result])
                batch_ocr_results.append(text)
            except Exception:
                batch_ocr_results.append(None)
                logger.exception("OCR processing error!")

        preprocess_time = time.time() - preprocess_start
        logger.info("Model inference completed | " "Preprocess: %.2fs", preprocess_time)

        return batch_ocr_results

    def transform(self, images: pa.Array) -> pa.Array:
        """批量处理图像数组并进行OCR识别.

        Args:
            images: 包含图像数据的数组，支持URL/base64/二进制格式

        Returns:
            pyarrow.Array: 包含OCR识别结果的数组，元素类型为字符串
        """
        all_ocr_results = []
        total_images = len(images)
        logger.info("Starting batch processing of %d images", total_images)

        current_batch = images.to_pylist()
        if current_batch:
            all_ocr_results = self._ocr(current_batch)

        logger.info("Batch processing completed | Total images: %s", len(all_ocr_results))
        return pa.array(all_ocr_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
