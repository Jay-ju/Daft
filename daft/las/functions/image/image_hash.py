# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np  # noqa: TID253
from PIL import Image  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import (
    generate_filename_base_input,
    save_file_to_local,
    tracking_usage,
)

logger = logging.getLogger(__name__)


class ImageHash(Operator):
    """**图片哈希计算算子：支持 URL、Base64、二进制三类输入格式，统一输出十六进制与二进制哈希。**

    主要功能
        - 支持五种哈希方法：`ahash`、`dhash`、`phash`、`whash`、`md5`。
        - 输入类型可选：`image_url`、`image_base64`、`image_binary`，按需落盘到临时目录后处理。
        - 批量处理：通过 `batch_size` 控制吞吐与资源使用，输出结构化字段以支持多列返回。

    适用场景
        - 去重、近似重复图片召回（基于感知哈希，如 phash/dhash/ahash/whash）。
        - 文件一致性校验（基于 md5）。
        - 需要同时获取十六进制与二进制哈希表示的多输出场景。

    注意事项
        - 输入类型需与 `image_src_type` 参数一致；URL/Base64/二进制将通过工具函数统一落盘到临时目录进行读取。
        - `md5` 为 128 位摘要，本算子将其统一为 64 位二进制输出，并对应 16 位 hex；非 md5 的感知哈希亦统一为 64 位二进制与 16 位 hex。
        - 感知哈希依赖 `imagededup.methods`（AHash/DHash/PHash/WHash），采用懒加载；若依赖不可用，将在推理时记录异常并为对应样本返回空字符串。
        - 图片读取使用 Pillow，处理为 RGB 格式后转为 numpy 数组。
    """  # noqa: D415

    def __init__(
        self,
        image_src_type: str = "image_url",
        method: str = "phash",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        """初始化图片哈希算子参数.

        Args:
            image_src_type: 输入图片的格式类型。支持：
                - "image_url": http(s)/tos/s3 等远程地址
                - "image_base64": Base64 编码字符串
                - "image_binary": 二进制字节流（bytes）
                可选值：["image_url", "image_base64", "image_binary"]
                默认值："image_url"
            method: 哈希计算方法。可选值：{"ahash", "dhash", "phash", "whash", "md5"}，默认 "phash"。
                - md5：使用 hashlib.md5 对图像字节计算摘要；输出统一为 64 位二进制与 16 位 hex（截取前 64 位）。
                - 其它：使用 imagededup.methods 中的 AHash/DHash/PHash/WHash。
            batch_size: 批处理大小，用于在吞吐与内存之间取得平衡，默认 64。
        """
        super().__init__(**kwargs)
        self.image_src_type = image_src_type
        self.method = method.lower()
        self.batch_size = batch_size

        self._supported_src_types = {"image_url", "image_base64", "image_binary"}
        self._supported_methods = {"ahash", "dhash", "phash", "whash", "md5"}

        if self.image_src_type not in self._supported_src_types:
            raise ValueError(
                f"Unsupported image_src_type: {self.image_src_type}. Supported: {sorted(self._supported_src_types)}"
            )
        if self.method not in self._supported_methods:
            raise ValueError(f"Unsupported method: {self.method}. Supported: {sorted(self._supported_methods)}")

        # 懒加载重依赖（仅在使用感知哈希时加载）
        self._hasher = None
        if self.method != "md5":
            try:
                from imagededup.methods import AHash, DHash, PHash, WHash  # heavy dep

                mapping = {"ahash": AHash, "dhash": DHash, "phash": PHash, "whash": WHash}
                self._hasher = mapping[self.method]()
            except Exception:
                # 不抛出致命错误，保留在 transform 中按样本级容错
                logger.exception("Failed to lazy-load imagededup methods for method=%s", self.method)
                self._hasher = None

        tracking_usage(
            op=self.__class__.__name__, model_service_or_lib=("imagededup" if self.method != "md5" else "hashlib")
        )

        logger.info(
            "Operator initialization configuration:\n" "- Source Type: %s\n" "- Method: %s\n" "- Batch Size: %d\n",
            self.image_src_type,
            self.method,
            self.batch_size,
        )

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                ("hash_hex", pa.string()),
                ("hash_bin", pa.string()),
            ]
        )

    @staticmethod
    def _read_image_rgb_to_numpy(file_path: str) -> np.ndarray:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
        return arr

    @staticmethod
    def _binary64_to_hex16(binary_64: str) -> str:
        # 保证为 64 位，左侧补零或截断
        b = (binary_64 or "").strip()
        b = b[:64] if len(b) >= 64 else b.zfill(64)
        return hex(int(b, 2))[2:].zfill(16)

    @staticmethod
    def _normalize_hash_output_to_hex_bin(hash_output: Any) -> tuple[str, str]:
        """将 imagededup 的输出统一为 64 位二进制和 16 位十六进制字符串.

        兼容返回类型：
        - 字符串（二进制 '0/1' 或十六进制）
        - list/ndarray（0/1 或 bool）
        """
        try:
            if hash_output is None:
                return "", ""

            if isinstance(hash_output, str):
                s = hash_output.strip()
                if s == "":
                    return "", ""
                # 二进制字符串
                if set(s) <= {"0", "1"}:
                    bin64 = s[:64] if len(s) >= 64 else s.zfill(64)
                    return ImageHash._binary64_to_hex16(bin64), bin64
                # 十六进制字符串
                if all(ch in "0123456789abcdefABCDEF" for ch in s):
                    bin_full = bin(int(s, 16))[2:].zfill(len(s) * 4)
                    bin64 = bin_full[:64] if len(bin_full) >= 64 else bin_full.zfill(64)
                    return ImageHash._binary64_to_hex16(bin64), bin64
                # 未知格式，尝试按二进制处理失败则返回空
                return "", ""

            # 列表/数组：转为扁平 0/1 字符串
            if isinstance(hash_output, (list, tuple)):
                flat = []
                for v in hash_output:
                    if isinstance(v, (list, tuple, np.ndarray)):
                        flat.extend(list(np.array(v).astype(int).flatten()))
                    else:
                        flat.append(int(bool(v)))
                s = "".join(str(x) for x in flat)
                if s:
                    bin64 = s[:64] if len(s) >= 64 else s.zfill(64)
                    return ImageHash._binary64_to_hex16(bin64), bin64
                return "", ""

            if isinstance(hash_output, np.ndarray):
                flat = list(hash_output.astype(int).flatten())
                s = "".join(str(x) for x in flat)
                if s:
                    bin64 = s[:64] if len(s) >= 64 else s.zfill(64)
                    return ImageHash._binary64_to_hex16(bin64), bin64
                return "", ""
        except Exception:
            logger.exception("Failed to normalize hash output: %s", str(hash_output)[:200])
            return "", ""

        # 其它类型不支持
        return "", ""

    @staticmethod
    def _compute_md5_hex_bin64(image_array: np.ndarray) -> tuple[str, str]:
        md5_hash = hashlib.md5()
        md5_hash.update(image_array.tobytes())
        full_hex = md5_hash.hexdigest()  # 32 hex chars (128 bits)
        bin_128 = bin(int(full_hex, 16))[2:].zfill(128)
        bin64 = bin_128[:64]
        hex16 = ImageHash._binary64_to_hex16(bin64)
        return hex16, bin64

    def _materialize_to_local(self, image_input: Any, tmp_dir: str, batch_idx: int, idx: int) -> str:
        """将输入素材按类型落盘至临时目录并返回本地文件路径.

        使用统一工具函数 generate_filename_base_input/save_file_to_local；
        对于本地路径字符串（不匹配三种类型），直接复制到临时目录。
        """
        try:
            # 统一生成文件名，后缀使用 png
            file_name = generate_filename_base_input(image_input, self.image_src_type, "png", batch_idx, idx)
            if self.image_src_type in self._supported_src_types:
                tmp_file = save_file_to_local(image_input, self.image_src_type, tmp_dir, file_name)
                logger.debug("Saved input to local tmp: %s", tmp_file)
                return tmp_file
        except Exception:
            logger.exception(
                "Failed to save input by source-type path, fallback to local path copy (batch=%d, idx=%d)",
                batch_idx,
                idx,
            )

        # 如果不是三种类型或保存失败，尝试直接将本地文件复制到临时目录
        try:
            src = str(image_input)
            src_path = Path(src)
            if not src_path.exists():
                raise FileNotFoundError(src)
            dst_path = Path(tmp_dir) / src_path.name
            # 直接读取时无需强制复制；为隔离处理，这里使用硬链接或复制
            try:
                dst_path.write_bytes(src_path.read_bytes())
            except Exception:
                # 回退到简单复制
                import shutil

                shutil.copyfile(src_path, dst_path)
            logger.debug("Copied local file to tmp: %s", dst_path)
            return str(dst_path)
        except Exception:
            logger.exception("Local path copy failed for input: %s", str(image_input)[:200])
            raise

    def transform(self, images: pa.Array) -> pa.Array:
        """对输入的图片数组进行批量处理，生成统一的哈希结果（hex 与 64 位二进制）。

        Args:
            images: 包含图片数据的数组，元素类型为 字符串 或 二进制（bytes），其真实含义由 `image_src_type` 指定。

        Returns:
            结构化数组，每个元素为 {"hash_hex": str, "hash_bin": str}。

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出。
        """  # noqa: D415
        start_time = time.monotonic()
        logger.info(
            "Starting batch processing | input type: %s | method: %s | batch_size: %d",
            self.image_src_type,
            self.method,
            self.batch_size,
        )

        total_images = len(images)
        total_batches = (total_images + self.batch_size - 1) // self.batch_size
        results: list[dict[str, str]] = []

        for batch_start in range(0, total_images, self.batch_size):
            sub = images.slice(batch_start, self.batch_size)
            current_batch = sub.to_pylist()
            batch_index = batch_start // self.batch_size
            if not current_batch:
                break

            logger.debug(
                "Processing batch %d/%d with %d images (progress=%.2f)",
                batch_index + 1,
                total_batches,
                len(current_batch),
                (batch_index + 1) / max(total_batches, 1),
            )

            with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                for idx, image_input in enumerate(current_batch):
                    try:
                        tmp_file = self._materialize_to_local(image_input, tmp_dir, batch_index, idx)
                        if not Path(tmp_file).exists():
                            raise FileNotFoundError(tmp_file)

                        # 读取图片为 RGB 的 numpy 数组
                        img_arr = self._read_image_rgb_to_numpy(tmp_file)

                        # 计算哈希
                        if self.method == "md5":
                            hex16, bin64 = self._compute_md5_hex_bin64(img_arr)
                        else:
                            if self._hasher is None:
                                raise RuntimeError(
                                    f"Hasher not initialized for method {self.method}. Check imagededup installation."
                                )
                            raw = self._hasher.encode_image(image_array=img_arr)
                            hex16, bin64 = self._normalize_hash_output_to_hex_bin(raw)

                        results.append({"hash_hex": hex16, "hash_bin": bin64})

                    except Exception as e:
                        logger.exception(
                            "image (batch=%d, idx=%d, src_type=%s) processing failed: %s",
                            batch_index,
                            idx,
                            self.image_src_type,
                            str(e),
                        )
                        results.append({"hash_hex": "", "hash_bin": ""})

        processing_time = time.monotonic() - start_time
        throughput = (total_images / processing_time) if processing_time > 0 else 0.0

        logger.info(
            "Completed %d images | Total time: %.2fs | Throughput: %.2f image/s",
            total_images,
            processing_time,
            throughput,
        )

        return pa.array(results, type=self.__return_column_type__())
