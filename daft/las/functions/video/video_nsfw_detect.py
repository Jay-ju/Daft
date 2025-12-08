from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np  # noqa: TID253
import torch
from PIL import Image  # noqa: TID253
from transformers import AutoModelForImageClassification, ViTImageProcessor

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.functions.video.video_frame_sampler import VideoFrameSampler

logger = logging.getLogger(__name__)


class VideoNsfwDetect(Operator):
    """**视频安全性(NSFW)检测器：多源输入、统一帧采样与批量推理。**

    核心能力：
    - 基于预训练图像分类模型，对视频采样得到的帧进行 NSFW 概率检测，并按聚合策略输出最终分数。
    - 支持多种视频输入来源：
      - URL 路径（video_url）
      - Base64 编码（video_base64）
      - 二进制流（video_binary）
    - 支持多种采样方式（通过 VideoFrameSampler）：
      - by_count_uniform / by_interval_time / by_interval_frames / by_fps / by_timestamps
    - 批量推理：
      - 可通过 batch_size 控制推理批量大小，提高吞吐性能。

    适用场景：
    - 内容安全审核：对用户上传视频进行 NSFW 风险评估。
    - 生产管线的预过滤：在后续处理前进行视频安全性筛查。

    注意事项：
    - 仅输出数值型 NSFW 置信度（0~1），不保存帧图片。
    """  # noqa: D415

    def __init__(
        self,
        video_src_type: str = "video_url",
        model_path: str = "/opt/las/models",
        model_name: str = "Falconsai/nsfw_image_detection",
        dtype: str = "float16",
        # 采样相关参数
        sample_mode: str = "by_count_uniform",
        start_time_sec: float = 0.0,
        end_time_sec: float | None = None,
        count_k: int | None = None,
        interval_sec: float | None = None,
        interval_frames: int | None = None,
        target_fps: float | None = None,
        timestamps_sec: list[float] | None = None,
        max_frames: int | None = None,
        reduce_mode: str = "avg",
        video_format: str = "mp4",
        batch_size: int = 16,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """视频 NSFW 检测器初始化方法.

        Args:
            video_src_type: 输入视频的格式类型，支持：
                - 路径/URL（video_url）
                - base64 编码（video_base64）
                - 二进制流（video_binary）
                可选值: ["video_url", "video_base64", "video_binary"]
                默认值: "video_url"
            model_path: 模型基础路径。
                默认值: "/opt/las/models"
            model_name: 模型名称/子目录。
                可选值: ["Falconsai/nsfw_image_detection"]
                默认值: "Falconsai/nsfw_image_detection"
            dtype: 模型推理精度选择：
                - float16: 更快推理
                - float32: 更高精度
                可选值：["float16", "float32"]，默认值："float16"
            batch_size: 批量推理的帧数量，默认 16。
            rank: GPU 编号，默认 0。
            sample_mode: 采样模式，默认 "by_count_uniform"。
            start_time_sec: 采样起始时间（秒），默认 0.0。
            end_time_sec: 采样结束时间（秒），None 表示到视频末尾。
            count_k: 均匀采样帧数（by_count_uniform 使用）。
            interval_sec: 时间间隔（秒，by_interval_time 使用）。
            interval_frames: 解码帧间隔（by_interval_frames 使用）。
            target_fps: 目标采样 FPS（by_fps 使用）。
            timestamps_sec: 采样时间戳列表（秒，by_timestamps 使用）。
            max_frames: 返回帧上限，None 表示不限制。
            reduce_mode: 多帧聚合策略，可选值："avg"|"max"|"min"；默认 "avg"。
            video_format: 二进制/base64 输入的视频格式（如 mp4、mov），默认 "mp4"。
        """
        super().__init__(**kwargs)
        self.video_src_type = video_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.rank = rank

        # 采样配置
        self.sample_mode = sample_mode
        self.start_time_sec = start_time_sec
        self.end_time_sec = end_time_sec
        self.count_k = count_k
        self.interval_sec = interval_sec
        self.interval_frames = interval_frames
        self.target_fps = target_fps
        self.timestamps_sec = timestamps_sec
        self.max_frames = max_frames
        self.reduce_mode = reduce_mode
        self.video_format = (video_format or "mp4").lower()

        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_mapping.get(dtype)
        if not self.torch_dtype:
            raise ValueError(f"Unsupported precision type: {dtype}")

        if self.video_src_type not in ["video_url", "video_base64", "video_binary"]:
            logger.error("Unsupported video source type: %s", self.video_src_type)
            raise ValueError(f"Invalid video source type: {self.video_src_type}")

        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            self.device: str = "cuda" if use_gpu else "cpu"
        else:
            self.device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
        logger.info("Model will be loaded on device: %s", self.device)

        model_dir = str(Path(self.model_path) / self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.processor = ViTImageProcessor.from_pretrained(model_dir)

        # 采样器初始化
        self.sampler = VideoFrameSampler(
            sample_mode=self.sample_mode,
            start_time_sec=self.start_time_sec,
            end_time_sec=self.end_time_sec,
            count_k=self.count_k,
            interval_sec=self.interval_sec,
            interval_frames=self.interval_frames,
            target_fps=self.target_fps,
            timestamps_sec=self.timestamps_sec,
            max_frames=self.max_frames,
            output_frames=True,
            output_base64=False,
        )

        logger.info("Model path is %s", self.model_path)
        logger.info("Model name is %s", self.model_name)
        logger.info("Video source type is %s", self.video_src_type)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="Falconsai/nsfw_image_detection")

    def _sample_all_videos(self, videos_py: list[Any]) -> list[Any]:
        """调用 VideoFrameSampler 对整列视频进行批量帧采样，返回结构体字典列表."""
        n = len(videos_py)
        try:
            if self.video_src_type == "video_url":
                arr_paths = pa.array(videos_py)
                out = self.sampler.transform(video_paths=arr_paths)
            else:
                # 二进制 / base64 输入
                binaries: list[bytes | None] = []
                for v in videos_py:
                    try:
                        if v is None:
                            binaries.append(None)
                        elif self.video_src_type == "video_binary":
                            if isinstance(v, (bytes, bytearray)):
                                binaries.append(bytes(v))
                            else:
                                binaries.append(None)
                        elif self.video_src_type == "video_base64":
                            if isinstance(v, str):
                                binaries.append(base64.b64decode(v))
                            else:
                                binaries.append(None)
                    except Exception:
                        logger.exception("Failed to decode base64/binary for one item")
                        binaries.append(None)
                arr_bins = pa.array(binaries)
                arr_fmts = pa.array([self.video_format] * n)
                out = self.sampler.transform(video_binaries=arr_bins, video_formats=arr_fmts)
        except Exception:
            logger.exception("Video frame sampling failed for entire column")
            return [{"frames": [], "timestamps": [], "frame_indices": []} for _ in range(n)]

        return out.to_pylist()

    def transform(self, videos: pa.Array) -> pa.Array:
        """批量执行视频 NSFW 检测（解码与采样 -> 批量推理 -> 按视频聚合）。

        Args:
            videos: 输入视频列，类型依据 video_src_type（url/base64/binary）。

        Returns:
            每个视频的 NSFW 置信度分数；若采样/推理失败或无有效帧，则为 None。
        """  # noqa: D415
        videos_py = [v.as_py() for v in videos]
        video_cnt = len(videos_py)
        final_scores: list[float | None] = [None] * video_cnt

        # 第一步：批量采样帧
        sample_results = self._sample_all_videos(videos_py)

        # 收集所有帧为 PIL 图像，并建立映射 video_idx -> scores 列表
        frames_pairs: list[tuple[int, Image.Image]] = []
        scores_by_video: list[list[float]] = [[] for _ in range(video_cnt)]

        for vidx, res in enumerate(sample_results):
            frames_list = res.get("frames") or []
            if not isinstance(frames_list, list):
                frames_list = []
            for f_list in frames_list:
                try:
                    # frames 为 BGR ndarray 列表，当前传输为嵌套 Python list，需还原并转 RGB
                    arr_bgr = np.array(f_list, dtype=np.uint8)
                    if arr_bgr.ndim != 3 or arr_bgr.shape[-1] != 3:
                        raise ValueError("Invalid frame shape")
                    arr_rgb = arr_bgr[:, :, ::-1]
                    img = Image.fromarray(arr_rgb).convert("RGB")
                    frames_pairs.append((vidx, img))
                except Exception:
                    logger.exception("Failed to convert frame to PIL.Image for video %d", vidx)
                    continue

        # 第二步：对所有帧进行批量推理
        total_frames = len(frames_pairs)
        if total_frames == 0:
            logger.warning("No valid frames extracted for NSFW detection")
            return pa.array(final_scores, type=self.__return_column_type__())

        with torch.no_grad():
            for start in range(0, total_frames, self.batch_size):
                end = min(total_frames, start + self.batch_size)
                batch_pairs = frames_pairs[start:end]
                if not batch_pairs:
                    continue
                try:
                    batch_indices = [vidx for vidx, _ in batch_pairs]
                    batch_images = [img for _, img in batch_pairs]
                    inputs = self.processor(images=batch_images, return_tensors="pt").to(self.model.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    nsfw_probs = torch.softmax(logits, dim=-1)
                    nsfw_scores = [float(p[1]) for p in nsfw_probs]
                    for vidx, score in zip(batch_indices, nsfw_scores):
                        scores_by_video[vidx].append(score)
                except RuntimeError as e:
                    logger.exception("Model inference failed (possibly OOM): %s", str(e))
                    logger.info("Current batch size: %d, consider reducing batch_size", self.batch_size)
                except Exception:
                    logger.exception("Inference error on frames [%d:%d]", start, end)
                    continue

        # 第三步：按视频进行聚合
        for i, group in enumerate(scores_by_video):
            try:
                if not group:
                    final_scores[i] = None
                    continue
                if self.reduce_mode == "max":
                    final_scores[i] = round(float(max(group)), 6)
                elif self.reduce_mode == "min":
                    final_scores[i] = round(float(min(group)), 6)
                else:  # default avg
                    final_scores[i] = round(float(sum(group) / len(group)), 6)
            except Exception:
                logger.exception("Failed to reduce scores for video %d", i)
                final_scores[i] = None

        logger.info("Video NSFW detection completed. Total processed: %d videos", video_cnt)
        return pa.array(final_scores, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
