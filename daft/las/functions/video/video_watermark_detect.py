# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class VideoWatermarkDetect(Operator):
    """**视频水印检测**

    **核心功能**

    - 智能检测：基于一致性分析识别视频中的固定文本水印区域
    - 采样优化：通过智能采样策略提升检测效率和准确性
    - 格式兼容：输出格式与视频修复算子完全兼容
    - CPU处理：当前版本仅支持CPU模式处理
    - 多输入支持：支持路径输入和二进制输入

    **推荐实践**
    - 建议处理分辨率不超过1080p的视频
    - 采样帧数可根据视频长度调整
    - 一致性阈值建议保持在0.7-0.9之间

    **技术特性**
    - 位置一致性：检测在多帧中位置稳定的文本区域
    - 坐标标准化：输出标准化的区域坐标格式
    - 检测加去除：与视频修复算子组合使用，实现水印去除
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "PP-OCRv4/ch_det",
        sample_count: int = 15,
        consistency_threshold: float = 0.8,
        position_tolerance: int = 15,
        **kwargs: Any,
    ) -> None:
        """初始化视频水印检测算子

        Args:
            model_path: 模型文件存储的根目录路径
                默认值："/opt/las/models"
            model_name: 使用的OCR检测模型名称
                默认值："PP-OCRv4/ch_det"
            sample_count: 从视频中采样的帧数，用于水印一致性检测
                默认值：15
            consistency_threshold: 水印位置一致性阈值，范围0-1，值越高要求越严格
                默认值：0.8
            position_tolerance: 位置容差像素值，用于判断多帧间文本位置是否一致
                默认值：15
        """  # noqa: D415
        super().__init__(**kwargs)

        self.model_path = model_path
        self.model_name = model_name
        self.sample_count = sample_count
        self.consistency_threshold = consistency_threshold
        self.position_tolerance = position_tolerance
        self.model_dir = str(Path(self.model_path) / self.model_name)

        self._text_detector = None

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="paddleocr")

    def _get_text_detector(self) -> Any:
        if self._text_detector is None:
            self._text_detector = self._initialize_detector()
        return self._text_detector

    def _initialize_detector(self) -> Any:
        try:
            import paddle

            paddle.disable_signal_handler()
            import argparse

            from paddleocr.tools.infer.predict_det import TextDetector

            args = argparse.Namespace()
            args.det_algorithm = "DB"
            args.det_model_dir = self.model_dir
            args.use_gpu = False
            args.use_onnx = False
            args.use_tensorrt = False
            args.use_npu = False
            args.use_xpu = False
            args.use_mlu = False
            args.use_gcu = False
            args.ir_optim = True
            args.precision = "fp32"
            args.gpu_mem = 500
            args.gpu_id = 0
            args.cpu_threads = 10
            args.enable_mkldnn = False
            args.min_subgraph_size = 15
            args.det_limit_side_len = 960
            args.det_limit_type = "max"
            args.det_db_thresh = 0.3
            args.det_db_box_thresh = 0.6
            args.det_db_unclip_ratio = 1.5
            args.max_batch_size = 10
            args.use_dilation = False
            args.det_db_score_mode = "fast"
            args.det_box_type = "quad"
            args.benchmark = False

            return TextDetector(args)
        except Exception as e:
            logger.error("Failed to initialize text detector: %s", str(e))
            raise

    def transform(
        self,
        video_paths: pa.Array,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
    ) -> pa.Array:
        """检测视频中的水印区域，支持路径和二进制输入

        Args:
            video_paths: 视频文件路径列（本地、TOS、HTTP等），与video_binaries二选一
            video_binaries: 视频二进制数据列，与video_paths二选一
            video_formats: 视频格式字符串列，配合video_binaries使用

        Returns:
            包含水印检测结果的结构化列，每个元素包含以下字段：
                - watermark_regions: 检测到的水印区域列表
                - video_resolution: 视频分辨率[width, height]
                - total_frames: 视频总帧数
        """  # noqa: D415
        paths_list = video_paths.to_pylist()
        n = len(paths_list)

        binaries_list = video_binaries.to_pylist() if video_binaries is not None else [None] * n
        formats_list = video_formats.to_pylist() if video_formats is not None else [None] * n

        results = []

        for i, (video_path, video_binary, video_format) in enumerate(zip(paths_list, binaries_list, formats_list)):
            try:
                logger.info("Processing video %d/%d: %s", i + 1, n, video_path if video_path else "binary data")
                result = self._process_video(video_path, video_binary, video_format)
                results.append(result)
                logger.info("Completed video %d/%d", i + 1, n)
            except Exception as e:
                logger.error("Failed to process video %s: %s", video_path if video_path else "binary data", str(e))
                results.append({"watermark_regions": None, "video_resolution": None, "total_frames": None})

        return pa.array(results, type=self.__return_column_type__())

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
    ) -> dict[str, Any]:
        is_valid_video_path = not_blank(video_path)
        if not is_valid_video_path and video_binary is None:
            return {"watermark_regions": None, "video_resolution": None, "total_frames": None}

        if is_valid_video_path and video_path is not None:

            def process_with_path(local_video_path: str) -> dict[str, Any]:
                return self._detect_watermark_regions(local_video_path)

            return run_on_local_path(str(video_path), process_with_path)
        elif video_binary is not None:
            with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                video_extension = f".{video_format}" if video_format else ".mp4"
                temp_filepath = Path(temp_sub_dir) / f"temp_video_{uuid.uuid4().hex}{video_extension}"

                with temp_filepath.open("wb") as tmp:
                    tmp.write(video_binary)

                return self._detect_watermark_regions(str(temp_filepath))
        else:
            return {"watermark_regions": [], "video_resolution": [0, 0], "total_frames": 0}

    def _detect_watermark_regions(self, video_path: str) -> dict[str, Any]:
        cap = cv2.VideoCapture(video_path)

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info("Starting watermark detection: frames=%d, resolution=%dx%d", total_frames, width, height)

            sample_frames = self._smart_sample_frames(cap, total_frames)

            all_detections = []
            text_detector = self._get_text_detector()
            for frame in sample_frames:
                dt_boxes, _ = text_detector(frame)
                coordinates = self._get_coordinates(dt_boxes)
                all_detections.append(coordinates)

            watermark_regions = self._analyze_consistency(all_detections)

            logger.info("Detection completed: found %d watermark regions", len(watermark_regions))

            return {
                "watermark_regions": watermark_regions,
                "video_resolution": [width, height],
                "total_frames": total_frames,
            }

        finally:
            cap.release()

    def _smart_sample_frames(self, cap: Any, total_frames: int) -> list[Any]:
        sample_indices = []

        skip_ratio = 0.05
        skip_frames = max(int(total_frames * skip_ratio), 30)

        effective_start = skip_frames
        effective_end = total_frames - skip_frames
        effective_frames = effective_end - effective_start

        if effective_frames <= 0:
            effective_start = total_frames // 4
            effective_end = total_frames * 3 // 4
            effective_frames = effective_end - effective_start

        if effective_frames > 0:
            interval = max(1, effective_frames // self.sample_count)
            for i in range(self.sample_count):
                frame_idx = effective_start + i * interval
                if frame_idx < effective_end:
                    sample_indices.append(frame_idx)

        sample_indices = sample_indices[: self.sample_count]

        sample_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)

        return sample_frames

    def _get_coordinates(self, dt_box: Any) -> list[tuple[int, int, int, int]]:
        coordinate_list = []
        boxes_to_process = dt_box.tolist() if hasattr(dt_box, "tolist") else list(dt_box)

        for box in boxes_to_process:
            import numpy as np

            points = np.array(box)
            xmin, ymin = points[:, 0].min(), points[:, 1].min()
            xmax, ymax = points[:, 0].max(), points[:, 1].max()

            coordinate_list.append((int(ymin), int(ymax), int(xmin), int(xmax)))
        return coordinate_list

    def _analyze_consistency(self, all_detections: list[list[tuple[int, int, int, int]]]) -> list[dict[str, Any]]:
        region_stats: dict[tuple[int, int, int, int], dict[str, Any]] = {}

        for detection in all_detections:
            for box in detection:
                matched_key = None
                for existing_key in region_stats:
                    if self._are_regions_similar(box, existing_key, self.position_tolerance):
                        matched_key = existing_key
                        break

                if matched_key:
                    region_stats[matched_key]["count"] += 1
                    region_stats[matched_key]["boxes"].append(box)
                else:
                    region_stats[box] = {"count": 1, "boxes": [box]}

        watermark_regions = []
        total_samples = len(all_detections)

        for stats in region_stats.values():
            consistency_score = stats["count"] / total_samples

            if consistency_score >= self.consistency_threshold:
                avg_box = self._calculate_average_box(stats["boxes"])
                watermark_regions.append(
                    {
                        "ymin": int(avg_box[0]),
                        "ymax": int(avg_box[1]),
                        "xmin": int(avg_box[2]),
                        "xmax": int(avg_box[3]),
                        "confidence": round(consistency_score, 3),
                    }
                )

        return watermark_regions

    def _are_regions_similar(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int], tolerance: int
    ) -> bool:
        ymin1, ymax1, xmin1, xmax1 = box1
        ymin2, ymax2, xmin2, xmax2 = box2

        return (
            abs(ymin1 - ymin2) <= tolerance
            and abs(ymax1 - ymax2) <= tolerance
            and abs(xmin1 - xmin2) <= tolerance
            and abs(xmax1 - xmax2) <= tolerance
        )

    def _calculate_average_box(self, boxes: list[tuple[int, int, int, int]]) -> tuple[float, float, float, float]:
        if not boxes:
            return (0, 0, 0, 0)

        ymin_sum = sum(box[0] for box in boxes)
        ymax_sum = sum(box[1] for box in boxes)
        xmin_sum = sum(box[2] for box in boxes)
        xmax_sum = sum(box[3] for box in boxes)

        count = len(boxes)
        return (ymin_sum / count, ymax_sum / count, xmin_sum / count, xmax_sum / count)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field(
                "watermark_regions",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("ymin", pa.int32()),
                            pa.field("ymax", pa.int32()),
                            pa.field("xmin", pa.int32()),
                            pa.field("xmax", pa.int32()),
                            pa.field("confidence", pa.float32()),
                        ]
                    )
                ),
            ),
            pa.field("video_resolution", pa.list_(pa.int32())),
            pa.field("total_frames", pa.int64()),
        ]
        return pa.struct(fields)
