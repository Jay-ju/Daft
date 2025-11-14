# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np  # noqa: TID253
from cv2 import CAP_PROP_POS_MSEC

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.functions.utils.video_utils import decode_video
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)

EXTRACT_KEYFRAMES_FEATURE_METHOD = {"difference", "optical_flow", "histogram"}
EXTRACT_KEYFRAMES_INTRA_METHOD = {"I_frame"}


class VideoKeyframes(Operator):
    """**视频关键帧抽取处理器，支持多算法动态检测。**

    **核心功能：**
    - 多算法支持：
        - 像素差分法(difference)
        - 光流法(optical_flow)
        - 直方图法(histogram)
        - I型关键帧标识(I_frame)
    - 支持自定义阈值与数量控制
    - 支持均匀抽取关键帧（by_count_uniform=True 时，keyframes_cnt 为均匀采样数量；False 时为顺序抽取前 N 帧）
    - 提供时间戳定位功能
    - 支持多种输出格式与存储选项

    **格式支持：**
    - 输入：MP4, AVI, MOV 等常见视频格式
    - 输出：JPG, PNG 图片格式

    **性能建议：**
    - I_frame 方法效率最高，推荐优先使用
    """  # noqa: D415

    def __init__(
        self,
        method: str = "I_frame",
        img_type: str = ".jpg",
        threshold: float = 0,
        keyframes_cnt: int = 10,
        seconds_per_frame: int = -1,
        output_tos_dir: str = "",
        by_count_uniform: bool = False,
        return_keyframes: bool = True,
        return_base64: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化视频关键帧抽取算子。

        Args:
            method: 抽取关键帧的方法，支持 "difference"（像素差分法）、"optical_flow"（光流法）、"histogram"（直方图法）、"I_frame"（I型关键帧标识）。
                可选值：["difference", "optical_flow", "histogram", "I_frame"]
                默认值："I_frame"
            img_type: 输出关键帧图片格式，支持 ".jpg"、".png"。
                可选值：[".jpg", ".png"]
                默认值：".jpg"
            threshold: 用于判断关键帧的阈值。difference 推荐 2000000，histogram 推荐 0.01，optical_flow 推荐 2.0。
                默认值：0
            keyframes_cnt: 指定抽取关键帧的数量，-1 表示不限制数量。
                默认值：10
            seconds_per_frame: 抽帧间隔，单位为秒，-1 表示不指定间隔。
                默认值：-1
            output_tos_dir: 保存关键帧图片到 TOS 的目标路径，若为空字符串则不上传。
                默认值：""
            by_count_uniform: 是否均匀采样关键帧数量（仅 I_frame 方法有效）。True 时 keyframes_cnt 为均匀采样数量，False 时为顺序抽取前 N 帧。
                默认值：False
            return_keyframes: 是否返回关键帧图片的 array 数据（大内存，建议只在需要时开启）。
                默认值：True
            return_base64: 是否返回关键帧图片的 base64 编码（大内存，建议只在需要时开启）。
                默认值：True
            **kwargs: 其他参数，透传给父类。
        """  # noqa: D415
        super().__init__(**kwargs)
        self.method = method
        self.img_type = img_type
        self.threshold = threshold if threshold is not None else 0
        self.keyframes_cnt = keyframes_cnt
        self.seconds_per_frame = seconds_per_frame
        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.by_count_uniform = by_count_uniform
        self.return_keyframes = return_keyframes
        self.return_base64 = return_base64

        logger.info("The extract keyframe method: %s", self.method)
        logger.info("The threshold of extracting keyframe : %s", self.threshold)
        logger.info("The count of extracting keyframe : %s", self.keyframes_cnt)
        logger.info("The seconds per frame : %s", self.seconds_per_frame)
        logger.info("The by_count_uniform : %s", self.by_count_uniform)
        logger.info("The return_keyframes : %s", self.return_keyframes)
        logger.info("The return_base64 : %s", self.return_base64)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="cv2")

    def extract_keyframes_by_img_feature(
        self, video_path: str, local_output_path: str, tos_output_dir: str | None
    ) -> tuple[list[np.ndarray], list[str], list[float], list[str]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 1
        logger.info("frame interval: %s", frame_interval)
        frame_count = 0

        ret, prev_frame = cap.read()
        if not ret:
            logger.info("Failed to read the video.")
            return [], [], [], []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        keyframe_index = 0
        frame_index = 0
        prev_keyframe_index = 0
        if self.method == "histogram":
            prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
        else:
            prev_hist = None

        keyframe_array_list: list[np.ndarray] = []
        keyframe_base64_list: list[str] = []
        keyframe_timestamps_list: list[float] = []
        keyframe_tos_paths_list: list[str] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if self.seconds_per_frame > 0 and frame_count % frame_interval % int(self.seconds_per_frame) != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            distance: float
            if self.method == "difference":
                frame_diff = cv2.absdiff(prev_gray, gray)
                distance = float(np.sum(frame_diff))
            elif self.method == "optical_flow":
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                distance = float(np.mean(magnitude))
            elif self.method == "histogram":
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                distance = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                prev_hist = hist
            else:
                raise ValueError(
                    f"Keep strategy [{self.method}] is not supported. "
                    f"Can only be one of [difference, optical_flow, histogram]."
                )

            if distance > self.threshold:
                if prev_keyframe_index != frame_index - 1:
                    if local_output_path and tos_output_dir:
                        keyframe_filename = f"{local_output_path}/" f"keyframe_{keyframe_index:04d}{self.img_type}"
                        cv2.imwrite(keyframe_filename, frame)
                        tos_path = f"{tos_output_dir}/{Path(keyframe_filename).name}"
                        upload_file(str(keyframe_filename), tos_path)
                        keyframe_tos_paths_list.append(tos_path)
                        logger.info("The keyframe output tos path: %s", tos_output_dir)

                    _, buffer = cv2.imencode(self.img_type, frame)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    keyframe_array_list.append(frame)
                    keyframe_base64_list.append(frame_base64)
                    timestamp = cap.get(CAP_PROP_POS_MSEC) / 1000  # Convert ms to seconds
                    keyframe_timestamps_list.append(timestamp)
                    keyframe_index += 1
                prev_keyframe_index = frame_index

            prev_gray = gray
            frame_index += 1
            if 0 < self.keyframes_cnt <= keyframe_index:
                break

        shutil.rmtree(local_output_path)
        cap.release()
        return keyframe_array_list, keyframe_base64_list, keyframe_timestamps_list, keyframe_tos_paths_list

    def _extract_keyframes_by_i_frame(
        self, input_video: str, local_output_path: str, tos_output_dir: str | None
    ) -> tuple[list[np.ndarray], list[str], list[float], list[str]]:
        container = decode_video(input_video)
        input_video_stream = container.streams.video[0]
        ori_skip_method = input_video_stream.codec_context.skip_frame
        input_video_stream.codec_context.skip_frame = "NONKEY"

        keyframe_array_list: list[np.ndarray] = []
        keyframe_base64_list: list[str] = []
        keyframe_timestamps_list: list[float] = []
        keyframe_tos_paths_list: list[str] = []

        container.seek(0)
        if self.by_count_uniform and self.keyframes_cnt > 0:
            # 收集所有 I 帧
            all_frames = []
            for frame in container.decode(input_video_stream):
                img = frame.to_ndarray(format="bgr24")
                timestamp = float(frame.pts * frame.time_base)
                all_frames.append((img, frame, timestamp))
            total = len(all_frames)
            if total == 0:
                input_video_stream.codec_context.skip_frame = ori_skip_method
                shutil.rmtree(local_output_path)
                container.close()
                return [], [], [], []
            # 均匀采样
            indices = np.linspace(0, total - 1, min(self.keyframes_cnt, total), dtype=int)
            for idx, i in enumerate(indices):
                img, frame, timestamp = all_frames[i]
                if local_output_path and tos_output_dir:
                    img_filename = f"{local_output_path}/keyframe_{idx:04d}{self.img_type}"
                    cv2.imwrite(img_filename, img)
                    logger.info("Saved %s", img_filename)
                    tos_path = f"{tos_output_dir}/{Path(img_filename).name}"
                    upload_file(str(img_filename), tos_path)
                    keyframe_tos_paths_list.append(tos_path)
                    logger.info("The keyframe output tos path: %s", tos_output_dir)
                if self.return_keyframes:
                    keyframe_array_list.append(img)
                if self.return_base64:
                    _, buffer = cv2.imencode(self.img_type, img)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    keyframe_base64_list.append(frame_base64)
                keyframe_timestamps_list.append(timestamp)
        else:
            # 顺序取前 n 个 I 帧
            keyframe_count = 0
            for frame in container.decode(input_video_stream):
                img = frame.to_ndarray(format="bgr24")
                if local_output_path and tos_output_dir:
                    img_filename = f"{local_output_path}/keyframe_{keyframe_count:04d}{self.img_type}"
                    cv2.imwrite(img_filename, img)
                    logger.info("Saved %s", img_filename)
                    tos_path = f"{tos_output_dir}/{Path(img_filename).name}"
                    upload_file(str(img_filename), tos_path)
                    keyframe_tos_paths_list.append(tos_path)
                    logger.info("The keyframe output tos path: %s", tos_output_dir)
                if self.return_keyframes:
                    keyframe_array_list.append(img)
                if self.return_base64:
                    _, buffer = cv2.imencode(self.img_type, img)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    keyframe_base64_list.append(frame_base64)
                timestamp = float(frame.pts * frame.time_base)
                keyframe_timestamps_list.append(timestamp)
                keyframe_count += 1
                if 0 < self.keyframes_cnt <= keyframe_count:
                    break

        input_video_stream.codec_context.skip_frame = ori_skip_method

        # Check if we have no keyframes by checking timestamps list (which is always populated)
        if len(keyframe_timestamps_list) == 0:
            container.seek(0)
            for frame in container.decode(input_video_stream):
                img = frame.to_ndarray(format="bgr24")
                if local_output_path and tos_output_dir:
                    img_filename = f"{local_output_path}/keyframe_0000{self.img_type}"
                    cv2.imwrite(img_filename, img)
                    logger.info("Saved %s", img_filename)
                    tos_path = f"{tos_output_dir}/{Path(img_filename).name}"
                    upload_file(str(img_filename), tos_path)
                    keyframe_tos_paths_list.append(tos_path)
                    logger.info("The keyframe output tos path: %s", tos_output_dir)
                if self.return_keyframes:
                    keyframe_array_list.append(img)
                if self.return_base64:
                    _, buffer = cv2.imencode(self.img_type, img)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    keyframe_base64_list.append(frame_base64)
                keyframe_timestamps_list.append(float(frame.pts * frame.time_base))
                break

        shutil.rmtree(local_output_path)
        container.close()

        # Ensure we return proper lists even when flags are False
        # keyframe_array_list/keyframe_base64_list 已经初始化为 list，不需要再判断 None

        return keyframe_array_list, keyframe_base64_list, keyframe_timestamps_list, keyframe_tos_paths_list

    def _extract_keyframes(
        self, video_path: str, local_output_dir: str, tos_output_dir: str | None
    ) -> tuple[list[np.ndarray], list[str], list[float], list[str]]:
        if self.method in EXTRACT_KEYFRAMES_FEATURE_METHOD:
            return self.extract_keyframes_by_img_feature(video_path, local_output_dir, tos_output_dir)
        if self.method in EXTRACT_KEYFRAMES_INTRA_METHOD:
            return self._extract_keyframes_by_i_frame(video_path, local_output_dir, tos_output_dir)
        return [], [], [], []

    def _prepare_output_dirs(self, video: str) -> str | None:
        if not self.output_tos_dir or not self.output_tos_dir.strip():
            return None

        video_sub_dir = Path(video).name.split(".")[0]
        tos_output_dir = f"{self.output_tos_dir}/{video_sub_dir}"
        logger.info("Video keyframes tos output dir: %s", tos_output_dir)
        mkdirs(tos_output_dir)
        return tos_output_dir

    def _process_video(
        self,
        video: str | None,
        video_binary: bytes | None,
        video_format: str | None,
    ) -> tuple[list[np.ndarray], list[str], list[float], list[str]]:
        try:
            if video is None and video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = f".{video_format.lower()}" if video_format else ".mp4"
                    temp_filename = f"binary_input_{uuid.uuid4().hex}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    video_name = "binary_input"
                    local_output_dir = Path(temp_dir) / Path(video_name)
                    Path(local_output_dir).mkdir(exist_ok=True)

                    tos_output_dir = self._prepare_output_dirs(f"binary_{uuid.uuid4().hex}")
                    return self._extract_keyframes(str(temp_filepath), str(local_output_dir), tos_output_dir)
            else:
                assert video is not None
                tos_output_dir = self._prepare_output_dirs(video)

                def process_video(local_path: str) -> tuple[list[np.ndarray], list[str], list[float], list[str]]:
                    video_path = Path(local_path)
                    video_name = video_path.stem
                    local_output_dir = video_path.parent.joinpath(video_name)
                    local_output_dir.mkdir(exist_ok=True)
                    return self._extract_keyframes(local_path, str(local_output_dir), tos_output_dir)

                return run_on_local_path(video, process_video)
        except Exception:
            logger.exception("Failed to extract keyframes from video %s", video)
            return [], [], [], []

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
    ) -> pa.Array:
        """批量抽取视频关键帧，支持多种输入类型。

        支持三种输入方式：视频路径、视频二进制、视频格式。输出结构包含关键帧 array、base64、时间戳、TOS 路径等。
        注意：`video_paths` 和 `video_binaries` 至少需要指定一个，否则返回空结果。

        Args:
            video_paths: 输入视频路径列，类型为数组。
                默认值：None
            video_binaries: 输入视频二进制数据列，类型为数组。
                默认值：None
            video_formats: 输入视频格式列（如 'mp4'、'avi' 等），类型为数组。
                默认值：None

        Returns:
            pa.Array: 处理后的数组，结构体字段包括：
                - keyframes: list[list[list[list[int]]]]，关键帧图片的 array 格式
                - base64: list[str]，关键帧图片的 base64 编码
                - timestamps: list[float]，关键帧对应的时间戳（单位：秒）
                - tos_paths: list[str]，关键帧在 TOS 上的存储路径
        """  # noqa: D415
        n = 0
        if video_paths is not None:
            n = len(video_paths)
        elif video_binaries is not None:
            n = len(video_binaries)
        elif video_formats is not None:
            n = len(video_formats)
        else:
            return pa.array([], type=self.__return_column_type__())

        paths_list = video_paths.to_pylist() if video_paths is not None else [None] * n
        binaries_list = video_binaries.to_pylist() if video_binaries is not None else [None] * n
        formats_list = video_formats.to_pylist() if video_formats is not None else [None] * n

        results = []
        for path, binary, fmt in zip(paths_list, binaries_list, formats_list):
            keyframes, base64s, timestamps, tos_paths = self._process_video(path, binary, fmt)

            if self.return_keyframes and keyframes:
                keyframes_list = [k.tolist() for k in keyframes]
            else:
                keyframes_list = []

            if self.return_base64 and base64s:
                base64s_list = base64s
            else:
                base64s_list = []

            result = {
                "keyframes": keyframes_list,
                "base64": base64s_list,
                "timestamps": timestamps,
                "tos_paths": tos_paths,
            }
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = []
        fields.append(pa.field("keyframes", pa.list_(pa.list_(pa.list_(pa.list_(pa.int64()))))))
        fields.append(pa.field("base64", pa.list_(pa.string())))
        fields.append(pa.field("timestamps", pa.list_(pa.float64())))
        fields.append(pa.field("tos_paths", pa.list_(pa.string())))
        return pa.struct(fields)
