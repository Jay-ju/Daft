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
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)

SAMPLE_MODES = {
    "by_count_uniform",
    "by_interval_time",
    "by_interval_frames",
    "by_fps",
    "by_timestamps",
}


class VideoFrameSampler(Operator):
    """从视频中采样图像帧处理器，支持多种采样模式及时间范围控制。

    核心功能:
    - 多种采样方式：
      - by_count_uniform：在时间范围内均匀采K帧
      - by_interval_time：按时间间隔Δt秒采样
      - by_interval_frames：按解码帧间隔N采样
      - by_fps：以目标fps采样
      - by_timestamps：按给定时间戳列表采样
    - 支持起止时间范围限制与是否包含尾帧
    - 输出原始帧（array）、base64 编码、时间戳、帧索引与可选的TOS存储路径
    - 路径输入或二进制输入两种来源，兼容远端URI（通过 run_on_local_path）

    """  # noqa: D415

    def __init__(
        self,
        sample_mode: str = "by_count_uniform",
        start_time_sec: float = 0.0,
        end_time_sec: float | None = None,
        count_k: int | None = None,
        interval_sec: float | None = None,
        interval_frames: int | None = None,
        target_fps: float | None = None,
        timestamps_sec: list[float] | None = None,
        img_type: str = ".jpg",
        output_tos_dir: str = "",
        max_frames: int | None = None,
        seed: int = 42,
        output_frames: bool = True,
        output_base64: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化视频关键帧抽取算子

        Args:
            sample_mode: 采样模式
                可选值: ["by_count_uniform", "by_interval_time", "by_interval_frames", "by_fps", "by_timestamps"]
            start_time_sec: 起始时间（秒）
            end_time_sec: 结束时间（秒），None 表示视频末尾

            count_k: 均匀/随机采样的帧数（by_count_uniform使用）
            interval_sec: 时间间隔Δt（秒，by_interval_time使用）
            interval_frames: 解码帧间隔N（by_interval_frames使用）
            target_fps: 目标FPS（by_fps使用）
            timestamps_sec: 目标时间戳列表（秒，by_timestamps使用）
            img_type: 输出图片格式（用于base64与可选TOS落盘），可选[".jpg", ".png", ".webp"]，默认 ".jpg"
            output_tos_dir: 若非空则将采样帧落盘本地并上传至 TOS 的目标目录（目录下每个视频单独子目录）
            max_frames: 返回帧上限（防御性限制），None 表示不限制
            seed: 当无法提前获知视频总时长时，算子会采用 reservoir sampling（水塘抽样）算法，从视频流中均匀随机采样指定数量的帧。此参数用于设置随机数种子，保证采样结果可复现，默认值为 42。
            output_frames: 是否输出原始帧数组（较大，默认 True）。注意：即使设置为 False，返回结果仍包含 frames 字段，但内容为空列表。
            output_base64: 是否输出 base64 编码（较大，默认 True）。注意：即使设置为 False，返回结果仍包含 base64 字段，但内容为空列表。
            **kwargs: 透传给基类
        """  # noqa: D415
        super().__init__(**kwargs)
        if sample_mode not in SAMPLE_MODES:
            raise ValueError(f"Unsupported sample_mode={sample_mode}, must be one of {sorted(SAMPLE_MODES)}")
        self.sample_mode = sample_mode

        self.start_time_sec = max(0.0, float(start_time_sec or 0.0))
        self.end_time_sec = float(end_time_sec) if end_time_sec is not None else None

        self.count_k = int(count_k) if count_k is not None else None
        self.interval_sec = float(interval_sec) if interval_sec is not None else None
        self.interval_frames = int(interval_frames) if interval_frames is not None else None
        self.target_fps = float(target_fps) if target_fps is not None else None
        self.timestamps_sec = sorted(timestamps_sec) if timestamps_sec else None

        self.img_type = img_type if img_type in [".jpg", ".png", ".webp"] else ".jpg"
        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.seed = int(seed)
        self.output_frames = bool(output_frames)
        self.output_base64 = bool(output_base64)

        logger.info("VideoFrameSampler init: mode=%s", self.sample_mode)
        logger.info("time range: start=%.3f, end=%s", self.start_time_sec, self.end_time_sec)
        logger.info(
            "params: count_k=%s interval_sec=%s interval_frames=%s target_fps=%s",
            self.count_k,
            self.interval_sec,
            self.interval_frames,
            self.target_fps,
        )
        if self.timestamps_sec:
            logger.info(
                "timestamps size: %d [%.3f ... %.3f]",
                len(self.timestamps_sec),
                self.timestamps_sec[0],
                self.timestamps_sec[-1],
            )

        # 仅追踪算子调用，无需指定 model_service_or_lib（未用具体模型）
        tracking_usage(op=self.__class__.__name__)

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        video_durations: pa.Array | None = None,
    ) -> pa.Array:
        """批量采样视频帧，支持路径、二进制输入和视频时长辅助

        video_paths 和 video_binaries 至少传入一个

        Args:
            video_paths: 输入视频路径列（本地或远程URI），类型 string，默认 None

            video_binaries: 输入视频二进制列，类型 binary，默认 None
            video_formats: 对应二进制输入的格式（如"mp4","mov"），类型 string，默认 None
            video_durations: 视频时长列（秒），类型 float，默认 None

        Returns:
            pa.Array: 结构体数组，字段包括：
                - frames: 采样帧图像的 array（H x W x C），可配置是否输出
                - base64: 采样帧对应的 base64 编码（img_type 指定格式），可配置是否输出
                - timestamps: 采样帧的时间戳（秒）
                - frame_indices: 采样帧在解码序列中的帧索引
                - tos_paths: 采样帧在 TOS 上的存储路径（若配置了 output_tos_dir）
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
        durations_list = video_durations.to_pylist() if video_durations is not None else [None] * n

        results = []
        for path, binary, fmt, duration in zip(paths_list, binaries_list, formats_list, durations_list):
            frames, base64s, timestamps, frame_indices = self._process_video(path, binary, fmt, duration)

            if path is not None:
                tos_output_dir = self._prepare_output_dirs(path)
            else:
                tos_output_dir = None

            if tos_output_dir and frames:
                tos_paths = [f"{tos_output_dir}/frame_{i:04d}{self.img_type}" for i in range(len(frames))]
            else:
                tos_paths = []

            frames_list = [f.tolist() for f in frames] if (self.output_frames and frames) else []
            base64s_out = base64s if self.output_base64 else []

            result = {
                "frames": frames_list,
                "base64": base64s_out,
                "timestamps": timestamps,
                "frame_indices": frame_indices,
                "tos_paths": tos_paths,
            }
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("frames", pa.list_(pa.list_(pa.list_(pa.list_(pa.int64()))))),
            pa.field("base64", pa.list_(pa.string())),
            pa.field("timestamps", pa.list_(pa.float64())),
            pa.field("frame_indices", pa.list_(pa.int64())),
            pa.field("tos_paths", pa.list_(pa.string())),
        ]
        return pa.struct(fields)

    def _prepare_output_dirs(self, video_path: str) -> str | None:
        if not self.output_tos_dir or not self.output_tos_dir.strip():
            return None
        video_sub_dir = Path(video_path).name.split(".")[0]
        tos_output_dir = f"{self.output_tos_dir}/{video_sub_dir}"
        logger.info("Video frames tos output dir: %s", tos_output_dir)
        mkdirs(tos_output_dir)
        return tos_output_dir

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        duration: float | None = None,
    ) -> tuple[list[np.ndarray], list[str], list[float], list[int]]:
        try:
            if video_path is None and video_binary is not None:
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
                    return self._sample_frames(str(temp_filepath), str(local_output_dir), tos_output_dir, duration)
            else:
                assert video_path is not None
                tos_output_dir = self._prepare_output_dirs(video_path)

                def process_video(local_path: str) -> tuple[list[np.ndarray], list[str], list[float], list[int]]:
                    video_path_obj = Path(local_path)
                    video_name = video_path_obj.stem
                    local_output_dir = video_path_obj.parent.joinpath(video_name)
                    local_output_dir.mkdir(exist_ok=True)
                    return self._sample_frames(local_path, str(local_output_dir), tos_output_dir, duration)

                return run_on_local_path(video_path, process_video)
        except Exception:
            logger.exception("Failed to sample frames from video %s", video_path)
            return [], [], [], []

    def _sample_frames(
        self, video_path: str, local_output_dir: str, tos_output_dir: str | None, duration: float | None = None
    ) -> tuple[list[np.ndarray], list[str], list[float], list[int]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Open video failed: %s", video_path)
            shutil.rmtree(local_output_dir, ignore_errors=True)
            return [], [], [], []

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_meta = (total_frames_meta / fps) if (fps > 0 and total_frames_meta > 0) else None

            # Prefer externally provided duration if available
            duration_used = float(duration) if duration is not None and duration > 0 else duration_meta

            start = max(0.0, self.start_time_sec)
            end = self.end_time_sec if self.end_time_sec is not None else duration_used

            # Build target timestamps for different modes (when applicable)
            targets: list[float] | None = None
            mode = self.sample_mode

            if mode == "by_timestamps":
                ts = self.timestamps_sec or []
                # Filter timestamps within the time range, always include end_time_sec if present
                targets = [t for t in ts if (t >= start) and (end is None or t <= end)]
            elif mode in ("by_interval_time", "by_fps", "by_count_uniform"):
                if mode == "by_interval_time":
                    if self.interval_sec is None or self.interval_sec <= 0:
                        logger.warning("interval_sec must be positive for by_interval_time")
                        shutil.rmtree(local_output_dir, ignore_errors=True)
                        cap.release()
                        return [], [], [], []
                    step = self.interval_sec
                elif mode == "by_fps":
                    if self.target_fps is None or self.target_fps <= 0:
                        logger.warning("target_fps must be positive for by_fps")
                        shutil.rmtree(local_output_dir, ignore_errors=True)
                        cap.release()
                        return [], [], [], []
                    step = 1.0 / self.target_fps
                else:  # by_count_uniform
                    if self.count_k is None or self.count_k <= 0:
                        logger.warning("count_k must be positive for by_count_uniform")
                        shutil.rmtree(local_output_dir, ignore_errors=True)
                        cap.release()
                        return [], [], [], []

                    if end is None:
                        # 时长未知，降级为解码流上的水塘抽样（见后续分支）
                        targets = None
                    else:
                        if self.count_k == 1:
                            targets = [start]
                        else:
                            total_span = max(0.0, end - start)
                            if total_span == 0.0:
                                targets = [start]
                            else:
                                step = total_span / (self.count_k - 1)
                                last = end
                                targets = [start + i * step for i in range(self.count_k - 1)]
                                targets.append(last)

                if mode in ("by_interval_time", "by_fps"):
                    if end is None:
                        # Unknown duration: cannot precompute targets; degrade to sequential decoding
                        targets = None
                    else:
                        cur = start
                        targets = []
                        while True:
                            if end is not None:
                                # Always include end_time_sec (closed interval)
                                if cur > end + 1e-9:
                                    break
                            targets.append(cur)
                            cur += step

            # by_interval_frames does not use targets; sample by decoded sequence

            keyframe_array_list: list[np.ndarray] = []
            base64_list: list[str] = []
            ts_list: list[float] = []
            idx_list: list[int] = []

            # Unified frame write + encoding logic
            def emit(frame_bgr: np.ndarray, sample_idx: int, timestamp_sec: float, frame_idx: int) -> None:
                # 根据配置决定是否输出 frames/base64
                if tos_output_dir and local_output_dir:
                    out_name = f"{local_output_dir}/frame_{sample_idx:04d}{self.img_type}"
                    cv2.imwrite(out_name, frame_bgr)
                    upload_file(str(out_name), f"{tos_output_dir}/{Path(out_name).name}")
                    logger.info("Uploaded to TOS: %s", f"{tos_output_dir}/{Path(out_name).name}")
                if self.output_base64:
                    ok, buffer = cv2.imencode(self.img_type, frame_bgr)
                    if ok:
                        b64 = base64.b64encode(buffer).decode("utf-8")
                    else:
                        b64 = ""
                    base64_list.append(b64)
                if self.output_frames:
                    keyframe_array_list.append(frame_bgr)
                ts_list.append(timestamp_sec)
                idx_list.append(frame_idx)

            # Sampling execution
            if mode == "by_interval_frames":
                if self.interval_frames is None or self.interval_frames <= 0:
                    logger.warning("interval_frames must be positive for by_interval_frames")
                    shutil.rmtree(local_output_dir, ignore_errors=True)
                    cap.release()
                    return [], [], [], []

                # Align start (seek to frame near start time if possible)
                if fps > 0:
                    start_frame_guess = int(max(0, round(start * fps)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)

                sample_idx = 0
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # position before reading
                picked_interval = self.interval_frames
                emitted = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx_now = frame_idx
                    frame_idx += 1

                    cur_ts = cap.get(CAP_PROP_POS_MSEC) / 1000.0
                    if cur_ts < start - 1e-9:
                        continue
                    if (self.end_time_sec is not None) and (cur_ts > self.end_time_sec + 1e-9):
                        break

                    # Take one every N frames using decode counter
                    if (frame_idx_now - int(start * fps if fps > 0 else 0)) % picked_interval == 0:
                        emit(frame, sample_idx, cur_ts, frame_idx_now)
                        sample_idx += 1
                        emitted += 1
                        if self.max_frames is not None and emitted >= self.max_frames:
                            break

            elif targets is not None:
                # Seek to specific timestamps for sampling (approximate)
                sample_idx = 0
                emitted = 0
                for t in targets:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    cur_ts = cap.get(CAP_PROP_POS_MSEC) / 1000.0
                    frame_idx_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 读完后位于下一帧，估一个当前索引
                    emit(frame, sample_idx, cur_ts, frame_idx_now)
                    sample_idx += 1
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break

            else:
                # Degraded path when targets cannot be constructed: sequential decode + time-based sampling
                # - by_interval_time/by_fps: trigger by time roll
                # - by_count_uniform: reservoir sampling (uniform over time range)
                rng = np.random.RandomState(self.seed)
                sample_idx = 0
                emitted = 0

                # Prepare time-based sampling step and next trigger time when applicable
                step_val: float = 0.0
                next_time_val: float = start
                if self.sample_mode in ("by_interval_time", "by_fps"):
                    if self.sample_mode == "by_interval_time":
                        if self.interval_sec is None or self.interval_sec <= 0:
                            logger.warning("Invalid step in degraded time-based sampling")
                            shutil.rmtree(local_output_dir, ignore_errors=True)
                            cap.release()
                            return [], [], [], []
                        step_val = float(self.interval_sec)
                    else:
                        if self.target_fps is None or self.target_fps <= 0:
                            logger.warning("Invalid step in degraded time-based sampling")
                            shutil.rmtree(local_output_dir, ignore_errors=True)
                            cap.release()
                            return [], [], [], []
                        step_val = 1.0 / float(self.target_fps)

                # Reservoir sampling buffer
                reservoir_frames: list[tuple[np.ndarray, float, int]] = []
                seen = 0
                K = self.count_k if (self.count_k and self.count_k > 0) else None

                # Align start
                if fps > 0:
                    start_frame_guess = int(max(0, round(start * fps)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cur_ts = cap.get(CAP_PROP_POS_MSEC) / 1000.0
                    cur_idx = frame_idx
                    frame_idx += 1

                    if cur_ts < start - 1e-9:
                        continue
                    if (self.end_time_sec is not None) and (cur_ts > self.end_time_sec + 1e-9):
                        break

                    if self.sample_mode in ("by_interval_time", "by_fps"):
                        # Time-triggered
                        if cur_ts + 1e-9 >= next_time_val:
                            emit(frame, sample_idx, cur_ts, cur_idx)
                            sample_idx += 1
                            emitted += 1
                            next_time_val += step_val
                            if self.max_frames is not None and emitted >= self.max_frames:
                                break
                    else:
                        # Reservoir sampling for by_count_uniform (uniform over time range)
                        if K is None:
                            logger.warning("count_k missing for by_count_uniform in degraded path")
                            break
                        seen += 1
                        if len(reservoir_frames) < K:
                            reservoir_frames.append((frame.copy(), cur_ts, cur_idx))
                        else:
                            j = rng.randint(0, seen)
                            if j < K:
                                reservoir_frames[j] = (frame.copy(), cur_ts, cur_idx)

                if self.sample_mode == "by_count_uniform" and K:
                    # Emit frames from reservoir in ascending timestamp order
                    reservoir_frames.sort(key=lambda x: x[1])
                    for f, t, iidx in reservoir_frames:
                        emit(f, sample_idx, t, iidx)
                        sample_idx += 1
                        emitted += 1
                        if self.max_frames is not None and emitted >= self.max_frames:
                            break

            shutil.rmtree(local_output_dir, ignore_errors=True)
            cap.release()
            return keyframe_array_list, base64_list, ts_list, idx_list

        except Exception:
            logger.exception("Exception while sampling frames for video: %s", video_path)
            try:
                shutil.rmtree(local_output_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            return [], [], [], []
