# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import cv2

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class VideoSttnInpaint(Operator):
    """**视频区域修复**

    **核心功能**
    - 智能修复：基于STTN时空记忆网络进行视频内容修复（去除水印、字幕等外来内容），支持设置默认修复区域
    - 多区域支持：支持同时修复一个、多个指定区域或全屏处理
    - 音频保留：视频修复不影响原始音频
    - GPU加速：支持CUDA加速提升处理效率
    - 多输入支持：支持路径输入和二进制输入
    - 适用性强：支持音视频流不对齐场景

    **推荐实践**
    - 建议处理分辨率不超过1080p的视频
    - 修复区域越小处理速度越快
    - 长视频建议预先分段处理

    **技术特性**
    - 时空一致性：保证修复区域的时间连续性
    - 格式兼容：输出标准MP4格式视频
    """  # noqa: D415

    def __init__(
        self,
        output_tos_dir: str,
        model_path: str = "/opt/las/models",
        model_name: str = "researchmm/STTN",
        neighbor_stride: int = 5,
        reference_length: int = 10,
        max_load_num: int = 50,
        is_keep_audio: bool = True,
        is_set_default_inpaint_areas: bool = False,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化STTN视频修复算子

        Args:
            output_tos_dir: TOS输出目录（必需）
                修复后视频的TOS存储路径
                格式："tos://bucket/path"
            model_path: 模型存储路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："researchmm/STTN"
            neighbor_stride: 相邻帧步长
                选择参考帧的间隔距离
                较小值(3-5)适合慢动作视频，较大值(8-10)适合快动作视频
                默认值：5
            reference_length: 参考帧数量
                用于修复当前帧的参考帧总数
                提供时序上下文信息，值越大修复质量越好但计算量增加
                默认值：10
            max_load_num: 最大加载帧数
                每次处理时最多同时加载到内存的视频帧数量
                控制内存使用，避免长视频导致内存溢出
                约束：max_load_num >= reference_length * neighbor_stride
                默认值：50
            is_keep_audio: 是否保留音频
                修复视频时是否保留原始音频
                默认值：True
            is_set_default_inpaint_areas: 是否设置默认修复区域
                是否根据输入参数默认设置修复区域为画面最下方30%
                默认值：False
            rank: GPU设备编号
                指定使用的GPU设备ID（多卡环境生效）
                None表示自动选择可用GPU
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)

        if not output_tos_dir:
            raise ValueError("output_tos_dir is required and cannot be empty")
        if not (isinstance(output_tos_dir, str) and output_tos_dir.startswith("tos://")):
            raise ValueError(f"Invalid output_tos_dir: {output_tos_dir!r}, should start with 'tos://'")

        self.model_dir = str(Path(model_path) / model_name)
        self.output_tos_dir = output_tos_dir
        mkdirs(self.output_tos_dir)

        self.neighbor_stride = neighbor_stride
        self.reference_length = reference_length
        self.max_load_num = max(max_load_num, reference_length * neighbor_stride)
        self.rank = rank
        self.is_keep_audio = is_keep_audio
        self.is_set_default_inpaint_areas = is_set_default_inpaint_areas

        self._sttn_model = None

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="sttn")

    def _create_mask(self, mask_size: tuple[int, int], coordinates: list[tuple[int, int, int, int]]) -> Any:
        import numpy as np

        mask = np.zeros(mask_size, dtype=np.uint8)
        for xmin, xmax, ymin, ymax in coordinates:
            mask[ymin:ymax, xmin:xmax] = 255
        return mask

    def _parse_inpaint_areas(self, inpaint_areas: list[dict[str, Any]] | None) -> list[tuple[int, int, int, int]]:
        if inpaint_areas is None:
            return []

        coordinates = []
        for area in inpaint_areas:
            if isinstance(area, dict) and all(k in area for k in ["ymin", "ymax", "xmin", "xmax"]):
                ymin = int(area["ymin"])
                ymax = int(area["ymax"])
                xmin = int(area["xmin"])
                xmax = int(area["xmax"])
                coordinates.append((xmin, xmax, ymin, ymax))
            else:
                logger.warning("Invalid area structure: %s", area)

        return coordinates

    def _load_sttn_model(self) -> Any:
        if self._sttn_model is not None:
            return self._sttn_model

        try:
            # 模型目录包含Python代码文件，需要添加到sys.path以便导入
            if self.model_dir not in sys.path:
                sys.path.append(self.model_dir)

            from inpaint.sttn_inpaint import STTNInpaint

            model_pth_path = str(Path(self.model_dir) / "models" / "sttn" / "infer_model.pth")

            if self.use_gpu:
                rank = 0 if self.rank is None else self.rank
                self.rank = rank % self.cuda_device_count
                device_str = f"cuda:{self.rank}"
            else:
                device_str = "cpu"

            self._sttn_model = STTNInpaint(
                model_path=model_pth_path,
                neighbor_stride=self.neighbor_stride,
                reference_length=self.reference_length,
                device=device_str,
            )
            return self._sttn_model
        except Exception as e:
            logger.error("Failed to load STTN model from %s: %s", self.model_dir, str(e))
            raise

    def _get_video_duration(self, video_path: str) -> str | None:
        """使用 ffprobe 获取视频的精确时长（秒）."""
        try:
            # 优先尝试从视频流获取时长
            command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=nw=1:nk=1",
                video_path,
            ]
            duration = subprocess.check_output(command, text=True, stderr=subprocess.PIPE).strip()
            if duration:
                return duration

            # 如果流时长为空，回退到容器格式时长
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                video_path,
            ]
            duration = subprocess.check_output(command, text=True, stderr=subprocess.PIPE).strip()
            if duration:
                return duration

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("无法使用 ffprobe 获取视频时长 '%s': %s", video_path, str(e))
        return None

    def _merge_audio_to_video(self, temp_video_path: str, original_video_path: str, output_path: str) -> bool:
        """将音频合并到视频，并精确对齐时长."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".aac", delete=True) as temp_audio:
                # 1. 提取原始音频
                audio_extract_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    original_video_path,
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-vn",
                    temp_audio.name,
                ]

                subprocess.run(audio_extract_cmd, check=True)

                # 2. 尝试获取视频时长并使用策略一（-t）
                video_duration = self._get_video_duration(temp_video_path)
                if video_duration:
                    logger.info("获取到视频时长为 %s 秒，使用 -t 参数进行精确对齐。", video_duration)
                    audio_merge_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_video_path,
                        "-i",
                        temp_audio.name,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "copy",
                        "-t",
                        video_duration,
                        "-movflags",
                        "+faststart",
                        output_path,
                    ]
                # 3. 如果无法获取时长，回退到策略二（apad + shortest）
                else:
                    logger.warning("无法获取视频时长，回退到 apad + shortest 方案（将重编码音频）。")
                    audio_merge_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_video_path,
                        "-i",
                        temp_audio.name,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-filter:a",
                        "apad",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",  # 音频重编码
                        "-shortest",
                        "-movflags",
                        "+faststart",
                        output_path,
                    ]

                subprocess.run(audio_merge_cmd, check=True, capture_output=True, text=True)
            return True

        except Exception as e:
            stderr = e.stderr if isinstance(e, subprocess.CalledProcessError) else ""
            logger.warning("合并音频失败: %s. FFmpeg输出: %s. 回退到仅视频输出。", str(e), stderr)
            shutil.copy2(temp_video_path, output_path)
            return False

    def _validate_and_prepare_input(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        output_basename: str | None,
    ) -> tuple[bool, str]:
        is_valid_video_path = not_blank(video_path)
        if not is_valid_video_path and video_binary is None:
            raise ValueError("Either video_path or video_binary must be provided")

        if not_blank(output_basename):
            video_name_base = str(output_basename)
        elif is_valid_video_path and video_path is not None:
            video_name_base = Path(str(video_path)).stem
        else:
            video_name_base = f"binary_{uuid.uuid4().hex}"

        return is_valid_video_path, video_name_base

    def _get_video_metadata(
        self, video_cap: cv2.VideoCapture
    ) -> tuple[int, float, int, int, tuple[int, int], tuple[int, int]]:
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        mask_size = (height, width)
        return frame_count, fps, width, height, size, mask_size

    def _drain_pipe(self, pipe: IO[bytes], sink: Callable[[str], None]) -> None:
        """持续读取管道内容并送入 sink 函数."""
        for line in iter(pipe.readline, b""):
            sink(line.decode(errors="ignore").strip())
        pipe.close()

    def _create_ffmpeg_process(self, width: int, height: int, fps: float, output_file: str) -> subprocess.Popen[bytes]:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            # 日志控制
            "-loglevel",
            "error",  # 只报告错误，大幅减少 stderr 输出
            "-hide_banner",  # 隐藏版本信息
            "-nostats",  # 关闭周期性统计信息
            output_file,
        ]

        # 启动后台线程读取 stderr (推荐，兼顾性能与调试)
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert ffmpeg_process.stdin is not None
        assert ffmpeg_process.stderr is not None

        # 创建并启动一个守护线程来消耗 stderr，避免阻塞
        stderr_thread = threading.Thread(
            target=self._drain_pipe,
            args=(ffmpeg_process.stderr, logger.warning),  # 将FFmpeg日志以warning级别打印
            daemon=True,
        )
        stderr_thread.start()
        return ffmpeg_process

    def _prepare_inpaint_mask(
        self, inpaint_areas: list[dict[str, Any]] | None, size: tuple[int, int], mask_size: tuple[int, int]
    ) -> Any:
        mask_coordinates = self._parse_inpaint_areas(inpaint_areas)

        if not mask_coordinates:
            mask_coordinates = [(0, size[0], 0, size[1])]
            logger.info("Using full-screen inpaint area")
        else:
            logger.info("Setting %d inpaint areas:", len(mask_coordinates))
            for i, (xmin, xmax, ymin, ymax) in enumerate(mask_coordinates):
                logger.info("  Area %d: ymin=%d, ymax=%d, xmin=%d, xmax=%d", i + 1, ymin, ymax, xmin, xmax)

        return self._create_mask(mask_size, mask_coordinates)

    def _process_video_frames(
        self, video_cap: cv2.VideoCapture, ffmpeg_process: subprocess.Popen[bytes], sttn_inpaint: Any, mask: Any
    ) -> int:
        frames_buffer: list[Any] = []
        current_frame = 0
        processed_frames = 0
        processed_batches = 0
        min_required = self.reference_length * self.neighbor_stride
        while True:
            ret, frame = video_cap.read()
            if not ret:
                # 最终 flush 尾批
                if frames_buffer:
                    processed_batches += 1
                    logger.info("Processing final batch %d: %d frames", processed_batches, len(frames_buffer))
                    # 尾批最小长度保护
                    if len(frames_buffer) < min_required and len(frames_buffer) >= 1:
                        last = frames_buffer[-1]
                        frames_buffer = frames_buffer + [last] * (min_required - len(frames_buffer))
                    inpainted_frames = sttn_inpaint(frames_buffer, mask)
                    # 输出长度对齐
                    if len(inpainted_frames) < len(frames_buffer):
                        pad = inpainted_frames[-1] if inpainted_frames else frames_buffer[-1]
                        inpainted_frames = inpainted_frames + [pad] * (len(frames_buffer) - len(inpainted_frames))
                    # 仅写入原始长度部分，避免填充导致时长变长
                    write_count = current_frame % self.max_load_num or len(inpainted_frames)
                    if ffmpeg_process.stdin is None:
                        logger.error("FFmpeg stdin pipe is closed.")
                        break
                    for f in inpainted_frames[:write_count]:
                        ffmpeg_process.stdin.write(f.tobytes())
                        # 手动flush
                        ffmpeg_process.stdin.flush()
                        processed_frames += 1
                break
            frames_buffer.append(frame)
            current_frame += 1
            if len(frames_buffer) >= self.max_load_num:
                processed_batches += 1
                logger.info("Processing batch %d: %d frames", processed_batches, len(frames_buffer))
                inpainted_frames = sttn_inpaint(frames_buffer, mask)
                # 输出长度对齐
                if len(inpainted_frames) < len(frames_buffer):
                    pad = inpainted_frames[-1]
                    inpainted_frames = inpainted_frames + [pad] * (len(frames_buffer) - len(inpainted_frames))
                if ffmpeg_process.stdin is None:
                    logger.error("FFmpeg stdin pipe is closed.")
                    break
                for f in inpainted_frames[: len(frames_buffer)]:
                    ffmpeg_process.stdin.write(f.tobytes())
                    # 手动flush
                    ffmpeg_process.stdin.flush()
                    processed_frames += 1
                frames_buffer = []
        return processed_frames

    def _finalize_ffmpeg_process(self, ffmpeg_process: subprocess.Popen[bytes]) -> None:
        # 1. 首先关闭 stdin，这是通知 FFmpeg 数据已写完的关键信号
        if ffmpeg_process.stdin:
            try:
                ffmpeg_process.stdin.close()
            except BrokenPipeError:
                # 如果 FFmpeg 已经因为其他错误退出，这里会触发 BrokenPipeError
                logger.warning("FFmpeg stdin pipe was already closed.")

        # 2. 等待进程结束，设置超时
        try:
            # 等待 FFmpeg 完成所有编码和文件写入操作
            ffmpeg_process.wait(timeout=60)  # 超时可以设得宽松一些
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out after 60s. Terminating...")
            ffmpeg_process.kill()  # 强制终止
            # 等待 kill 完成
            ffmpeg_process.wait()
            # 即使超时，也尝试读取残余的 stderr 输出以供分析
            stderr_output = ""
            if ffmpeg_process.stderr:
                stderr_output = ffmpeg_process.stderr.read().decode(errors="ignore")
            raise RuntimeError(f"FFmpeg encoding timed out. Stderr: {stderr_output}")

        # 3. 检查退出码
        if ffmpeg_process.returncode != 0:
            # 如果使用了后台线程，大部分日志已被消费。这里可能只能读到最后一点。
            stderr_output = ""
            if ffmpeg_process.stderr:
                stderr_output = ffmpeg_process.stderr.read().decode(errors="ignore")
            logger.error("FFmpeg failed with exit code %s. Stderr: %s", ffmpeg_process.returncode, stderr_output)
            raise RuntimeError(
                f"FFmpeg encoding failed. Exit code: {ffmpeg_process.returncode}. Stderr: {stderr_output}"
            )
        else:
            logger.info("FFmpeg process finished successfully.")

    def _upload_final_video(self, temp_video_path: str, local_video_path: str, video_name: str) -> str:
        output_path = f"{video_name}_inpainted.mp4"
        tos_output_path = f"{self.output_tos_dir}/{output_path}"
        if self.is_keep_audio:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_final:
                self._merge_audio_to_video(temp_video_path, local_video_path, temp_final.name)
                upload_file(temp_final.name, tos_output_path)
        # 忽略音频，只保留视频
        else:
            upload_file(temp_video_path, tos_output_path)
        return tos_output_path

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        inpaint_areas: list[dict[str, Any]] | None = None,
        output_basename: str | None = None,
    ) -> dict[str, Any]:
        try:
            is_valid_video_path, video_name_base = self._validate_and_prepare_input(
                video_path, video_binary, output_basename
            )
        except ValueError:
            return {"output_path": None, "processed_frames": None, "processed_resolution": None}

        def process_single_video(
            local_video_path: str, inpaint_areas: list[dict[str, Any]] | None = None
        ) -> dict[str, Any]:
            video_cap = cv2.VideoCapture(local_video_path)
            if not video_cap.isOpened():
                logger.error("Cannot open video: %s", local_video_path)
                return {"output_path": None, "processed_frames": None, "processed_resolution": None}

            try:
                frame_count, fps, width, height, size, mask_size = self._get_video_metadata(video_cap)
                total_batches = (frame_count + self.max_load_num - 1) // self.max_load_num

                # 计算默认字幕位置
                if inpaint_areas is None and self.is_set_default_inpaint_areas:
                    inpaint_areas = [{"ymin": int(height * 0.7), "ymax": height, "xmin": 0, "xmax": width}]

                logger.info(
                    "Starting video processing: %s - frames: %d, resolution: %dx%d, estimated batches: %d",
                    local_video_path,
                    frame_count,
                    width,
                    height,
                    total_batches,
                )

                with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
                    ffmpeg_process = self._create_ffmpeg_process(width, height, fps, temp_video.name)

                    try:
                        sttn_inpaint = self._load_sttn_model()
                        mask = self._prepare_inpaint_mask(inpaint_areas, size, mask_size)

                        processed_frames = self._process_video_frames(video_cap, ffmpeg_process, sttn_inpaint, mask)

                        video_cap.release()
                        self._finalize_ffmpeg_process(ffmpeg_process)

                        final_output_path = self._upload_final_video(temp_video.name, local_video_path, video_name_base)

                        logger.info(
                            "Video processing completed - processed frames: %d/%d, "
                            "original: %d frames %dx%d, processed: %d frames %dx%d",
                            processed_frames,
                            frame_count,
                            frame_count,
                            width,
                            height,
                            processed_frames,
                            size[0],
                            size[1],
                        )

                        return {
                            "output_path": final_output_path,
                            "processed_frames": int(processed_frames),
                            "processed_resolution": [int(size[0]), int(size[1])],
                        }

                    except Exception as e:
                        logger.error("Failed to process video %s: %s", local_video_path, str(e))
                        video_cap.release()

                        if "ffmpeg_process" in locals() and ffmpeg_process.poll() is None:
                            ffmpeg_process.terminate()
                            ffmpeg_process.wait()

                        return {"output_path": None, "processed_frames": None, "processed_resolution": None}

            except Exception as e:
                logger.error("Failed to get video metadata %s: %s", local_video_path, str(e))
                video_cap.release()
                return {"output_path": None, "processed_frames": None, "processed_resolution": None}

        if is_valid_video_path and video_path is not None:
            # 限定识别的区域
            return run_on_local_path(str(video_path), lambda path: process_single_video(path, inpaint_areas))

        elif video_binary is not None:
            with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                video_extension = f".{video_format}" if video_format else ".mp4"
                temp_filepath = Path(temp_sub_dir) / f"temp_video_{uuid.uuid4().hex}{video_extension}"

                with temp_filepath.open("wb") as tmp:
                    tmp.write(video_binary)

                return process_single_video(str(temp_filepath), inpaint_areas)
        else:
            return {"output_path": None, "processed_frames": None, "processed_resolution": None}

    def transform(
        self,
        video_paths: pa.Array,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        inpaint_areas: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """批量处理视频列进行智能修复

        该方法使用预加载的STTN模型对输入的视频进行批量修复处理，生成包含修复后视频路径、
        处理帧数和分辨率信息的结构化结果。

        Args:
            video_paths: 视频文件路径列（本地、TOS、HTTP等），与video_binaries二选一
            video_binaries: 视频二进制数据列，与video_paths二选一
            video_formats: 视频格式字符串列，配合video_binaries使用
            inpaint_areas: 修复区域坐标列，List<Struct>格式：
                [{'ymin': int, 'ymax': int, 'xmin': int, 'xmax': int}, ...]
                与video_watermark_detect算子输出格式兼容
                若为None则进行全屏修复
            output_basenames: 输出文件基础名称列（不含扩展名）
                若为None则使用原文件名加"_inpainted"后缀

        Returns:
            pyarrow.Array: 处理后的结构化列，每个元素包含以下字段：
                - output_path: 修复后视频的TOS存储路径；
                - processed_frames: 实际处理的视频帧数；
                - processed_resolution: 处理后视频的分辨率[width, height]

        Raises:
            ValueError: 当视频路径无效或修复区域坐标错误时抛出
            RuntimeError: 模型加载失败或视频处理过程中出现错误时抛出
        """  # noqa: D415
        paths_list = video_paths.to_pylist()
        total_videos = len(paths_list)

        binaries_list = video_binaries.to_pylist() if video_binaries is not None else [None] * total_videos
        formats_list = video_formats.to_pylist() if video_formats is not None else [None] * total_videos
        areas_list = inpaint_areas.to_pylist() if inpaint_areas is not None else [None] * total_videos
        basenames_list = output_basenames.to_pylist() if output_basenames is not None else [None] * total_videos

        results = []

        for i, (video_path, video_binary, video_format, inpaint_area, basename) in enumerate(
            zip(paths_list, binaries_list, formats_list, areas_list, basenames_list)
        ):
            try:
                logger.info(
                    "Processing video %d/%d: %s", i + 1, total_videos, video_path if video_path else "binary data"
                )
                result = self._process_video(video_path, video_binary, video_format, inpaint_area, basename)
                results.append(result)
                logger.info("Completed video %d/%d", i + 1, total_videos)
            except Exception as e:
                logger.error("Failed to process video %s: %s", video_path if video_path else "binary data", str(e))
                results.append({"output_path": None, "processed_frames": None, "processed_resolution": None})

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("output_path", pa.string()),
            pa.field("processed_frames", pa.int32()),
            pa.field("processed_resolution", pa.list_(pa.int32())),
        ]
        return pa.struct(fields)
