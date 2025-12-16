# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file


class AudioMetascore(Operator):
    """**音频评分算子 - 使用 audiobox_aesthetics 对音频进行质量评分**

    **核心功能**
    - 使用 audiobox_aesthetics 模型对音频片段进行质量评分
    - 提供四个评分维度：CE (连贯性/听感投入度)、CU (清晰度/可懂度)、PC (制作质量/构成质量)、PQ (感知质量/主观音质)
    - 支持本地和远程 (tos://) 音频文件
    - 适用于音频质量评估、音频筛选、质量控制等场景

    **评分维度说明**
    - CE (连贯性/听感投入度): 反映音频内容是否连贯，以及音频在主观听感上是否具有吸引力
    - CU (清晰度/可懂度): 反映声音清晰度、语音可懂度
    - PC (制作质量/构成质量): 反映录音、混音、结构等制作层面的质量
    - PQ (感知质量/主观音质): 综合主观听感质量
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "audiobox-aesthetics/checkpoint.pt",
        **kwargs: Any,
    ) -> None:
        """初始化音频评分算子。

        参数：
            model_path (str): 模型存储根路径
                默认值："/opt/las/models"
            model_name (str): 预训练模型名称
                默认值："audiobox-aesthetics/checkpoint.pt"
        """  # noqa: D415
        super().__init__(**kwargs)

        # 初始化 logger 实例
        self.logger = get_logger(f"AudioMetascore-{id(self)}")

        # 初始化计数器
        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.model_path = model_path
        self.model_name = model_name

        # 初始化 audiobox_aesthetics predictor
        try:
            from audiobox_aesthetics.infer import initialize_predictor

            model_file = Path(self.model_path) / self.model_name
            self.predictor = initialize_predictor(str(model_file))
            self.logger.info("AudioMetascore initialized with model from %s", model_file)
        except Exception as e:
            self.logger.error("Failed to initialize predictor: %s", e)
            raise RuntimeError(f"Failed to initialize audiobox_aesthetics predictor: {e}") from e

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="audiobox_aesthetics")

    def log_progress(self) -> None:
        """记录处理进度."""
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        """Define the return type for the operator."""
        # 返回评分结构：CE, CU, PC, PQ
        return pa.struct(
            [
                pa.field("CE", pa.float64()),  # 连贯性/听感投入度
                pa.field("CU", pa.float64()),  # 清晰度/可懂度
                pa.field("PC", pa.float64()),  # 制作质量/构成质量
                pa.field("PQ", pa.float64()),  # 感知质量/主观音质
            ]
        )

    def transform(self, audio_paths: pa.Array) -> pa.Array:
        """对输入的音频片段进行评分。

        本方法基于 audiobox_aesthetics 模型实现音频评分，处理输入的音频文件路径，
        自动完成远程文件下载和预处理，输出四个维度的质量评分。

        Args:
            audio_paths (pa.Array): PyArrow array of audio file paths (strings)

        Returns:
            一个结构化结果数组，其中每个元素包含以下字段：
                - CE (float): 连贯性/听感投入度评分
                - CU (float): 清晰度/可懂度评分
                - PC (float): 制作质量/构成质量评分
                - PQ (float): 感知质量/主观音质评分
            处理失败的音频返回包含 0.0 值的结构。
        """  # noqa: D415
        results = []

        # 将数组转换为 Python 列表
        audio_paths_list = audio_paths.to_pylist()

        # 逐个处理每个音频文件
        for audio_path in audio_paths_list:
            self.submit_counter.increment()

            tmp_audio_file = None
            try:
                # 如果是远程路径（tos://），下载到临时文件
                if str(audio_path).startswith("tos://"):
                    tmp_in = tempfile.NamedTemporaryFile(
                        suffix=os.path.splitext(audio_path)[1] or ".flac", delete=False
                    )
                    tmp_in.close()
                    tmp_audio_file = tmp_in.name
                    download_file(audio_path, tmp_audio_file)
                    local_path = tmp_audio_file
                else:
                    # 本地路径直接使用
                    local_path = audio_path

                # 调用 predictor 进行评分
                score = self.predictor.forward([{"path": local_path}])[0]

                # 确保分数是字典格式
                if isinstance(score, dict):
                    # 提取四个指标，如果缺失则使用 0.0
                    result = {
                        "CE": float(score.get("CE", 0.0)),
                        "CU": float(score.get("CU", 0.0)),
                        "PC": float(score.get("PC", 0.0)),
                        "PQ": float(score.get("PQ", 0.0)),
                    }
                else:
                    # 如果返回格式不是字典，使用默认值
                    self.logger.warning("Unexpected score format: %s, using defaults", type(score))
                    result = {"CE": 0.0, "CU": 0.0, "PC": 0.0, "PQ": 0.0}

                results.append(result)
                self.success_counter.increment()

            except Exception as e:
                self.logger.exception("Failed to process audio %s: %s", audio_path, e)
                # 添加空结果
                results.append({"CE": 0.0, "CU": 0.0, "PC": 0.0, "PQ": 0.0})
                self.failed_counter.increment()

            finally:
                # 清理临时文件
                if tmp_audio_file and os.path.exists(tmp_audio_file):
                    try:
                        os.remove(tmp_audio_file)
                    except Exception as e:
                        self.logger.warning("Failed to remove temporary audio file %s: %s", tmp_audio_file, e)

            # 定期打印进度
            if len(results) % 10 == 0:
                self.log_progress()

        # 打印最终统计信息
        self.log_progress()

        return pa.array(results, type=self.__return_column_type__())
