# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pyannote.audio import Pipeline

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio
from daft.las.functions.utils.common_utils import tracking_usage

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@contextmanager
def temporary_cd(path: Path) -> Generator[None, None, None]:
    original_dir = Path.cwd()
    Path(path).mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


def load_pipeline_from_pretrained(config_path: str | Path) -> Pipeline:
    if isinstance(config_path, str):
        config_path = Path(config_path)

    logger.info("Loading pyannote pipeline from %s...", config_path)

    # the model file defined in config file is relative path,
    # so we have to go to model path to load data.
    # Example:
    # config_path: /xxxx/model_name/models/xxx.yaml
    # model_path: /xxxx/model_name
    model_path = config_path.parent.parent.resolve()
    with temporary_cd(model_path):
        return Pipeline.from_pretrained(config_path)


class AudioSpeakerDiarization(Operator):
    """**基于pyannote-audio的说话人分离处理器**

    **核心功能**
    - 多说话人语音分离与时间戳标注
    - 输出带说话人标签的语音分段元数据

    **输入/输出**
    - 输入：音频
    - 输出：包含以下字段的结构化数据
        - speaker: 说话人唯一标识
        - start: 语音段开始时间（秒）
        - end: 语音段结束时间（秒）

    **技术特性**
    - 使用`speaker-diarization-3.1`说话人分离模型
    - 支持GPU加速推理（需配置CUDA环境）
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """初始化 AudioSpeakerDiarization 类的实例

        Args:
            model_path: 模型文件所在的路径
            rank: 用于指定使用的 GPU 设备编号
            **kwargs: 传递给父类构造函数的其他关键字参数
        """  # noqa: D415
        super().__init__(**kwargs)
        model_config_file = Path(model_path).joinpath("speaker-diarization-3.1/models/pyannote_diarization_config.yaml")
        self.model = load_pipeline_from_pretrained(model_config_file)

        if self.use_gpu:
            rank = 0 if rank is None else rank
            # change cuda_device_count
            self.rank = rank % self.cuda_device_count
            logger.info("Model will be loaded on device: %s", f"cuda:{self.rank}")
            self.model.to(torch.device(f"cuda:{self.rank}"))
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info("The model is loaded from %s.", model_config_file)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="speaker-diarization-3.1")

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("end", pa.float64()),
            pa.field("speaker", pa.string()),
            pa.field("start", pa.float64()),
        ]
        return pa.list_(pa.struct(fields))

    def _diarization(self, audio: Any) -> list[dict[str, Any]] | None:
        try:
            samples = decode_audio(audio).get_all_samples()
            waveform = samples.data.to(device=self.device)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            diarization = self.model({"waveform": waveform, "sample_rate": samples.sample_rate, "channel": 0})
            return [
                {
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                }
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        except Exception:
            logger.exception("Failed to process")
            return None

    def transform(self, audios: pa.Array) -> pa.Array:
        """对输入的音频数组进行批量说话人分离处理

        Args:
            audios: 包含多个音频数据的数组

        Returns:
            pa.Array: 包含说话人分离结果的数组
        """  # noqa: D415
        result = [self._diarization(audio.as_py()) for audio in audios]
        return pa.array(result, type=self.__return_column_type__())
