# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn import Module
from torch.nn import functional as F

from daft.dependencies import np, pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio, encode_audio
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class AudioSourceSeparation(Operator):
    """**音频人声分离模块 - 使用 Demucs 模型提取人声分量**

    **核心功能**
    - 使用 Demucs 模型进行人声分离（去除背景音乐、噪声）
    - 自动进行采样率对齐、声道调整、响度恢复
    - 适合用于语音增强、字幕生成、音视频前处理等场景
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        chunk_batch_size: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化音频人声分离算子。

        参数：
            model_path (str): 本地 Demucs 模型文件所在目录。
            chunk_batch_size (int): 推理时的音频分块的批次大小。
            device (str): 运行模型的设备（'cuda' 或 'cpu'）。
            num_coroutines (int): 异步处理并发限制。
        """  # noqa: D415
        super().__init__(**kwargs)
        self.chunk_batch_size = chunk_batch_size
        self.device = device
        self.num_coroutines = num_coroutines

        self.model_sr = 44100
        model_name = str(Path(model_path) / "demucs/htdemucs.pth")
        self.model = torch.load(model_name, weights_only=False)
        self.model.eval().to(self.device)
        logger.info("Loaded Demucs model from %s on %s", model_path, device)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="demucs")

    def _preprocess(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref = tensor.mean(0)
        norm_tensor = (tensor - ref.mean()) / (ref.std() + 1e-9)
        return norm_tensor, ref

    def _postprocess(self, sources: torch.Tensor, ref: torch.Tensor, original_sr: int) -> np.ndarray:
        sources = sources * ref.std() + ref.mean()
        vocals = sources[3]  # [drums, bass, other, vocals]
        vocals = vocals.mean(dim=0, keepdim=True)
        if original_sr != self.model_sr:
            vocals = torchaudio.functional.resample(vocals, orig_freq=self.model_sr, new_freq=original_sr)
        return vocals.squeeze(0).cpu().numpy()

    async def process(self, audio: bytes) -> bytes | None:
        if not audio:
            return None
        try:
            audio_samples = decode_audio(audio, sample_rate=self.model_sr, num_channels=2).get_all_samples()
            original_sr = audio_samples.sample_rate

            input_tensor, ref = self._preprocess(audio_samples.data.to(self.device))
            sources = self._apply_model_pytorch_batch(self.model, input_tensor[None])  # shape: (1, 4, C, T)
            vocals = self._postprocess(sources[0], ref, original_sr)

            return encode_audio({"samples": vocals, "sample_rate": original_sr})  # type: ignore
        except Exception as e:
            logger.warning("Voice separation failed: %s", e)
            return None

    def _apply_model_pytorch_batch(
        self, model: Module, mix: torch.Tensor, overlap: float = 0.25, transition_power: float = 1.0
    ) -> torch.Tensor:
        device = mix.device
        batch_size, channels, total_length = mix.shape
        out = torch.zeros(batch_size, len(model.sources), channels, total_length, device=mix.device)
        sum_weight = torch.zeros(total_length, device=mix.device)

        model_segment = 39 / 5
        segment = int(model_segment * self.model_sr)
        stride = int(segment * (1 - overlap))

        weight = torch.cat(
            [
                torch.arange(1, segment // 2 + 1, device=device),
                torch.arange(segment - segment // 2, 0, -1, device=device),
            ]
        )
        weight = (weight / weight.max()) ** transition_power

        offsets = list(range(0, total_length, stride))

        for i in range(0, len(offsets), self.chunk_batch_size):
            batch_offsets = offsets[i : i + self.chunk_batch_size]

            chunks, chunk_lengths, valid_lengths = [], [], []
            for offset in batch_offsets:
                length = min(segment, total_length - offset)
                chunk = mix[..., offset : offset + length]
                chunk_lengths.append(length)
                valid_length = model.valid_length(length) if hasattr(model, "valid_length") else length
                valid_lengths.append(valid_length)
                pad = valid_length - length
                chunk = F.pad(chunk, (pad // 2, pad - pad // 2))
                chunks.append(chunk.to(device))

            padded_chunks = torch.cat(chunks, dim=0)
            with torch.no_grad():
                training_length = int(model_segment * self.model_sr)
                padded_chunks = F.pad(padded_chunks, (0, training_length - padded_chunks.shape[-1]))
                chunk_out = model(padded_chunks)
                chunk_out = chunk_out[..., :segment]

            for j, offset in enumerate(batch_offsets):
                trimmed_chunk = self._center_trim(chunk_out[j], chunk_lengths[j])
                length = trimmed_chunk.shape[-1]
                out[..., offset : offset + segment] += (weight[:length] * trimmed_chunk).to(mix.device)
                sum_weight[offset : offset + segment] += weight[:length].to(mix.device)

        out /= sum_weight
        return out

    def _center_trim(self, tensor: torch.Tensor, ref_len: int) -> torch.Tensor:
        delta = tensor.size(-1) - ref_len
        if delta < 0:
            raise ValueError("tensor must be larger than reference")
        if delta:
            tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        return tensor

    async def async_run(self, audio_list: list[bytes]) -> list[bytes | None]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded(audio: bytes) -> bytes | None:
            async with semaphore:
                return await self.process(audio)

        return await asyncio.gather(*[bounded(a) for a in audio_list])

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量处理音频数据，提取人声音频内容。

        本方法基于 Demucs 模型实现音频人声分离，异步处理输入的音频 byte 数组，
        自动完成采样率和声道调整，输出分离后的人声部分音频。

        Args:
            audio_col: 包含音频二进制数据的数组（pa.binary 类型），每个元素应为一段完整的音频内容。

        Returns:
            一个二进制数组（pa.binary 类型），每个元素为对应音频中提取出的人声部分，格式为编码后的音频 bytes。
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(audio_col.to_pylist()))
        return pa.array(results, type=AudioSourceSeparation.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.binary()
