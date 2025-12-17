# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import json
import logging
import time
from typing import Any

import torch

from daft.dependencies import pa
from daft.las.functions.audio.beats.BEATs import BEATs, BEATsConfig  # type: ignore[attr-defined]
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio_torchaudio

logger = logging.getLogger(__name__)


class AudioBeatsClassifier(Operator):
    """基于 BEATs 模型实现的音频分类算子，用于识别音频中的节拍或事件并返回前 Top K 个最可能的分类结果标签和对应的概率值。

    Args:
        model_path: BEATs 模型文件所在目录，默认为 /opt/las/models，你可以从 https://github.com/microsoft/unilm/blob/master/beats/README.md 下载对应模型文件
        model_name: BEATs 模型文件名，默认值为 BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
        top_k: 设置返回 Top K 个最可能的分类标签，默认返回 5 个
        precision: 设置分类标签概率值保留的小数位数，默认为 None 表示不限制

    Returns:
        返回结果为 JSON 格式字符串（如下所示），包含 Top K 个最可能的分类标签和对应的概率值，其中分类标签参考 Google AudioSet 标签定义。

        [
            {"label": "/m/04rlf", "probability": 0.85},
            {"label": "/m/09x0r", "probability": 0.39},
            {"label": "/m/03qc9zr", "probability": 0.33},
            {"label": "/m/07sr1lc", "probability": 0.27},
            {"label": "/m/07s2xch", "probability": 0.15},
        ]

    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        top_k: int = 5,
        precision: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        start_time = time.time()
        self.model_path = f"{model_path}/{model_name}"
        self.top_k = top_k
        self.precision = precision

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.label_dict = self.checkpoint.get("label_dict", None)
        assert self.label_dict is not None, f"No label dict found in the checkpoint, model path is {self.model_path}"

        self.model = BEATs(cfg=BEATsConfig(self.checkpoint.get("cfg", None)))
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()

        # Move model if GPU requested
        if torch.cuda.is_available():
            self.model.to(self.device)

        # Monkey patch preprocess
        def _patched_preprocess(
            source: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582
        ) -> torch.Tensor:
            import torchaudio.compliance.kaldi as ta_kaldi

            fbanks = []
            for waveform in source:
                waveform = waveform.unsqueeze(0) * 2**15
                fbank = ta_kaldi.fbank(
                    waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10
                )
                fbanks.append(fbank)
            fbank = torch.stack(fbanks, dim=0)
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            return fbank.to(self.device)

        self.model.preprocess = _patched_preprocess

        logger.info(
            "Finish initializing audio beats classifier with top %s labels, model path: %s, device: %s, elapsed: %ss",
            self.top_k,
            self.model_path,
            self.device,
            round(time.time() - start_time, 2),
        )

    def transform(self, audios: pa.Array) -> pa.Array:
        return pa.array(obj=[self._classify(audio.as_py()) for audio in audios], type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()

    def _classify(self, audio: str | bytes) -> str:
        # Load audio and build padding mask
        waveform, _ = decode_audio_torchaudio(source=audio, sample_rate=16000, num_channels=1)

        # Do classify
        padding_mask = torch.zeros(1, waveform.shape[1], dtype=torch.bool)
        with torch.inference_mode():
            probs, _ = self.model.extract_features(waveform, padding_mask=padding_mask.to(self.device))

        # Build results
        top_probs, top_indices = probs.topk(k=self.top_k)
        top_labels = [self.label_dict[idx.item()] for idx in top_indices[0]]
        top_probs = top_probs[0].cpu().numpy().tolist()
        return json.dumps(
            [
                {
                    "label": label,
                    "probability": round(float(prob), self.precision) if self.precision else float(prob),
                }
                for label, prob in zip(top_labels, top_probs)
            ]
        )
