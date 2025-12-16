# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa

from daft.dependencies import np, pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage

logger = logging.getLogger(__name__)


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    """ComputeScore class for evaluating DNSMOS."""

    def __init__(self, primary_model_path: str, device: str = "cpu") -> None:
        """Initialize the ComputeScore object.

        Args:
            primary_model_path: Path to the primary model.
            device: Device to run the models on ('cpu' or 'cuda').

        Raises:
            RuntimeError: If the device is not supported.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for AudioQualityScore. Install with: pip install onnxruntime-gpu"
            ) from e

        if device == "cuda":
            self.onnx_sess = ort.InferenceSession(primary_model_path, providers=["CUDAExecutionProvider"])
            logger.info("Using CUDA: %s", self.onnx_sess.get_providers())
        else:
            self.onnx_sess = ort.InferenceSession(primary_model_path)

    def get_polyfit_val(
        self, sig: float, bak: float, ovr: float, is_personalized_mos: bool
    ) -> tuple[float, float, float]:
        """Apply polynomial fitting to MOS scores.

        Args:
            sig: Signal MOS score.
            bak: Background MOS score.
            ovr: Overall MOS score.
            is_personalized_mos: Flag for personalized MOS.

        Returns:
            Tuple containing the adjusted signal, background, and overall MOS scores.
        """
        if is_personalized_mos:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio: np.ndarray, sampling_rate: int, is_personalized_mos: bool) -> dict[str, Any]:
        """Compute DNSMOS scores for an audio signal.

        Args:
            audio: Input audio signal.
            sampling_rate: Sampling rate of the input audio.
            is_personalized_mos: Flag for personalized MOS.

        Returns:
            Dictionary containing MOS scores.

        Raises:
            ValueError: If the input audio is not valid.
        """
        fs = SAMPLING_RATE
        if sampling_rate != fs:
            # resample audio
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=fs)

        actual_audio_len = len(audio)

        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_mos)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        return {
            "filename": "audio_clip",
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
        }


class AudioQualityScore(Operator):
    """**音频质量评分模块 - 使用 DNSMOS 模型评估音频质量**

    **核心功能**
    - 使用 DNSMOS (Deep Noise Suppression Mean Opinion Score) 模型进行音频质量评估
    - 提供整体质量 (OVRL)、信号质量 (SIG)、背景噪声 (BAK) 三个维度的评分
    - 评分范围在 1 到 5 之间，分数越高表示音频质量越好
    - 支持自动采样率转换和音频预处理
    - 适用于语音质量评估、音频筛选、质量控制等场景

    **评分维度说明**
    - OVRL (Overall): 音频整体质量评分，综合评估音频的可听性
    - SIG (Signal): 信号质量评分，评估语音信号的清晰度和自然度
    - BAK (Background): 背景噪声评分，评估背景噪声的干扰程度
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        device: str = "cpu",
        is_personalized_mos: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化音频质量评分算子。

        参数：
            model_path (str): 本地 DNSMOS 模型文件所在目录。
            device (str): 运行模型的设备（'cuda' 或 'cpu'）。
            is_personalized_mos (bool): 是否使用个性化 MOS 评分。
        """  # noqa: D415
        super().__init__(**kwargs)
        self.device = device
        self.is_personalized_mos = is_personalized_mos

        model_name = str(Path(model_path) / "dnsmos/sig_bak_ovr.onnx")
        self.model = ComputeScore(model_name, device)

        # Initialize counters for progress tracking
        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        # Initialize logger with unique identifier
        self.logger = get_logger(f"AudioQualityScore-{id(self)}")

        self.logger.info("Loaded DNSMOS model from %s on %s", model_path, device)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="dnsmos")

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def process(self, audio: bytes) -> dict[str, float | None]:
        """Process a single audio file and return the quality scores.

        Args:
            audio: Input audio binary data.

        Returns:
            Dictionary containing OVRL, SIG, and BAK scores, or None values if processing fails.
        """
        self.submit_counter.increment()

        if not audio:
            self.failed_counter.increment()
            self.log_progress()
            return {"ovrl": None, "sig": None, "bak": None}

        try:
            audio_samples = decode_audio(audio, sample_rate=SAMPLING_RATE, num_channels=1).get_all_samples()
            audio_array = audio_samples.data.squeeze().cpu().numpy()
            quality_scores = self.model(audio_array, audio_samples.sample_rate, self.is_personalized_mos)

            self.success_counter.increment()
            self.log_progress()
            return {
                "ovrl": float(quality_scores["OVRL"]),
                "sig": float(quality_scores["SIG"]),
                "bak": float(quality_scores["BAK"]),
            }
        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            logger.warning("Audio quality score computation failed: %s", e)
            return {"ovrl": None, "sig": None, "bak": None}

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量计算音频质量评分，对每段音频进行质量评估。

        本方法基于 DNSMOS 模型实现音频质量评分，处理输入的音频 byte 数组，
        自动完成采样率转换和预处理，输出三个维度的质量评分。

        Args:
            audio_col: 包含音频二进制数据的数组（pa.binary 类型），每个元素应为一段完整的音频内容。

        Returns:
            一个结构化结果数组，其中每个元素包含以下字段：
                - ovrl (float): 音频整体质量评分，综合评估音频的可听性，范围在 1.0 到 5.0 之间
                - sig (float): 信号质量评分，评估语音信号的清晰度和自然度，范围在 1.0 到 5.0 之间
                - bak (float): 背景噪声评分，评估背景噪声的干扰程度，范围在 1.0 到 5.0 之间
            处理失败的音频返回包含 null 值的结构。
        """  # noqa: D415
        audio_list = audio_col.to_pylist()
        results = [self.process(audio) for audio in audio_list]
        return pa.array(results, type=AudioQualityScore.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("ovrl", pa.float64()),
                pa.field("sig", pa.float64()),
                pa.field("bak", pa.float64()),
            ]
        )
