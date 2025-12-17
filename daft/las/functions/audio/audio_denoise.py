from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np  # noqa: TID253
import torch
import torchaudio

from daft.dependencies import pa
from daft.las.functions.types import AsyncOperatorStats, EventLooper, Operator
from daft.las.functions.utils.common_utils import pre_sign_url_for_tos, run_on_local_path
from daft.las.io import upload_file

from .mossformer2_se_48k_pipeline import MossFormer2_SE_48K_Pipeline


class AudioDenoise(Operator):
    """Audio denoising operator using alibabasglab/MossFormer2_SE_48K model."""

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "MossFormer2_SE_48K",
        device: str | None = None,
        timeout: int | None = None,
        output_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AudioDenoise operator.

        Args:
            model_path: Path to the directory containing the model.
            model_name: Name of the pre-trained model to use.
            device: Device to run inference on (cuda or cpu), defaults to auto-detect.
            timeout: Processing timeout in seconds.
            output_format: The format to save the audio in (e.g., 'wav', 'flac').
                If None, uses the input file's extension.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"AudioDenoise-{id(self)}")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Model requires 48000 sample rate
        self.sample_rate = 48000
        self.timeout = timeout
        self.max_duration = kwargs.get("max_duration", 7200)
        self.output_format = output_format

        # Initialize pipeline
        mossformer_model_path = os.path.join(model_path, model_name)
        self.pipeline = MossFormer2_SE_48K_Pipeline(mossformer_model_path, device=self.device)

        # Initialize operator stats
        self.stats = AsyncOperatorStats(self.logger)
        self.looper = EventLooper()

        self.logger.info(
            "Initialized AudioDenoise with model=%s, device=%s, sample_rate=%s",
            mossformer_model_path,
            self.device,
            self.sample_rate,
        )

    def load_audio(self, input_path: str) -> tuple[torch.Tensor, int]:
        """Load an audio file from a given path.

        The path can be a local path or a remote URL (tos://, s3://).

        Args:
            input_path: Path to the audio file.

        Returns:
            A tuple containing the audio data as a numpy array and the sample rate.
        """

        def _load(path: str) -> tuple[torch.Tensor, int]:
            audio, sr = torchaudio.load(path)
            return audio, sr

        path_to_load = input_path
        if input_path.startswith(("tos://", "s3://")):
            path_to_load = pre_sign_url_for_tos(input_path, expires=3600)
        return run_on_local_path(path_to_load, _load)

    def denoise_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Denoise an audio signal.

        If the audio is longer than `self.max_duration`, it will be split into chunks,
        denoised separately, and then concatenated.

        Args:
            audio: The input audio signal as a numpy array.
            sample_rate: The sample rate of the audio.

        Returns:
            The denoised audio signal as a numpy array.
        """
        duration = audio.shape[1] / sample_rate
        if duration <= self.max_duration:
            # Run inference
            outputs = self.pipeline.enhance([audio], sample_rates=[sample_rate])

            # Extract denoised audio
            denoised_audio = outputs[0]
            return denoised_audio
        else:
            # Split audio into chunks
            chunk_size = self.max_duration * sample_rate
            chunks = [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]

            denoised_chunks = []
            for chunk in chunks:
                denoised_chunk = self.denoise_audio(chunk, sample_rate)
                denoised_chunks.append(denoised_chunk)

            return np.concatenate(denoised_chunks, axis=0)

    def save_audio(self, audio: torch.Tensor, output_path: str, input_path: str) -> str:
        """Save an audio signal to a given path.

        If the output path is a remote URL, the file is first saved locally and then uploaded.

        Args:
            audio: The audio signal to save.
            output_path: The path to save the audio file to.
            input_path: The path of the original audio file, used to determine the output format if not specified.
        """
        if self.output_format:
            suffix = f".{self.output_format}"
        else:
            _, suffix = os.path.splitext(input_path)
            if not suffix:
                suffix = ".wav"
        # if suffix is different from output file extension, change to output_format
        if not output_path.endswith(suffix):
            output_path = os.path.splitext(output_path)[0] + suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio, self.sample_rate)
            tmp_file_path = tmp_file.name

        try:
            # Upload to output path
            upload_file(tmp_file_path, output_path)
            return output_path
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    async def process(self, input_path: str, output_path: str) -> str | None:
        """Process a single audio file: load, denoise, and save.

        Args:
            input_path: Path to the input audio file.
            output_path: Path to save the denoised audio file.

        Returns:
            The output path if successful, otherwise None.
        """
        await self.stats.log_submit()
        try:
            # Load audio
            audio, sample_rate = self.load_audio(input_path)

            # Denoise audio
            denoised_audio = self.denoise_audio(audio, sample_rate)

            # Save result, output_path may be modified to match output format
            output_path = self.save_audio(denoised_audio, output_path, input_path)

            await self.stats.log_succeed()
            self.logger.info("Successfully denoised %s → %s", input_path, output_path)
            return output_path

        except Exception:
            await self.stats.log_failed()
            self.logger.exception("Failed to denoise %s", input_path)
            return None
        finally:
            await self.stats.log_process()

    def transform(self, input_col: pa.Array, output_col: pa.Array) -> pa.Array:
        """Denoise a batch of audio files.

        Args:
            input_col: A PyArrow array of input file paths.
            output_col: A PyArrow array of output file paths.

        Returns:
            A PyArrow array of output file paths for successfully processed files,
            with nulls for failures.
        """
        results = []
        self.looper.run(self.stats.log_accept(len(input_col)))
        for input_path, output_path in zip(input_col.to_pylist(), output_col.to_pylist()):
            try:
                result = self.looper.run(self.process(input_path, output_path))
            except Exception as e:
                self.logger.error("Error processing %s: %s", input_path, e)
                result = None
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        """Return the PyArrow data type for the output column."""
        return pa.large_string()
