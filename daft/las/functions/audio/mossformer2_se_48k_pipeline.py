from __future__ import annotations

import os
from argparse import Namespace

import torch
import torchaudio
from clearvoice.utils.misc import compute_fbank, istft, stft

# Constant for normalizing audio values
MAX_WAV_VALUE = 32768.0


class MossFormer2_SE_48K_Pipeline:
    def __init__(self, model_path: str, batch_size: int = 10, device: str | None = None):
        self.args = Namespace(
            mode="inference",
            sampling_rate=48000,
            network="MossFormer2_SE_48K",
            checkpoint_dir=model_path,
            one_time_decode_length=20,
            decode_window=4,
            win_type="hamming",
            win_len=1920,
            win_inc=384,
            fft_len=1920,
            num_mels=60,
            task="speech_enhancement",
        )

        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Import the MossFormer2 speech enhancement model for 48 kHz
        from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K

        # Initialize the model
        self.model = MossFormer2_SE_48K(self.args).model
        self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def enhance(self, waveforms: list[torch.Tensor], sample_rates: list[int]) -> list[torch.Tensor]:
        enhanced_inputs = []
        for waveform, sample_rate in zip(waveforms, sample_rates):
            enhanced_inputs.append(self.enhance_single_audio(waveform, sample_rate))
        return enhanced_inputs

    def load_model(self, checkpoint_dir: str, model_key: str | None = None) -> None:
        model_name = os.path.join(checkpoint_dir, "last_best_checkpoint")
        with open(model_name) as f:
            model_name = f.readline().strip()
            # Form the full path to the model's checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, model_name)
        # Load the checkpoint file into memory (map_location ensures compatibility with different devices)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Load the model's state dictionary (weights and biases) into the current model
        if model_key in checkpoint:
            pretrained_model = checkpoint[model_key]
        else:
            pretrained_model = checkpoint
        state = self.model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif (
                key.replace("module.", "") in pretrained_model
                and state[key].shape == pretrained_model[key.replace("module.", "")].shape
            ):
                state[key] = pretrained_model[key.replace("module.", "")]
            elif "module." + key in pretrained_model and state[key].shape == pretrained_model["module." + key].shape:
                state[key] = pretrained_model["module." + key]
            else:
                print(f"{key} not loaded")
        self.model.load_state_dict(state)

    def enhance_single_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        single_channel_audios = []

        if waveform.dim == 1:
            single_channel_audios.append(waveform)
        else:
            for i in range(waveform.shape[0]):
                single_channel_audios.append(waveform[i])

        MAX_WAV_VALUE_16B = 32768.0
        MAX_WAV_VALUE_32B = 2147483648.0

        target_sr = self.args.sampling_rate
        resampled_audios = []

        for i in range(len(single_channel_audios)):
            audio = (single_channel_audios[i] * 32768).round().to(torch.int16)
            if torch.max(audio) > MAX_WAV_VALUE_16B:
                audio = audio / MAX_WAV_VALUE_32B
            else:
                audio = audio / MAX_WAV_VALUE_16B
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            resampled_audio = resampler(audio)

            resampled_audios.append(resampled_audio)

        length = resampled_audios[0].shape[0]
        enhanced_audios = []
        for i in range(len(resampled_audios)):
            enhanced_audios.append(self.decode_one_audio_mossformer2_se_48k(resampled_audios[i], self.batch_size))

        result = torch.stack([enhanced_audio[:length] for enhanced_audio in enhanced_audios], dim=0).to(waveform.device)

        if waveform.dim == 1:
            result = result.reshape(-1)
        else:
            result = result.reshape(waveform.shape[0], -1)
        return result

    def decode_one_audio_mossformer2_se_48k(self, inputs: torch.Tensor, batch_size: int = 10) -> torch.Tensor:
        """Processes audio inputs through the MossFormer2 model for speech enhancement at 48kHz.

        This function decodes audio input using the following steps:
        1. Normalizes the audio input to a maximum WAV value.
        2. Checks the length of the input to decide between online decoding and batch processing.
        3. For longer inputs, processes the audio in segments using a sliding window.
        4. Computes filter banks and their deltas for the audio segment.
        5. Passes the filter banks through the model to get a predicted mask.
        6. Applies the mask to the spectrogram of the audio segment and reconstructs the audio.
        7. For shorter inputs, processes them in one go without segmentation.

        Args:
            model (nn.Module): The trained MossFormer2 model used for decoding.
            device (torch.device): The device (CPU or GPU) for computation.
            inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size and T is the number of time steps.
            args (Namespace): Contains arguments for sampling rate, window size, and other parameters.

        Returns:
            torch.Tensor: The decoded audio output, normalized to the range [-1, 1].
        """
        inputs = inputs.to(self.device)
        input_len = inputs.shape[0]  # Get the length of the input audio
        inputs = inputs * MAX_WAV_VALUE  # Normalize the input to the maximum WAV value

        # Check if input length exceeds the defined threshold for online decoding
        if input_len > self.args.sampling_rate * self.args.one_time_decode_length:  # 20 seconds
            window = int(self.args.sampling_rate * self.args.decode_window)  # Define window length (e.g., 4s for 48kHz)
            stride = int(window * 0.75)  # Define stride length (e.g., 3s for 48kHz)
            t = inputs.shape[0]  # Update length after potential padding

            # Pad input if necessary to match window size
            if t < window:
                inputs = torch.concatenate([inputs, torch.zeros(window - t, device=inputs.device)], 0)
            elif t < window + stride:
                padding = window + stride - t
                inputs = torch.concatenate([inputs, torch.zeros(padding, device=inputs.device)], 0)
            else:
                if (t - window) % stride != 0:
                    padding = ((t - window) // stride + 1) * stride - (t - window)
                    inputs = torch.concatenate([inputs, torch.zeros(padding, device=inputs.device)], 0)

            audio = inputs.to(torch.float32)  # Convert to float32 Torch tensor
            t = audio.shape[0]  # Update length after conversion
            outputs = torch.zeros(t, device=audio.device)  # Initialize output tensor
            give_up_length = (window - stride) // 2  # Determine length to ignore at the edges
            dfsmn_memory_length = 0  # Placeholder for potential memory length
            batch_start_idx = 0  # Initialize current index for sliding window

            # Process audio in sliding window segments
            while batch_start_idx + window <= t:
                audio_segments = []
                # Select appropriate segment of audio for processing
                for current_idx in range(batch_start_idx, batch_start_idx + stride * batch_size, stride):
                    if current_idx + window > t:
                        break
                    if current_idx < dfsmn_memory_length:
                        audio_segments.append(audio[0 : current_idx + window])
                    else:
                        audio_segments.append(audio[current_idx - dfsmn_memory_length : current_idx + window])

                all_fbanks = []
                for audio_segment in audio_segments:
                    # Compute filter banks for the audio segment
                    fbanks = compute_fbank(audio_segment.unsqueeze(0), self.args)
                    # Compute deltas for filter banks
                    fbank_tr = torch.transpose(fbanks, 0, 1)  # Transpose for delta computation
                    fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)  # First-order delta
                    fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)  # Second-order delta

                    # Transpose back to original shape
                    fbank_delta = torch.transpose(fbank_delta, 0, 1)
                    fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

                    # Concatenate the original filter banks with their deltas
                    fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
                    all_fbanks.append(fbanks)
                fbanks = torch.stack(all_fbanks, dim=0)  # Add batch dimension and move to device
                # Pass filter banks through the model
                with torch.no_grad():
                    Out_List = self.model(fbanks)
                pred_mask = Out_List[-1]  # Get the predicted mask from the output
                pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
                # Apply STFT to the audio segment
                current_idx = batch_start_idx
                for i, audio_segment in enumerate(audio_segments):
                    spectrum = stft(audio_segment, self.args)

                    # print(pred_mask)
                    masked_spec = spectrum * pred_mask[:, :, i : i + 1]  # Apply mask to the spectrum
                    masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form

                    # Reconstruct audio from the masked spectrogram
                    output_segment = istft(masked_spec_complex, self.args, len(audio_segment))

                    # Store the output segment in the output tensor
                    if current_idx == 0:
                        outputs[current_idx : current_idx + window - give_up_length] = output_segment[:-give_up_length]
                    else:
                        output_segment = output_segment[-window:]  # Get the latest window of output
                        outputs[current_idx + give_up_length : current_idx + window - give_up_length] = output_segment[
                            give_up_length:-give_up_length
                        ]

                    current_idx += stride  # Move to the next segment
                batch_start_idx += len(audio_segments) * stride

        else:
            # Process the entire audio at once if it is shorter than the threshold
            audio = inputs.to(torch.float32)
            fbanks = compute_fbank(audio.unsqueeze(0), self.args)

            # Compute deltas for filter banks
            fbank_tr = torch.transpose(fbanks, 0, 1)
            fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
            fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
            fbank_delta = torch.transpose(fbank_delta, 0, 1)
            fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

            # Concatenate the original filter banks with their deltas
            fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
            fbanks = fbanks.unsqueeze(0)  # Add batch dimension and move to device

            # Pass filter banks through the model
            with torch.no_grad():
                Out_List = self.model(fbanks)
            pred_mask = Out_List[-1]  # Get the predicted mask
            spectrum = stft(audio, self.args)  # Apply STFT to the audio
            pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
            masked_spec = spectrum * pred_mask  # Apply mask to the spectrum
            masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form

            # Reconstruct audio from the masked spectrogram
            outputs = istft(masked_spec_complex, self.args, len(audio))

        return outputs / MAX_WAV_VALUE  # Return the output normalized to [-1, 1]
