# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Offline tool to extract 1024-d reference speaker embeddings for Qwen3-TTS voice cloning.

Supports extracting voice embeddings from reference WAV audio (~3s, 24kHz mono)
using the converted LiteRT TFLite speaker encoder model.
Outputs both `.npy` and raw `.bin` (4096 bytes) formats required by C++ runtime.

Usage:
  bazel run //third_party/py/litert_torch/generative/export_hf/model_ext/qwen3_tts:extract_speaker_embedding_offline -- \
    --wav_path="/path/to/target_speaker_3sec.wav" \
    --output_path="/tmp/custom_voice.bin" \
    --tflite_model="/tmp/qwen3_tts_gpu/speaker_encoder_fp32.tflite"
"""

import os

from absl import app
from absl import flags
import librosa
import numpy as np
import tensorflow as tf
import torch

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "wav_path",
    None,
    "Path to ~3s reference WAV audio file.",
)
flags.DEFINE_string(
    "output_path",
    None,
    "Destination path for extracted 1024-d embedding (.bin or .npy).",
)
flags.DEFINE_string(
    "tflite_model",
    "/tmp/qwen3_tts_gpu/speaker_encoder_fp32.tflite",
    "Path to exported speaker_encoder_fp32.tflite model.",
)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    num_mels: int = 128,
    hop_size: int = 256,
    win_size: int = 1024,
) -> np.ndarray:
  """Extracts 128-channel mel spectrogram from raw floating-point waveform."""
  mel_basis = librosa.filters.mel(
      sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=0, fmax=12000
  )
  y = torch.from_numpy(audio).float().unsqueeze(0)
  hann_window = torch.hann_window(win_size).to(y.device)

  padding = (n_fft - hop_size) // 2
  y = torch.nn.functional.pad(
      y.unsqueeze(1), (padding, padding), mode="reflect"
  ).squeeze(1)

  spec = torch.stft(
      y,
      n_fft,
      hop_length=hop_size,
      win_length=win_size,
      window=hann_window,
      center=False,
      pad_mode="reflect",
      normalized=False,
      onesided=True,
      return_complex=True,
  )
  spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
  mel_spec = torch.matmul(torch.from_numpy(mel_basis).float(), spec)
  mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
  return mel_spec.transpose(1, 2).numpy()  # [1, T, 128]


def run_tflite_inference(mel: np.ndarray, tflite_path: str) -> np.ndarray:
  """Executes speaker embedding extraction using LiteRT TFLite model."""

  interpreter = tf.lite.Interpreter(model_path=tflite_path)

  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  target_shape = input_details["shape"]
  target_frames = target_shape[1]
  current_frames = mel.shape[1]

  if target_frames > 0:
    if current_frames < target_frames:
      pad_width = target_frames - current_frames
      mel = np.pad(mel, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
    elif current_frames > target_frames:
      mel = mel[:, :target_frames, :]

  interpreter.set_tensor(input_details["index"], mel.astype(np.float32))
  interpreter.invoke()
  embedding = interpreter.get_tensor(output_details["index"])
  return embedding[0]


def main(_):
  if not os.path.exists(FLAGS.wav_path):
    raise FileNotFoundError(f"Reference audio not found at: {FLAGS.wav_path}")
  if not os.path.exists(FLAGS.tflite_model):
    raise FileNotFoundError(f"TFLite model not found at: {FLAGS.tflite_model}")

  print(f"Loading reference audio: {FLAGS.wav_path}")
  audio, sample_rate = librosa.load(FLAGS.wav_path, sr=24000, mono=True)
  print(
      f"Loaded audio sample rate: {sample_rate} Hz, length:"
      f" {len(audio)/sample_rate:.2f} s"
  )

  print("Extracting 128-channel mel spectrogram...")
  mel = extract_mel_spectrogram(
      audio=audio.astype(np.float32), sample_rate=sample_rate
  )
  print(f"Extracted acoustic features shape: {mel.shape}")

  print("Running TFLite speaker encoder inference...")
  embedding = run_tflite_inference(mel, FLAGS.tflite_model)
  embedding = embedding.astype(np.float32)

  out_dir = os.path.dirname(FLAGS.output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  if FLAGS.output_path.endswith(".npy"):
    np.save(FLAGS.output_path, embedding)
    print(
        f"Successfully saved NumPy embedding: {FLAGS.output_path} (shape"
        f" {embedding.shape})"
    )
  else:
    embedding.tofile(FLAGS.output_path)
    file_size = os.path.getsize(FLAGS.output_path)
    print(
        f"Successfully saved raw binary embedding: {FLAGS.output_path}"
        f" ({file_size} bytes)"
    )


if __name__ == "__main__":
  app.run(main)
