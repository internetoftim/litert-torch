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
"""Unit test verifying parity and GPU-compatibility of Qwen3-TTS MtpStepGpu."""

import os
import tempfile

from absl.testing import absltest
import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F

import litert_torch
from litert_torch.generative.export_hf.model_ext.qwen3_tts import exportable_module
from litert_torch.generative.export_hf.model_ext.qwen3_tts import speaker_encoder


class MtpStepReference5D(nn.Module):
  """Original un-fused 5D reference implementation with repeat_interleave."""

  def __init__(self, weights: dict[str, torch.Tensor]):
    super().__init__()
    for key, tensor in weights.items():
      self.register_buffer(key.replace(".", "_"), tensor, persistent=False)
    inv_freq = 1.0 / (
        exportable_module.THETA
        ** (
            torch.arange(0, exportable_module.HEAD_DIM, 2, dtype=torch.float32)
            / exportable_module.HEAD_DIM
        )
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.register_buffer(
        "slots",
        torch.arange(exportable_module.CACHE, dtype=torch.int32),
        persistent=False,
    )

  def _rms(self, x, weight):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + exportable_module.EPS)) * weight

  def _rotate_half(self, x):
    a, b = (
        x[..., : exportable_module.HEAD_DIM // 2],
        x[..., exportable_module.HEAD_DIM // 2 :],
    )
    return torch.cat((-b, a), dim=-1)

  def forward(self, embed, pos, mask, k_all, v_all):
    x = embed
    angles = pos.float().reshape(1, 1) * self.inv_freq.reshape(1, -1)
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos().reshape(1, 1, 1, exportable_module.HEAD_DIM)
    sin = angles.sin().reshape(1, 1, 1, exportable_module.HEAD_DIM)
    one_hot = (
        (self.slots == pos.reshape(1))
        .float()
        .reshape(1, 1, exportable_module.CACHE, 1)
    )

    k_new, v_new = [], []
    for i in range(exportable_module.LAYERS):
      w = (
          lambda name, _i=i: getattr(
              self, f"layers_{_i}_{name}".replace(".", "_")
          )
      )
      h = self._rms(x, w("input_layernorm.weight"))
      q = F.linear(h, w("self_attn.q_proj.weight")).view(
          1, 1, exportable_module.HEADS, exportable_module.HEAD_DIM
      )
      k = F.linear(h, w("self_attn.k_proj.weight")).view(
          1, 1, exportable_module.KV_HEADS, exportable_module.HEAD_DIM
      )
      v = F.linear(h, w("self_attn.v_proj.weight")).view(
          1, 1, exportable_module.KV_HEADS, exportable_module.HEAD_DIM
      )
      q = self._rms(q, w("self_attn.q_norm.weight")).transpose(1, 2)
      k = self._rms(k, w("self_attn.k_norm.weight")).transpose(1, 2)
      v = v.transpose(1, 2)
      q = q * cos + self._rotate_half(q) * sin
      k = k * cos + self._rotate_half(k) * sin

      k_cache = k_all[i] * (1.0 - one_hot) + k * one_hot
      v_cache = v_all[i] * (1.0 - one_hot) + v * one_hot
      k_new.append(k_cache)
      v_new.append(v_cache)

      k_rep = k_cache.repeat_interleave(
          exportable_module.HEADS // exportable_module.KV_HEADS, dim=1
      )
      v_rep = v_cache.repeat_interleave(
          exportable_module.HEADS // exportable_module.KV_HEADS, dim=1
      )
      attn = (
          torch.matmul(q, k_rep.transpose(2, 3))
          * (exportable_module.HEAD_DIM**-0.5)
          + mask
      )
      attn = attn.softmax(dim=-1)
      out = (
          torch.matmul(attn, v_rep)
          .transpose(1, 2)
          .reshape(1, 1, exportable_module.HEADS * exportable_module.HEAD_DIM)
      )
      x = x + F.linear(out, w("self_attn.o_proj.weight"))

      h2 = self._rms(x, w("post_attention_layernorm.weight"))
      gate = F.silu(F.linear(h2, w("mlp.gate_proj.weight")))
      ff = F.linear(
          gate * F.linear(h2, w("mlp.up_proj.weight")),
          w("mlp.down_proj.weight"),
      )
      x = x + ff

    x = self._rms(x, getattr(self, "norm_weight"))
    logits_all = torch.matmul(
        getattr(self, "heads"), x.reshape(1024, 1)
    ).reshape(15, exportable_module.VOCAB)
    return logits_all, torch.stack(k_new), torch.stack(v_new)


class MtpStepGpuTest(absltest.TestCase):

  def test_gpu_rank4_parity_with_5d_reference(self):
    torch.manual_seed(42)
    weights = {}
    for i in range(exportable_module.LAYERS):
      weights[f"layers.{i}.input_layernorm.weight"] = torch.randn(1024)
      weights[f"layers.{i}.self_attn.q_proj.weight"] = (
          torch.randn(
              exportable_module.HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.k_proj.weight"] = (
          torch.randn(
              exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.v_proj.weight"] = (
          torch.randn(
              exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.q_norm.weight"] = torch.randn(
          exportable_module.HEAD_DIM
      )
      weights[f"layers.{i}.self_attn.k_norm.weight"] = torch.randn(
          exportable_module.HEAD_DIM
      )
      weights[f"layers.{i}.self_attn.o_proj.weight"] = (
          torch.randn(
              1024, exportable_module.HEADS * exportable_module.HEAD_DIM
          )
          * 0.02
      )
      weights[f"layers.{i}.post_attention_layernorm.weight"] = torch.randn(1024)
      weights[f"layers.{i}.mlp.gate_proj.weight"] = (
          torch.randn(3072, 1024) * 0.02
      )
      weights[f"layers.{i}.mlp.up_proj.weight"] = (
          torch.randn(3072, 1024) * 0.02
      )
      weights[f"layers.{i}.mlp.down_proj.weight"] = (
          torch.randn(1024, 3072) * 0.02
      )
    weights["norm.weight"] = torch.randn(1024)
    weights["heads"] = (
        torch.randn(15, exportable_module.VOCAB, 1024) * 0.02
    )

    ref_model = MtpStepReference5D(weights).eval()
    gpu_model = exportable_module.MtpStepGpu(weights).eval()

    embed = torch.randn(1, 1, 1024)
    pos = torch.tensor([5], dtype=torch.int32)
    mask = torch.zeros(1, 1, 1, exportable_module.CACHE)
    k_5d = torch.randn(
        exportable_module.LAYERS,
        1,
        exportable_module.KV_HEADS,
        exportable_module.CACHE,
        exportable_module.HEAD_DIM,
    )
    v_5d = torch.randn(
        exportable_module.LAYERS,
        1,
        exportable_module.KV_HEADS,
        exportable_module.CACHE,
        exportable_module.HEAD_DIM,
    )
    k_list = [
        k_5d[i].transpose(1, 2).contiguous()
        for i in range(exportable_module.LAYERS)
    ]
    v_list = [
        v_5d[i].transpose(1, 2).contiguous()
        for i in range(exportable_module.LAYERS)
    ]

    with torch.no_grad():
      logits_ref, k_out_ref, v_out_ref = ref_model(
          embed, pos, mask, k_5d, v_5d
      )
      out_gpu = gpu_model(embed, pos, mask, *k_list, *v_list)
      logits_gpu = out_gpu["logits"]
      k_out_gpu = torch.stack(
          [out_gpu[f"kv_cache_k_{i}"] for i in range(exportable_module.LAYERS)],
          dim=0,
      ).transpose(2, 3)
      v_out_gpu = torch.stack(
          [out_gpu[f"kv_cache_v_{i}"] for i in range(exportable_module.LAYERS)],
          dim=0,
      ).transpose(2, 3)

    # Confirm all GPU output tensors are rank <= 4
    self.assertLessEqual(logits_gpu.ndim, 4)
    for i in range(exportable_module.LAYERS):
      self.assertLessEqual(out_gpu[f"kv_cache_k_{i}"].ndim, 4)
      self.assertEqual(out_gpu[f"kv_cache_k_{i}"].shape[0], 1)
      self.assertLessEqual(out_gpu[f"kv_cache_v_{i}"].ndim, 4)
      self.assertEqual(out_gpu[f"kv_cache_v_{i}"].shape[0], 1)

    # Verify numerical equivalence to high floating-point precision
    torch.testing.assert_close(logits_gpu, logits_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_out_gpu, k_out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_out_gpu, v_out_ref, rtol=1e-5, atol=1e-5)

  def test_mtp_export_tflite(self):
    torch.manual_seed(42)
    weights = {}
    for i in range(exportable_module.LAYERS):
      weights[f"layers.{i}.input_layernorm.weight"] = torch.randn(1024)
      weights[f"layers.{i}.self_attn.q_proj.weight"] = (
          torch.randn(
              exportable_module.HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.k_proj.weight"] = (
          torch.randn(
              exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.v_proj.weight"] = (
          torch.randn(
              exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
          )
          * 0.02
      )
      weights[f"layers.{i}.self_attn.q_norm.weight"] = torch.randn(
          exportable_module.HEAD_DIM
      )
      weights[f"layers.{i}.self_attn.k_norm.weight"] = torch.randn(
          exportable_module.HEAD_DIM
      )
      weights[f"layers.{i}.self_attn.o_proj.weight"] = (
          torch.randn(
              1024, exportable_module.HEADS * exportable_module.HEAD_DIM
          )
          * 0.02
      )
      weights[f"layers.{i}.post_attention_layernorm.weight"] = torch.randn(1024)
      weights[f"layers.{i}.mlp.gate_proj.weight"] = (
          torch.randn(3072, 1024) * 0.02
      )
      weights[f"layers.{i}.mlp.up_proj.weight"] = (
          torch.randn(3072, 1024) * 0.02
      )
      weights[f"layers.{i}.mlp.down_proj.weight"] = (
          torch.randn(1024, 3072) * 0.02
      )
    weights["norm.weight"] = torch.randn(1024)
    weights["heads"] = (
        torch.randn(15, exportable_module.VOCAB, 1024) * 0.02
    )
    gpu_model = exportable_module.MtpStepGpu(weights).eval()
    sample_inputs = (
        torch.zeros(1, 1, 1024, dtype=torch.float32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, 1, 1, 32, dtype=torch.float32),
        *[torch.zeros(1, 32, 8, 128, dtype=torch.float32) for _ in range(5)],
        *[torch.zeros(1, 32, 8, 128, dtype=torch.float32) for _ in range(5)],
    )
    edge_model = litert_torch.convert(gpu_model, sample_inputs)
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = os.path.join(tmp_dir, "mtp_test.tflite")
      edge_model.export(tmp_path)
      interp = tf.lite.Interpreter(model_path=tmp_path)
      interp.allocate_tensors()
      self.assertNotEmpty(interp.get_input_details())
      self.assertNotEmpty(interp.get_output_details())

  def test_speaker_encoder_parity(self):
    torch.manual_seed(42)
    model = speaker_encoder.Qwen3TTSSpeakerEncoder().eval()
    sample_mel = torch.randn(1, 300, 128, dtype=torch.float32)
    with torch.no_grad():
      pt_out = model(sample_mel).numpy()

    edge_model = litert_torch.convert(model, (sample_mel,))
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = os.path.join(tmp_dir, "spk_enc.tflite")
      edge_model.export(tmp_path)
      interp = tf.lite.Interpreter(model_path=tmp_path)
      interp.allocate_tensors()
      input_idx = interp.get_input_details()[0]["index"]
      output_idx = interp.get_output_details()[0]["index"]
      interp.set_tensor(input_idx, sample_mel.numpy())
      interp.invoke()
      tflite_out = interp.get_tensor(output_idx)

    max_diff = np.max(np.abs(pt_out - tflite_out))
    print(f"Speaker Encoder Max Abs Diff: {max_diff:.8e}")
    self.assertLess(max_diff, 1e-4)


if __name__ == "__main__":
  absltest.main()
