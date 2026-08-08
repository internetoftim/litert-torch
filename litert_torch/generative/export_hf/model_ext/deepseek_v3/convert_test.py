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
"""End-to-end `.tflite` conversion + numeric verification for DeepSeek-V3.

M2 of the Moonlight/Kimi port: drives `export_lib`'s real converter path
(`converter_utils.Converter` + `add_signature`) on a tiny random-weight
DeepSeek-V3 config with `litert_moe_sequential` experts, producing a single
`.tflite` with prefill + decode signatures, then verifies it numerically with
the LiteRT interpreter against the unpatched eager `transformers`
`DeepseekV3ForCausalLM` reference:

  - fp32: prefill K/V cache parity (filled region) and one cached decode
    step's logits, atol 1e-4.
  - int8 (`dynamic_wi8_afp32`, the repo's standard dynamic-range recipe):
    the same decode step; quantization error is expected and only bounded
    loosely — the measured max-abs-diff is logged.

Skips gracefully when the converter/interpreter deps (`ai_edge_litert`,
`ai_edge_quantizer`, `litert-converter`) are not installed.
"""

import os
import tempfile

import numpy as np
import torch

from absl.testing import absltest as googletest

try:
  # export_lib transitively requires ai_edge_litert + ai_edge_quantizer; the
  # interpreter is needed to execute the converted model.
  from ai_edge_litert import interpreter as interpreter_lib
  import litert_torch.generative.export_hf  # pylint: disable=unused-import  # registers litert_moe fns.
  from litert_torch.generative.export_hf.core import export_lib
  from litert_torch.generative.export_hf.core import exportable_module
  from litert_torch.generative.export_hf.core import exportable_module_config
  from litert_torch.generative.export_hf.model_ext import patches as patches_lib
  from litert_torch.generative.export_hf.model_ext.deepseek_v3 import test_utils
  from transformers.models.deepseek_v3 import modeling_deepseek_v3

  _CONVERTER_DEPS_AVAILABLE = True
  _IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - environment dependent.
  _CONVERTER_DEPS_AVAILABLE = False
  _IMPORT_ERROR = e

_CACHE_LENGTH = 40
_PREFILL_LENGTH = 8
_QUANTIZATION_RECIPE = "dynamic_wi8_afp32"


def _build_patched_model(config, state_dict):
  """Builds the export-path model (patched router + sequential experts)."""
  config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access
  config._experts_implementation = "litert_moe_sequential"  # pylint: disable=protected-access
  with patches_lib.get_patch_context("deepseek_v3"):
    model = modeling_deepseek_v3.DeepseekV3ForCausalLM(config)
  model.load_state_dict(state_dict)
  model.eval()
  return export_lib.pre_split_model_experts(model)


def _export_config(work_dir):
  return exportable_module_config.ExportableModuleConfig(
      model="deepseek_v3",
      batch_size=1,
      cache_length=_CACHE_LENGTH,
      prefill_lengths=[_PREFILL_LENGTH],
      cache_implementation="LiteRTLMCache",
      k_ts_idx=2,
      v_ts_idx=3,
      moe_exports_implementation="litert_moe_sequential",
      quantization_recipe=None,  # fp32 first; quantized separately.
      use_random_weights=True,
      work_dir=work_dir,
  )


@googletest.skipIf(
    not _CONVERTER_DEPS_AVAILABLE,
    f"Converter deps unavailable: {_IMPORT_ERROR}",
)
class ConvertAndVerifyTest(googletest.TestCase):
  """Converts the toy model once, then verifies fp32 and int8 variants."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch.manual_seed(2)
    cls.work_dir = tempfile.mkdtemp(prefix="deepseek_v3_toy_export_")

    # Eager unpatched reference; the export-path model loads its weights.
    ref_config = test_utils.tiny_deepseek_v3_config()
    ref_config._attn_implementation = "eager"  # pylint: disable=protected-access
    ref_config._experts_implementation = None  # pylint: disable=protected-access
    cls.ref_model = modeling_deepseek_v3.DeepseekV3ForCausalLM(ref_config)
    cls.ref_model.eval()

    cls.config = test_utils.tiny_deepseek_v3_config()
    cls.model = _build_patched_model(cls.config, cls.ref_model.state_dict())
    cls.export_cfg = _export_config(cls.work_dir)

    artifacts = export_lib.SourceModelArtifacts(
        model=cls.model,
        model_config=cls.config,
        text_model_config=cls.config,
        tokenizer=None,
    )
    export_cfg = export_lib.update_export_config(cls.export_cfg, artifacts)
    exported = export_lib.export_text_prefill_decode_model(
        artifacts, export_cfg, export_lib.ExportedModelArtifacts()
    )
    cls.fp32_tflite = exported.prefill_decode_model_path
    cls.quantized_tflite = export_lib.maybe_quantize_model(
        cls.fp32_tflite, _QUANTIZATION_RECIPE
    )

    # Shared prompt / positions / masks.
    torch.manual_seed(3)
    cls.prompt = torch.randint(
        1, cls.config.vocab_size, (1, _PREFILL_LENGTH), dtype=torch.int32
    )
    cls.next_token = torch.randint(
        1, cls.config.vocab_size, (1, 1), dtype=torch.int32
    )
    cls.prefill_pos = torch.arange(_PREFILL_LENGTH, dtype=torch.int32)
    cls.prefill_mask = test_utils.create_causal_mask(
        _PREFILL_LENGTH, _CACHE_LENGTH, cls.prefill_pos
    )
    cls.decode_pos = torch.tensor([_PREFILL_LENGTH], dtype=torch.int32)
    cls.decode_mask = test_utils.create_causal_mask(
        1, _CACHE_LENGTH, cls.decode_pos
    )
    full_ids = torch.cat([cls.prompt, cls.next_token], dim=1).long()
    with torch.no_grad():
      cls.ref_logits = cls.ref_model(input_ids=full_ids).logits

  def _run_tflite_prefill_decode(self, tflite_path):
    """Runs prefill then one decode step; returns (prefill_out, decode_out)."""
    interpreter = interpreter_lib.Interpreter(tflite_path)
    signatures = interpreter.get_signature_list()
    self.assertIn(f"prefill_{_PREFILL_LENGTH}", signatures)
    self.assertIn("decode", signatures)
    prefill_runner = interpreter.get_signature_runner(
        f"prefill_{_PREFILL_LENGTH}"
    )
    decode_runner = interpreter.get_signature_runner("decode")

    zero_kv = {}
    for name, detail in prefill_runner.get_input_details().items():
      if name.startswith("kv_cache"):
        zero_kv[name] = np.zeros(detail["shape"], dtype=np.float32)

    prefill_out = prefill_runner(
        tokens=self.prompt.numpy(),
        input_pos=self.prefill_pos.numpy(),
        mask=self.prefill_mask.numpy(),
        **zero_kv,
    )
    decode_out = decode_runner(
        tokens=self.next_token.numpy(),
        input_pos=self.decode_pos.numpy(),
        mask=self.decode_mask.numpy(),
        **{name: prefill_out[name] for name in zero_kv},
    )
    return prefill_out, decode_out

  def test_fp32_decode_logits_match_eager_reference(self):
    _, decode_out = self._run_tflite_prefill_decode(self.fp32_tflite)
    diff = np.abs(
        decode_out["logits"][:, -1] - self.ref_logits[:, -1].numpy()
    ).max()
    print(f"[fp32] decode logits max abs diff vs eager HF: {diff:.3e}")
    self.assertLess(diff, 1e-4)

  def test_fp32_prefill_cache_matches_torch_export_modules(self):
    prefill_mod = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            self.model, self.export_cfg
        )
    )
    kv_cache = prefill_mod.get_sample_inputs(self.config)[
        f"prefill_{_PREFILL_LENGTH}"
    ][0]["kv_cache"]
    with torch.no_grad():
      torch_out = prefill_mod(
          tokens=self.prompt,
          input_pos=self.prefill_pos,
          kv_cache=kv_cache,
          mask=self.prefill_mask,
      )
    prefill_out, _ = self._run_tflite_prefill_decode(self.fp32_tflite)
    # Only the first _PREFILL_LENGTH positions are written by prefill; the
    # rest keeps whatever the input cache held (zeros for tflite, the sample
    # values for the torch module), so compare the filled region only.
    # K layout: [B, H, cache_len, qk_head_dim] (k_ts_idx=2);
    # V layout: [B, H, v_head_dim, cache_len] (v_ts_idx=3).
    for i, layer in enumerate(torch_out["kv_cache"].layers):
      k_diff = np.abs(
          prefill_out[f"kv_cache_k_{i}"][:, :, :_PREFILL_LENGTH, :]
          - layer.keys.numpy()[:, :, :_PREFILL_LENGTH, :]
      ).max()
      v_diff = np.abs(
          prefill_out[f"kv_cache_v_{i}"][:, :, :, :_PREFILL_LENGTH]
          - layer.values.numpy()[:, :, :, :_PREFILL_LENGTH]
      ).max()
      print(
          f"[fp32] layer {i} prefill cache max abs diff:"
          f" k={k_diff:.3e} v={v_diff:.3e}"
      )
      self.assertLess(k_diff, 1e-4)
      self.assertLess(v_diff, 1e-4)

  def test_quantized_decode_logits_close_to_eager_reference(self):
    self.assertTrue(os.path.exists(self.quantized_tflite))
    self.assertLess(
        os.path.getsize(self.quantized_tflite),
        os.path.getsize(self.fp32_tflite),
    )
    _, decode_out = self._run_tflite_prefill_decode(self.quantized_tflite)
    logits = decode_out["logits"][:, -1]
    self.assertTrue(np.isfinite(logits).all())
    diff = np.abs(logits - self.ref_logits[:, -1].numpy()).max()
    print(
        f"[{_QUANTIZATION_RECIPE}] decode logits max abs diff vs eager HF:"
        f" {diff:.3e}"
    )
    # Quantization error is expected; this only guards against gross
    # divergence (measured ~2.5e-2 on this toy config).
    self.assertLess(diff, 0.5)


if __name__ == "__main__":
  googletest.main()
