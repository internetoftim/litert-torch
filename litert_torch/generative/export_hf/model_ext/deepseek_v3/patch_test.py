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
"""Parity tests for the DeepSeek-V3 (Moonlight/Kimi) export patches.

Compares the patched export-path model (LiteRT router + sequential experts +
`lrt_transposed_attention` + `LiteRTLMCache`) against the unpatched
`transformers` DeepseekV3ForCausalLM reference on tiny random-weight configs,
in fp32 on CPU. Covers prefill and single-token decode with cache.
"""

import copy

import torch
from transformers.models.deepseek_v3 import modeling_deepseek_v3

import litert_torch.generative.export_hf  # pylint: disable=unused-import  # registers litert_moe fns.
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.deepseek_v3 import patch
from litert_torch.generative.export_hf.model_ext.deepseek_v3 import test_utils
from absl.testing import parameterized

from absl.testing import absltest as googletest


def _dense_router_weights(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
  """Scatters (weights, indices) into a dense [T, E] tensor for comparison.

  The LiteRT router may order its top-k differently than the HF router
  (sorted vs unsorted topk), which is irrelevant downstream since expert
  outputs are weight-summed. Comparing the dense per-expert weights makes the
  parity check order-invariant.
  """
  dense = torch.zeros(
      (topk_weights.shape[0], num_experts), dtype=topk_weights.dtype
  )
  dense.scatter_add_(1, topk_indices.long(), topk_weights)
  return dense


class RouterParityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("moonlight_n_group_1", 1, 1),
      ("grouped_n_group_4_topk_group_2", 4, 2),
  )
  def test_router_parity(self, n_group, topk_group):
    torch.manual_seed(0)
    config = test_utils.tiny_deepseek_v3_config(
        n_group=n_group, topk_group=topk_group
    )

    original = modeling_deepseek_v3.DeepseekV3TopkRouter(config)
    with torch.no_grad():
      original.weight.normal_(0.0, 1.0)
      original.e_score_correction_bias.normal_(0.0, 0.5)

    litert = patch.LiteRTDeepseekV3TopkRouter(config)
    litert.load_state_dict(original.state_dict())

    hidden_states = torch.randn(7, config.hidden_size)
    with torch.no_grad():
      ref_logits, ref_weights, ref_indices = original(hidden_states)
      lrt_logits, lrt_weights, lrt_indices = litert(hidden_states)

    self.assertTrue(torch.allclose(ref_logits, lrt_logits, atol=1e-6))
    ref_dense = _dense_router_weights(
        ref_weights, ref_indices, config.num_local_experts
    )
    lrt_dense = _dense_router_weights(
        lrt_weights, lrt_indices, config.num_local_experts
    )
    self.assertTrue(
        torch.allclose(ref_dense, lrt_dense, atol=1e-6),
        "Router per-expert weight mismatch.\n"
        f"Expected: {ref_dense}\nActual: {lrt_dense}",
    )
    # The set of selected experts must match per token.
    self.assertTrue(
        torch.equal(
            ref_indices.long().sort(dim=-1).values,
            lrt_indices.long().sort(dim=-1).values,
        )
    )


class MoeBlockParityTest(googletest.TestCase):

  def test_moe_block_parity_with_sequential_experts(self):
    torch.manual_seed(1)
    ref_config = test_utils.tiny_deepseek_v3_config()
    ref_config._experts_implementation = None  # pylint: disable=protected-access
    ref_moe = modeling_deepseek_v3.DeepseekV3MoE(ref_config)
    with torch.no_grad():
      for p in ref_moe.parameters():
        p.normal_(0.0, 0.2)
      ref_moe.gate.e_score_correction_bias.normal_(0.0, 0.5)
    ref_moe.eval()

    lrt_config = copy.deepcopy(ref_config)
    lrt_config._experts_implementation = "litert_moe_sequential"  # pylint: disable=protected-access
    with patches_lib.get_patch_context("deepseek_v3"):
      lrt_moe = modeling_deepseek_v3.DeepseekV3MoE(lrt_config)
    self.assertIsInstance(lrt_moe.gate, patch.LiteRTDeepseekV3TopkRouter)
    lrt_moe.load_state_dict(ref_moe.state_dict())
    export_lib.pre_split_model_experts(lrt_moe)
    lrt_moe.eval()

    hidden_states = torch.randn(1, 6, ref_config.hidden_size)
    with torch.no_grad():
      ref_out = ref_moe(hidden_states)
      lrt_out = lrt_moe(hidden_states)

    self.assertTrue(
        torch.allclose(ref_out, lrt_out, atol=1e-5),
        "MoE block output mismatch. Max diff:"
        f" {(ref_out - lrt_out).abs().max()}",
    )


class ModelParityTest(googletest.TestCase):
  """End-to-end prefill + decode parity on a tiny random-weight model."""

  CACHE_LENGTH = 40
  PREFILL_LENGTH = 8

  def _build_models(self):
    torch.manual_seed(2)
    ref_config = test_utils.tiny_deepseek_v3_config()
    ref_config._attn_implementation = "eager"  # pylint: disable=protected-access
    ref_config._experts_implementation = None  # pylint: disable=protected-access
    ref_model = modeling_deepseek_v3.DeepseekV3ForCausalLM(ref_config)
    ref_model.eval()

    lrt_config = copy.deepcopy(ref_config)
    lrt_config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access
    lrt_config._experts_implementation = "litert_moe_sequential"  # pylint: disable=protected-access
    with patches_lib.get_patch_context("deepseek_v3"):
      lrt_model = modeling_deepseek_v3.DeepseekV3ForCausalLM(lrt_config)
    lrt_model.load_state_dict(ref_model.state_dict())
    export_lib.pre_split_model_experts(lrt_model)
    lrt_model.eval()
    return ref_model, lrt_model

  def _export_config(self):
    return exportable_module_config.ExportableModuleConfig(
        model="deepseek_v3",
        batch_size=1,
        cache_length=self.CACHE_LENGTH,
        prefill_lengths=[self.PREFILL_LENGTH],
        cache_implementation="LiteRTLMCache",
        k_ts_idx=2,
        v_ts_idx=3,
        moe_exports_implementation="litert_moe_sequential",
    )

  def test_cache_shapes_use_asymmetric_head_dims(self):
    _, lrt_model = self._build_models()
    export_config = self._export_config()
    gen_mod = exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate(
        lrt_model, export_config
    )
    kv_cache = gen_mod.get_sample_inputs(lrt_model.config)["decode"][0][
        "kv_cache"
    ]
    config = lrt_model.config
    num_kv_heads = config.num_key_value_heads
    for layer in kv_cache.layers:
      self.assertEqual(
          layer.keys.shape,
          (1, num_kv_heads, self.CACHE_LENGTH, config.qk_head_dim),
      )
      self.assertEqual(
          layer.values.shape,
          (1, num_kv_heads, config.v_head_dim, self.CACHE_LENGTH),
      )

  def test_prefill_and_decode_logits_parity(self):
    ref_model, lrt_model = self._build_models()
    export_config = self._export_config()

    prefill_mod = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            lrt_model, export_config
        )
    )
    gen_mod = exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate(
        lrt_model, export_config
    )

    vocab_size = lrt_model.config.vocab_size
    torch.manual_seed(3)
    # Avoid token id 0: it is the pad id used to build the valid mask.
    prompt = torch.randint(
        1, vocab_size, (1, self.PREFILL_LENGTH), dtype=torch.int32
    )
    next_token = torch.randint(1, vocab_size, (1, 1), dtype=torch.int32)
    full_ids = torch.cat([prompt, next_token], dim=1).long()

    with torch.no_grad():
      ref_logits = ref_model(input_ids=full_ids).logits  # [1, T+1, V]

    # --- Path A: Generate module over the whole prompt (per-position
    # prefill logits parity).
    kv_cache_a = gen_mod.get_sample_inputs(lrt_model.config)["decode"][0][
        "kv_cache"
    ]
    input_pos = torch.arange(self.PREFILL_LENGTH, dtype=torch.int32)
    mask = test_utils.create_causal_mask(
        self.PREFILL_LENGTH, self.CACHE_LENGTH, input_pos
    )
    with torch.no_grad():
      out_a = gen_mod(
          tokens=prompt, input_pos=input_pos, kv_cache=kv_cache_a, mask=mask
      )
    self.assertTrue(
        torch.allclose(
            out_a["logits"], ref_logits[:, : self.PREFILL_LENGTH], atol=1e-4
        ),
        "Prefill logits mismatch. Max diff:"
        f" {(out_a['logits'] - ref_logits[:, : self.PREFILL_LENGTH]).abs().max()}",
    )

    # --- Path B: Prefill module on the prompt, then a single-token decode
    # step with the populated cache.
    kv_cache_b = prefill_mod.get_sample_inputs(lrt_model.config)[
        f"prefill_{self.PREFILL_LENGTH}"
    ][0]["kv_cache"]
    with torch.no_grad():
      prefill_out = prefill_mod(
          tokens=prompt, input_pos=input_pos, kv_cache=kv_cache_b, mask=mask
      )
    kv_cache_b = prefill_out["kv_cache"]

    decode_pos = torch.tensor([self.PREFILL_LENGTH], dtype=torch.int32)
    decode_mask = test_utils.create_causal_mask(1, self.CACHE_LENGTH, decode_pos)
    with torch.no_grad():
      decode_out = gen_mod(
          tokens=next_token,
          input_pos=decode_pos,
          kv_cache=kv_cache_b,
          mask=decode_mask,
      )
    self.assertTrue(
        torch.allclose(
            decode_out["logits"][:, -1], ref_logits[:, -1], atol=1e-4
        ),
        "Decode logits mismatch. Max diff:"
        f" {(decode_out['logits'][:, -1] - ref_logits[:, -1]).abs().max()}",
    )


if __name__ == "__main__":
  googletest.main()
