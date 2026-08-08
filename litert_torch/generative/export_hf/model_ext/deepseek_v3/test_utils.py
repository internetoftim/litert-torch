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
"""Shared test helpers for the DeepSeek-V3 (Moonlight/Kimi) model_ext tests."""

import torch
from transformers.models.deepseek_v3 import modeling_deepseek_v3


def tiny_deepseek_v3_config(**overrides):
  """Tiny random-weight DeepSeek-V3 config.

  Mirrors the Moonlight-16B-A3B architecture knobs at toy scale, keeping the
  properties that matter for export parity: asymmetric head dims
  (qk_head_dim != v_head_dim), MLA kv down-projection, one dense layer
  followed by one MoE layer, sigmoid `noaux_tc` routing with n_group=1, two
  shared experts, and interleaved RoPE.
  """
  kwargs = dict(
      vocab_size=256,
      hidden_size=256,
      intermediate_size=512,
      moe_intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=4,
      head_dim=16,  # Rotary dim; equals qk_rope_head_dim as in Moonlight.
      qk_nope_head_dim=32,
      qk_rope_head_dim=16,
      v_head_dim=32,
      kv_lora_rank=64,
      q_lora_rank=None,
      n_routed_experts=8,
      num_experts_per_tok=3,
      n_shared_experts=2,
      n_group=1,
      topk_group=1,
      norm_topk_prob=True,
      routed_scaling_factor=2.5,
      first_k_dense_replace=1,
      hidden_act="silu",
      rope_interleave=True,
      rms_norm_eps=1e-5,
      attention_bias=False,
      attention_dropout=0.0,
      max_position_embeddings=128,
      rope_theta=50000.0,
      pad_token_id=0,
      tie_word_embeddings=False,
      use_cache=True,
  )
  kwargs.update(overrides)
  return modeling_deepseek_v3.DeepseekV3Config(**kwargs)


def create_causal_mask(
    seq_len: int, cache_length: int, input_pos: torch.Tensor
) -> torch.Tensor:
  """Float causal mask of shape (1, 1, seq_len, cache_length)."""
  cache_positions = torch.arange(cache_length).view(1, 1, 1, cache_length)
  q_pos = input_pos.view(1, 1, seq_len, 1)
  causal_bool = cache_positions <= q_pos
  return torch.where(
      causal_bool,
      torch.zeros((1,), dtype=torch.float32),
      torch.full((1,), -1e38, dtype=torch.float32),
  )
