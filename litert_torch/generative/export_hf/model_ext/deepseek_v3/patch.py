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
"""Patches for DeepSeek-V3 (incl. Moonlight/Kimi) for on-device deployment.

DeepSeek-V3 specifics handled here:
  - The `noaux_tc` top-k router in HF transformers uses `scatter_`,
    `masked_fill` and group reductions that do not lower to LiteRT ops.
    `LiteRTDeepseekV3TopkRouter` is a numerically equivalent rewrite using
    the gemma4 pattern: sigmoid scores -> `torch.topk` -> `arange` +
    equality-mask arithmetic with fully static shapes.
  - MLA's asymmetric head dims (qk_head_dim != v_head_dim) are handled
    generically in `export_hf/core/cache.py` (K/V cache shapes read
    `qk_head_dim` / `v_head_dim` from the config) and
    `export_hf/core/attention.py` (SDPA output reshape uses the value head
    dim). The naive full-K/V cache is used; latent (kv_lora_rank) caching is
    a later optimization.
  - Experts run through `litert_moe_sequential`
    (`litert_torch/generative/layers/moe.py`), the dense fallback registered
    for `transformers.integrations.moe`; DeepSeek's SiLU activation comes
    from the module's own `act_fn`. The shared experts are a plain dense MLP
    in the HF implementation and need no patching.
"""

import contextlib

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import torch
from torch import nn
import torch.nn.functional as F

try:
  from transformers.models.deepseek_v3 import modeling_deepseek_v3  # pylint: disable=g-import-not-at-top

  class LiteRTDeepseekV3TopkRouter(nn.Module):
    """LiteRT-compatible DeepSeek-V3 `noaux_tc` top-k router.

    Numerically equivalent to
    `transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3TopkRouter`
    but expressed with static shapes and without `scatter_`, `gather`, or
    `masked_fill`, following the gemma4 router pattern
    (`model_ext/gemma4/patch.py`). Parameter and buffer names match the
    original so checkpoints load unchanged.
    """

    def __init__(self, config):
      super().__init__()
      self.top_k = config.num_experts_per_tok
      self.num_experts = config.num_local_experts
      self.hidden_dim = config.hidden_size
      self.weight = nn.Parameter(
          torch.zeros(self.num_experts, self.hidden_dim)
      )
      self.routed_scaling_factor = config.routed_scaling_factor
      self.num_group = config.n_group
      self.topk_group = config.topk_group
      self.norm_topk_prob = config.norm_topk_prob
      self.register_buffer(
          "e_score_correction_bias", torch.zeros(self.num_experts)
      )

    def forward(self, hidden_states):
      hidden_states = hidden_states.view(-1, self.hidden_dim)
      router_logits = F.linear(
          hidden_states.type(torch.float32), self.weight.type(torch.float32)
      )
      scores = router_logits.sigmoid()  # [T, E]
      scores_for_choice = scores + self.e_score_correction_bias

      # Group-limited routing. Moonlight ships with n_group == 1, in which
      # case every expert is a routing candidate and the group stage
      # collapses entirely.
      if self.num_group > 1:
        experts_per_group = self.num_experts // self.num_group
        # Group score: sum of the top-2 candidate scores within each group.
        group_scores = (
            scores_for_choice.view(-1, self.num_group, experts_per_group)
            .topk(min(2, experts_per_group), dim=-1)[0]
            .sum(dim=-1)
        )  # [T, G]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[
            1
        ].int()  # [T, topk_group]
        group_ids = torch.arange(
            self.num_group, dtype=group_idx.dtype, device=group_idx.device
        )
        # arange + equality instead of scatter_: [T, topk_group, G] -> [T, G].
        # Top-k indices are distinct, so the sum is a 0/1 mask.
        group_mask = (
            (group_idx.unsqueeze(-1) == group_ids)
            .to(scores_for_choice.dtype)
            .sum(dim=1)
        )
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, experts_per_group)
            .reshape(-1, self.num_experts)
        )
        # Additive mask instead of masked_fill(-inf); sigmoid scores plus
        # bias are O(1) so -1e9 dominates.
        scores_for_choice = scores_for_choice - (1.0 - score_mask) * 1e9

      topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[
          1
      ].int()  # [T, K]

      # Recover the *un-biased* scores for the selected experts without
      # `gather`: one-hot via arange + equality, then mask-multiply-reduce.
      expert_ids = torch.arange(
          self.num_experts,
          dtype=topk_indices.dtype,
          device=topk_indices.device,
      )
      match_mask = (topk_indices.unsqueeze(-1) == expert_ids).to(
          scores.dtype
      )  # [T, K, E]
      topk_weights = (match_mask * scores.unsqueeze(1)).sum(dim=-1)  # [T, K]

      if self.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
      topk_weights = topk_weights * self.routed_scaling_factor
      return router_logits, topk_weights, topk_indices

  @patches_lib.register_patch(["deepseek_v3"])
  @contextlib.contextmanager
  def deepseek_v3_litert_patch():
    """DeepSeek-V3 patch: swap in the LiteRT-compatible router."""
    print("DeepSeek-V3 patch applied.")
    original_router = modeling_deepseek_v3.DeepseekV3TopkRouter
    modeling_deepseek_v3.DeepseekV3TopkRouter = LiteRTDeepseekV3TopkRouter
    try:
      yield
    finally:
      modeling_deepseek_v3.DeepseekV3TopkRouter = original_router


except ImportError:

  class LiteRTDeepseekV3TopkRouter(nn.Module):  # pyrefly: ignore[redefinition]
    pass

  @patches_lib.register_patch(["deepseek_v3"])
  @contextlib.contextmanager
  def deepseek_v3_litert_patch():
    yield
