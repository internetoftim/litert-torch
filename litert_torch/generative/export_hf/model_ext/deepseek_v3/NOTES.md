# deepseek_v3 model_ext — status notes (Kimi/Moonlight port, M1+M2)

Target model: `moonshotai/Moonlight-16B-A3B-Instruct` (HF `model_type`
`deepseek_v3`, `DeepseekV3ForCausalLM`). Verified against the actual hub
config: `n_group=1`, `topk_group=1`, `num_experts_per_tok=6`,
`n_routed_experts=64`, `n_shared_experts=2`, `routed_scaling_factor=2.446`,
`norm_topk_prob=True`, `first_k_dense_replace=1`, `q_lora_rank=None`,
`kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`
(`qk_head_dim=192`), `v_head_dim=128`, `head_dim=64` (rotary dim only!),
`rope_interleave=True`, standard RoPE (`rope_theta=50000`),
`scoring_func=sigmoid`, `topk_method=noaux_tc`.

## What is implemented (M1)

- `patch.py`: `LiteRTDeepseekV3TopkRouter` — static-shape rewrite of the
  `noaux_tc` router (no `scatter_`/`gather`/`masked_fill`), registered via
  `patches_lib.register_patch(["deepseek_v3"])`. Handles both `n_group == 1`
  (Moonlight; group stage collapses) and grouped routing.
- `export_hf/core/cache.py`: K/V cache shapes honor asymmetric head dims —
  `_infer_cache_shape_from_config` reads `qk_head_dim`/`v_head_dim` when both
  are present on the config (precedent: `global_head_dim`), and
  `LiteRTLMCacheLayer` derives `k_head_dim` from the K cache shape instead of
  reusing the V head dim. Naive full-K/V cache; latent (`kv_lora_rank`)
  caching is deliberately out of scope for M1 (planned as an M3
  optimization).
- `export_hf/core/attention.py`: `transposed_attention` reshapes the SDPA
  output using the value head dim (from the output tensor shape) instead of
  the query head dim. No-op for models with symmetric head dims.
- Experts run through `litert_moe_sequential` (dense fallback in
  `generative/layers/moe.py`) + `export_lib.pre_split_model_experts`. The
  HF `DeepseekV3Experts` weight layout (`gate_up_proj [E, 2I, H]`,
  `down_proj [E, H, I]`, `act_fn` = SiLU from `ACT2FN`) is directly
  compatible — no expert patching needed. Shared experts are a plain dense
  MLP and need no patching.
- `patch_test.py`: parity vs unpatched `transformers`
  `DeepseekV3ForCausalLM` on tiny random-weight configs (fp32 CPU,
  atol 1e-4 for logits): router unit parity (n_group=1 and grouped), MoE
  block parity, cache-shape checks, and full-model prefill + single-token
  decode-with-cache logits parity through
  `LiteRTExportableModuleForDecoderOnlyLM{Prefill,Generate}` +
  `LiteRTLMCache`.

## Verified beyond the M1 gate

- `torch.export.export` succeeds for both the prefill and decode exportable
  modules on the tiny config (strict mode, static shapes).
- `run_decompositions(torch_tfl.decomps)` also succeeds; the residual
  non-`tfl.*` ops in the decode graph (`aten.sigmoid`, `aten.eq.Tensor`,
  `aten.sum.dim_IntList`, `aten._to_copy`, `aten.clone`,
  `aten.scalar_tensor`, `litert_torch.bmm_4d`,
  `litert_torch.dynamic_update_slice`) are the same categories other
  supported models leave for the later lowering stages.

## What is implemented (M2)

- **Working toy `.tflite`** through `export_lib`'s real converter path
  (`converter_utils.Converter` + `add_signature` + `convert`, i.e.
  `export_lib.export_text_prefill_decode_model`), on the tiny random-weight
  config with `moe_exports_implementation="litert_moe_sequential"`. One
  flatbuffer, two signatures: `prefill_8` (outputs the updated K/V caches
  only — no logits) and `decode` (caches + logits). ~5.3 MB fp32; converts
  in seconds on CPU. **No converter blockers** — no missing op lowerings,
  no crash on the MoE subgraph.
- `convert_test.py`: converts once, then verifies with the
  `ai_edge_litert` interpreter against the *unpatched eager*
  `DeepseekV3ForCausalLM` reference (fp32 CPU). Skips gracefully when the
  converter deps (`ai-edge-litert`, `ai-edge-quantizer`, `litert-converter`)
  are not installed. Measured on the toy config:
  - fp32: decode-step logits max abs diff **4.8e-7**; prefill K/V cache
    (filled region) max abs diff **≤3.6e-7** per layer. (The unfilled cache
    region is don't-care — it keeps whatever the input cache held.)
  - int8 `dynamic_wi8_afp32` (the repo's standard dynamic-range recipe,
    default of `ExportableModuleConfig.quantization_recipe`): same decode
    step max abs diff **2.5e-2**; model shrinks 5.08 → 1.60 MiB (3.2x).
    Expected quantization error, quantified only.
- `test_utils.py`: shared tiny-config + causal-mask helpers used by both
  `patch_test.py` and `convert_test.py`.

## M2 `moe` custom-op probe (go/no-go input for M4)

Probed with `moe_exports_implementation="litert_moe"` on the same toy config
(ai-edge-litert-nightly 2.2.0.dev20260807, macOS arm64):

- **Conversion works** once one repo-side gap is patched:
  `litert_moe_experts_forward` (`generative/layers/moe.py` line ~452) reads
  `self.config.top_k_experts` (gemma4 naming); `DeepseekV3Config` calls it
  `num_experts_per_tok`, so an alias (`config.top_k_experts =
  config.num_experts_per_tok`) is needed. The HF `DeepseekV3Experts` tensor
  layout (`gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`) is directly
  compatible with `flatten_expert_weight`.
- **The runtime kernel exists and runs** — the classic
  `ai_edge_litert.interpreter.Interpreter` resolves and executes the `moe`
  custom op (fp32 CPU/XNNPACK path in `libLiteRt.dylib`), and
  `CompiledModel` loads it too. It is NOT closed: source is public in
  `google-ai-edge/LiteRT`:
  - CPU: `tflite/delegates/xnnpack/moe_delegate_kernel.cc` — fp32 weights
    only, **rejects any `activation` other than `'gelu'` at prepare time**
    (verified empirically: binary-patching the flexbuffer to
    `activation='silu'` fails with `moe node #149 only supports
    activation='gelu'`), and does **not** read `renormalized_top_weights`
    at all — top weights are used exactly as passed.
  - GPU (ML Drift): `ml_drift_delegate/delegate/composite/
    moe_experts_parser.cc` — also gelu-only, additionally *requires*
    `renormalized_top_weights=true`, and accepts
    `weight_type ∈ {fp32, int8, int4}` (an int4 path exists here, revising
    M0's "no int4"; the authoring wrapper in `moe.py` still only emits
    fp32/int8).
- **Numerics confirm the gelu semantics**: toy decode logits vs the eager
  SiLU reference differ by **2.4e-2**, but vs a GELU(tanh)-substituted
  reference by **1.7e-5** — i.e. the CPU kernel computes
  `gelu_tanh(gate) * up` and applies our sigmoid-scaled, non-renormalized
  top weights (incl. `routed_scaling_factor`) unchanged. So for DeepSeek the
  *only* semantic gap on the CPU path is the activation function.
- **What would need to change for DeepSeek semantics:**
  1. LiteRT kernels (upstream `google-ai-edge/LiteRT`): accept
     `activation='silu'` in `moe_delegate_kernel.cc` (CPU) and
     `moe_experts_parser.cc`/kernel (GPU); the GPU path must also either
     honor `renormalized_top_weights=false` or drop the hard requirement.
  2. This repo: parameterize `"activation"` and `"renormalized_top_weights"`
     in `_moe_custom_options` (`generative/layers/moe.py` lines ~141-151),
     thread them through `moe_experts`/`litert_moe_experts_forward` from the
     experts module's `act_fn`/config, fix the `top_k_experts` vs
     `num_experts_per_tok` naming, and use the real activation in
     `_moe_experts_reference` (currently hardcoded
     `F.gelu(approximate="tanh")`, line ~125).
  Until the upstream kernel change lands, the shipping path for DeepSeek
  remains `litert_moe_sequential` (correct but dense → 2-4 tok/s class
  unpruned) or a REAP-pruned model on the same dense path.

## Known limitations / out of M1+M2 scope

- **No `.litertlm` bundling / on-device run yet** — M2 produced and verified
  the raw `.tflite` on host CPU only; tokenizer/bundle plumbing and the
  full-size convert are M3.
- **Latent (rank-512) KV caching not implemented** — the naive cache costs
  ~8.9x the latent size (~1.13 GB fp16 at 4K context for Moonlight-16B).
- **Split-cache variant untested** for deepseek_v3
  (`lrt_split_cache_attention`); only the `LiteRTLMCache` +
  `lrt_transposed_attention` path is covered.
- Router top-k tie-breaking can differ from HF when two experts score
  exactly equally (measure-zero with real weights); the parity tests compare
  order-invariant per-expert dense weights.
- `sdpa_use_composite` / `apply_gpu_composites` paths get the same
  value-head-dim reshape fix but have not been exercised with asymmetric
  head dims.

## Environment used for the parity + conversion runs

Python 3.11, `torch==2.12.0`, `transformers==5.14.1` (has native
`deepseek_v3` with the modern `DeepseekV3Experts` 3D-weight layout +
`use_experts_implementation` dispatch), fp32 CPU. Conversion additionally:
`ai-edge-litert-nightly==2.2.0.dev20260807`,
`ai-edge-quantizer-nightly==0.9.0.dev20260808`, `litert-converter==0.3.0`
(macOS arm64 wheels).

Test commands:

```
pytest litert_torch/generative/export_hf/model_ext/deepseek_v3/patch_test.py
pytest litert_torch/generative/export_hf/model_ext/deepseek_v3/convert_test.py
```
