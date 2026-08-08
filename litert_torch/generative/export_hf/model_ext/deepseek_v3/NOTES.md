# deepseek_v3 model_ext — status notes (Kimi/Moonlight port, M1)

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

## Known limitations / out of M1 scope

- **No `.tflite`/`.litertlm` conversion yet** — that is M2 (toy 2-layer
  export with `litert_moe_sequential` first, then the `moe` custom op
  experiment; the custom op currently hardcodes gelu + renormalized weights
  and fp32/int8 only, while DeepSeek needs SiLU + sigmoid-scaled
  non-renormalized weights).
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

## Environment used for the parity run

Python 3.11, `torch==2.12.0`, `transformers==5.14.1` (has native
`deepseek_v3` with the modern `DeepseekV3Experts` 3D-weight layout +
`use_experts_implementation` dispatch), fp32 CPU.

Test command:

```
pytest litert_torch/generative/export_hf/model_ext/deepseek_v3/patch_test.py
```
