"""Full-model reference logits via layer-streamed eager forward.

Never holds the whole model: builds the transformers model on meta device,
then for each layer loads its weights fp32 from safetensors, runs the layer,
frees the weights. Produces decode-step logits for the same fixed prompt used
by ref_logits.py (seed 7) or a supplied token list.

RAM: ~1 layer (0.6 GB fp32 params) + activations. Works for 27 layers.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from safetensors import safe_open  # noqa: E402
from transformers.models.deepseek_v3 import modeling_deepseek_v3  # noqa: E402


def load_keys_for(prefix, model_dir, index, views):
  """Loads all checkpoint keys under prefix into a dict of fp32 tensors."""
  out = {}
  by_shard = {}
  for k, shard in index["weight_map"].items():
    if k.startswith(prefix):
      by_shard.setdefault(shard, []).append(k)
  for shard, keys in by_shard.items():
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
      for k in keys:
        out[k] = f.get_tensor(k).to(torch.float32)
  return out


def materialize(module, own_prefix, weights):
  """Loads weights (checkpoint naming) into module, converting experts."""
  sd = {}
  experts = {}
  for k, t in weights.items():
    rk = k[len(own_prefix):]
    m = re.match(
        r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", rk
    )
    if m:
      experts.setdefault(int(m.group(1)), {})[m.group(2)] = t
    elif rk.endswith("rotary_emb.inv_freq"):
      continue
    else:
      sd[rk] = t
  if experts:
    n = len(experts)
    gu = torch.stack(
        [
            torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]], dim=0)
            for i in range(n)
        ]
    )
    dn = torch.stack([experts[i]["down_proj"] for i in range(n)])
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn
  missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
  # inv_freq-style non-persistent buffers are not in sd; anything else
  # missing is a bug.
  real_missing = [k for k in missing if "inv_freq" not in k]
  if real_missing or unexpected:
    raise RuntimeError(f"missing={real_missing} unexpected={unexpected}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model-dir", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--prefill-len", type=int, default=8)
  p.add_argument("--ids", default=None,
                 help="comma-separated token ids incl. final decode token; "
                 "default: seed-7 random prompt like ref_logits.py")
  args = p.parse_args()

  config = modeling_deepseek_v3.DeepseekV3Config.from_pretrained(
      args.model_dir
  )
  config._attn_implementation = "eager"
  config._experts_implementation = None
  index = json.load(
      open(os.path.join(args.model_dir, "model.safetensors.index.json"))
  )

  if args.ids:
    full_ids = torch.tensor(
        [[int(x) for x in args.ids.split(",")]], dtype=torch.long
    )
    prompt = full_ids[:, :-1].to(torch.int32)
    next_token = full_ids[:, -1:].to(torch.int32)
  else:
    torch.manual_seed(7)
    n = args.prefill_len
    prompt = torch.randint(1, config.vocab_size, (1, n), dtype=torch.int32)
    next_token = torch.randint(
        1, config.vocab_size, (1, 1), dtype=torch.int32
    )
    full_ids = torch.cat([prompt, next_token], dim=1).long()

  with torch.device("meta"):
    model = modeling_deepseek_v3.DeepseekV3ForCausalLM(config)
  model.eval()

  S = full_ids.shape[1]
  position_ids = torch.arange(S).unsqueeze(0)
  # Causal float mask [1,1,S,S].
  causal = torch.triu(
      torch.full((S, S), torch.finfo(torch.float32).min), diagonal=1
  )[None, None]

  # Rotary embeddings (shared module).
  rot = modeling_deepseek_v3.DeepseekV3RotaryEmbedding(config=config)
  dummy = torch.zeros(1, S, config.hidden_size)
  cos, sin = rot(dummy, position_ids)

  # Embedding.
  w = load_keys_for("model.embed_tokens.", args.model_dir, index, None)
  emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)
  emb.load_state_dict({"weight": w["model.embed_tokens.weight"]},
                      assign=True)
  h = emb(full_ids)
  del emb, w

  ks, vs = [], []
  for i in range(config.num_hidden_layers):
    with torch.device("meta"):
      layer = modeling_deepseek_v3.DeepseekV3DecoderLayer(config, i)
    layer.eval()
    prefix = f"model.layers.{i}."
    weights = load_keys_for(prefix, args.model_dir, index, None)
    materialize(layer, prefix, weights)
    del weights
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=config)
    with torch.no_grad():
      out = layer(
          h,
          attention_mask=causal,
          position_ids=position_ids,
          past_key_values=cache,
          use_cache=True,
          cache_position=torch.arange(S),
          position_embeddings=(cos, sin),
      )
    h = out[0] if isinstance(out, tuple) else out
    ks.append(cache.layers[i].keys[:, :, : S - 1, :].numpy().copy())
    vs.append(cache.layers[i].values[:, :, : S - 1, :].numpy().copy())
    del layer, cache
    print(f"layer {i} done", flush=True)

  w = load_keys_for("model.norm.", args.model_dir, index, None)
  norm = modeling_deepseek_v3.DeepseekV3RMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
  )
  norm.load_state_dict({"weight": w["model.norm.weight"]}, assign=True)
  with torch.no_grad():
    h = norm(h)
  w = load_keys_for("lm_head.", args.model_dir, index, None)
  head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
  head.load_state_dict({"weight": w["lm_head.weight"]}, assign=True)
  with torch.no_grad():
    logits = head(h)

  save = {
      "prompt": prompt.numpy(),
      "next_token": next_token.numpy(),
      "decode_logits": logits[:, -1].numpy(),
      "prefill_last_logits": logits[:, -2].numpy(),
  }
  for i in range(config.num_hidden_layers):
    save[f"k_{i}"] = ks[i]
    save[f"v_{i}"] = vs[i]
  np.savez_compressed(args.out, **save)
  print(f"saved {args.out}; decode argmax={logits[0, -1].argmax().item()}")


if __name__ == "__main__":
  main()
