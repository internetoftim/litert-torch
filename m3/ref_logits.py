"""Reference logits/caches from unpatched transformers DeepseekV3ForCausalLM.

Loads REAL weights fp32, runs an 8-token prefill + one decode token in a
single eager forward, saves logits and per-layer K/V for parity checks.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import memlog  # noqa: E402

from transformers.models.deepseek_v3 import modeling_deepseek_v3  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model-dir", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--prefill-len", type=int, default=8)
  args = p.parse_args()
  memlog.start("ref")

  config = modeling_deepseek_v3.DeepseekV3Config.from_pretrained(
      args.model_dir
  )
  config._attn_implementation = "eager"
  config._experts_implementation = None
  model = modeling_deepseek_v3.DeepseekV3ForCausalLM.from_pretrained(
      args.model_dir, config=config, torch_dtype=torch.float32
  )
  model.eval()
  print("model loaded", flush=True)

  torch.manual_seed(7)
  n = args.prefill_len
  prompt = torch.randint(1, config.vocab_size, (1, n), dtype=torch.int32)
  next_token = torch.randint(1, config.vocab_size, (1, 1), dtype=torch.int32)
  full_ids = torch.cat([prompt, next_token], dim=1).long()

  with torch.no_grad():
    out = model(input_ids=full_ids, use_cache=True)
  logits = out.logits  # [1, n+1, V]
  cache = out.past_key_values

  save = {
      "prompt": prompt.numpy(),
      "next_token": next_token.numpy(),
      "decode_logits": logits[:, -1].numpy(),
      "prefill_last_logits": logits[:, n - 1].numpy(),
  }
  for i in range(config.num_hidden_layers):
    # keys/values: [B, H, S, D]; keep only the prefill region [:, :, :n, :]
    save[f"k_{i}"] = cache.layers[i].keys[:, :, :n, :].numpy()
    save[f"v_{i}"] = cache.layers[i].values[:, :, :n, :].numpy()
  np.savez_compressed(args.out, **save)
  print(f"saved {args.out}")


if __name__ == "__main__":
  main()
