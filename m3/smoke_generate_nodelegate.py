"""Short greedy generation smoke test on a converted Moonlight .tflite.

Uses the Moonlight tokenizer (remote code from the HF snapshot) + chat
template, runs prefill in fixed-size chunks + decode steps with the LiteRT
interpreter, greedy-decodes max-new tokens.
"""

import argparse
import os
import time

import numpy as np
from ai_edge_litert import interpreter as interpreter_lib

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--moonshotai--Moonlight-16B-A3B-Instruct/"
    "snapshots/4e735b07a89f73647dfab71ab91b840f362ede5b"
)


def causal_mask(seq_len, cache_len, positions):
  cache_positions = np.arange(cache_len).reshape(1, 1, 1, cache_len)
  q = positions.reshape(1, 1, seq_len, 1)
  return np.where(cache_positions <= q, 0.0, -1e38).astype(np.float32)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--tflite", required=True)
  p.add_argument("--prefill-len", type=int, default=8)
  p.add_argument("--cache-len", type=int, default=64)
  p.add_argument("--prompt", default="Reply with the single word: ready")
  p.add_argument("--max-new", type=int, default=8)
  args = p.parse_args()

  from transformers import AutoTokenizer

  tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
  ids = tok.apply_chat_template(
      [{"role": "user", "content": args.prompt}],
      add_generation_prompt=True,
  )
  if not isinstance(ids, list):
    ids = ids["input_ids"]
  ids = list(ids)
  print(f"prompt ids ({len(ids)}): {ids}")

  interp = interpreter_lib.Interpreter(
      args.tflite,
      experimental_op_resolver_type=interpreter_lib.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
      num_threads=8,
  )
  prefill = interp.get_signature_runner(f"prefill_{args.prefill_len}")
  decode = interp.get_signature_runner("decode")
  kv = {}
  for name, det in prefill.get_input_details().items():
    if name.startswith("kv_cache"):
      kv[name] = np.zeros(det["shape"], dtype=np.float32)

  pos = 0
  n = args.prefill_len
  t0 = time.time()
  while len(ids) - pos > 1 and len(ids) - pos >= n:
    chunk = np.array([ids[pos : pos + n]], dtype=np.int32)
    positions = np.arange(pos, pos + n, dtype=np.int32)
    out = prefill(
        tokens=chunk,
        input_pos=positions,
        mask=causal_mask(n, args.cache_len, positions),
        **kv,
    )
    for k in kv:
      kv[k] = out[k]
    pos += n
    print(f"prefill chunk done, pos={pos} ({time.time()-t0:.1f}s)", flush=True)

  # Remaining prompt tokens (all but the last) go through decode steps.
  while pos < len(ids) - 1:
    out = decode(
        tokens=np.array([[ids[pos]]], dtype=np.int32),
        input_pos=np.array([pos], dtype=np.int32),
        mask=causal_mask(1, args.cache_len, np.array([pos])),
        **kv,
    )
    for k in kv:
      kv[k] = out[k]
    pos += 1
    print(f"prompt decode step, pos={pos} ({time.time()-t0:.1f}s)", flush=True)

  cur = ids[-1]
  generated = []
  for _ in range(args.max_new):
    t1 = time.time()
    out = decode(
        tokens=np.array([[cur]], dtype=np.int32),
        input_pos=np.array([pos], dtype=np.int32),
        mask=causal_mask(1, args.cache_len, np.array([pos])),
        **kv,
    )
    for k in kv:
      kv[k] = out[k]
    pos += 1
    cur = int(out["logits"][0, -1].argmax())
    generated.append(cur)
    print(
        f"gen token {cur!r} -> {tok.decode([cur])!r}"
        f" ({time.time()-t1:.1f}s/step)",
        flush=True,
    )
    if cur in (tok.eos_token_id, 163586):
      break
  print("GENERATED:", tok.decode(generated))


if __name__ == "__main__":
  main()
