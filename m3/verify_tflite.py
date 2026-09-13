"""LiteRT interpreter parity check vs saved transformers reference.

Runs prefill_<N> + one decode step on the given .tflite, compares:
  - decode-step logits vs reference (max abs diff)
  - per-layer prefill K/V cache (filled region) vs reference (max abs diff)
"""

import argparse

import numpy as np
from ai_edge_litert import interpreter as interpreter_lib


def create_causal_mask(seq_len, cache_length, input_pos):
  cache_positions = np.arange(cache_length).reshape(1, 1, 1, cache_length)
  q_pos = input_pos.reshape(1, 1, seq_len, 1)
  return np.where(
      cache_positions <= q_pos, 0.0, -1e38
  ).astype(np.float32)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--tflite", required=True)
  p.add_argument("--ref", required=True)
  p.add_argument("--cache-len", type=int, default=64)
  p.add_argument("--skip-cache-check", action="store_true")
  args = p.parse_args()

  ref = np.load(args.ref)
  prompt = ref["prompt"]
  next_token = ref["next_token"]
  n = prompt.shape[1]

  interp = interpreter_lib.Interpreter(args.tflite)
  sigs = interp.get_signature_list()
  print("signatures:", list(sigs))
  prefill = interp.get_signature_runner(f"prefill_{n}")
  decode = interp.get_signature_runner("decode")

  zero_kv = {}
  for name, det in prefill.get_input_details().items():
    if name.startswith("kv_cache"):
      zero_kv[name] = np.zeros(det["shape"], dtype=np.float32)

  prefill_pos = np.arange(n, dtype=np.int32)
  prefill_out = prefill(
      tokens=prompt.astype(np.int32),
      input_pos=prefill_pos,
      mask=create_causal_mask(n, args.cache_len, prefill_pos),
      **zero_kv,
  )
  decode_pos = np.array([n], dtype=np.int32)
  decode_out = decode(
      tokens=next_token.astype(np.int32),
      input_pos=decode_pos,
      mask=create_causal_mask(1, args.cache_len, decode_pos),
      **{k: prefill_out[k] for k in zero_kv},
  )

  logits = decode_out["logits"][:, -1]
  d = np.abs(logits - ref["decode_logits"]).max()
  print(f"decode logits max abs diff vs eager HF: {d:.3e}")
  print(f"decode argmax tflite={logits.argmax()} ref={ref['decode_logits'].argmax()}")

  if not args.skip_cache_check:
    i = 0
    worst_k = worst_v = 0.0
    while f"k_{i}" in ref:
      # K: tflite [B,H,cache,dk] vs ref [B,H,n,dk]
      kd = np.abs(
          prefill_out[f"kv_cache_k_{i}"][:, :, :n, :] - ref[f"k_{i}"]
      ).max()
      # V: tflite [B,H,dv,cache] vs ref [B,H,n,dv]
      vd = np.abs(
          prefill_out[f"kv_cache_v_{i}"][:, :, :, :n]
          - ref[f"v_{i}"].transpose(0, 1, 3, 2)
      ).max()
      worst_k, worst_v = max(worst_k, kd), max(worst_v, vd)
      print(f"layer {i}: prefill cache max abs diff k={kd:.3e} v={vd:.3e}")
      i += 1
    print(f"worst prefill cache diff: k={worst_k:.3e} v={worst_v:.3e}")


if __name__ == "__main__":
  main()
