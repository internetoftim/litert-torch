"""Parity check: tflite (prefill_N + decode) vs reference npz.

The prefill signature emits only KV caches (no logits), so:
  - prefill path: run prefill_N chunks over the prompt, compare per-layer
    KV caches (filled region) vs ref, then one decode step with next_token
    -> decode logits max abs diff + cosine.
  - prefill-last logits: replay the prompt token-by-token through decode
    from an empty cache; logits of the last prompt token == prefill_last_logits.
"""
import argparse
import resource
import time
import numpy as np
from ai_edge_litert import interpreter as interpreter_lib


def causal_mask(seq_len, cache_len, positions):
  cache_positions = np.arange(cache_len).reshape(1, 1, 1, cache_len)
  q = positions.reshape(1, 1, seq_len, 1)
  return np.where(cache_positions <= q, 0.0, -1e38).astype(np.float32)


def cos(a, b):
  a = a.ravel().astype(np.float64)
  b = b.ravel().astype(np.float64)
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--tflite", required=True)
  p.add_argument("--ref", required=True)
  p.add_argument("--cache-len", type=int, default=64)
  p.add_argument("--prefill-len", type=int, default=8)
  p.add_argument("--skip-cache-check", action="store_true")
  p.add_argument("--skip-prefill-logits", action="store_true")
  args = p.parse_args()

  ref = np.load(args.ref)
  prompt = ref["prompt"]
  next_token = ref["next_token"]
  total = prompt.shape[1]
  n = args.prefill_len

  interp = interpreter_lib.Interpreter(args.tflite)
  print("signatures:", list(interp.get_signature_list()))
  prefill = interp.get_signature_runner(f"prefill_{n}")
  decode = interp.get_signature_runner("decode")

  kv_names = [name for name in prefill.get_input_details()
              if name.startswith("kv_cache")]

  def zero_kv():
    return {name: np.zeros(prefill.get_input_details()[name]["shape"],
                           dtype=np.float32) for name in kv_names}

  ids = list(prompt[0])

  # ---- prefill path ----
  kv = zero_kv()
  pos = 0
  t0 = time.time()
  while len(ids) - pos >= n:
    chunk = np.array([ids[pos:pos + n]], dtype=np.int32)
    positions = np.arange(pos, pos + n, dtype=np.int32)
    out = prefill(tokens=chunk, input_pos=positions,
                  mask=causal_mask(n, args.cache_len, positions), **kv)
    for k in kv:
      kv[k] = out[k]
    pos += n
    print(f"prefill chunk done pos={pos} ({time.time()-t0:.1f}s)", flush=True)
  while pos < len(ids):
    out = decode(tokens=np.array([[ids[pos]]], dtype=np.int32),
                 input_pos=np.array([pos], dtype=np.int32),
                 mask=causal_mask(1, args.cache_len, np.array([pos])), **kv)
    for k in kv:
      kv[k] = out[k]
    pos += 1
    print(f"prompt decode step pos={pos} ({time.time()-t0:.1f}s)", flush=True)

  if not args.skip_cache_check:
    worst_k = worst_v = 0.0
    i = 0
    while f"k_{i}" in ref:
      m = ref[f"k_{i}"].shape[2]
      kd = np.abs(kv[f"kv_cache_k_{i}"][:, :, :m, :] - ref[f"k_{i}"]).max()
      vd = np.abs(kv[f"kv_cache_v_{i}"][:, :, :, :m]
                  - ref[f"v_{i}"].transpose(0, 1, 3, 2)).max()
      worst_k, worst_v = max(worst_k, kd), max(worst_v, vd)
      i += 1
    print(f"worst prefill cache diff over {i} layers: "
          f"k={worst_k:.4e} v={worst_v:.4e}")

  dec = decode(tokens=next_token.astype(np.int32),
               input_pos=np.array([total], dtype=np.int32),
               mask=causal_mask(1, args.cache_len, np.array([total])), **kv)
  dl = dec["logits"][:, -1]
  d = np.abs(dl - ref["decode_logits"]).max()
  print(f"decode logits: max abs diff={d:.4e} "
        f"cos={cos(dl, ref['decode_logits']):.6f} "
        f"argmax tflite={dl.argmax()} ref={ref['decode_logits'].argmax()}")

  # ---- prefill-last logits via decode replay ----
  if not args.skip_prefill_logits:
    kv = zero_kv()
    last = None
    t0 = time.time()
    for pos, tok in enumerate(ids):
      last = decode(tokens=np.array([[tok]], dtype=np.int32),
                    input_pos=np.array([pos], dtype=np.int32),
                    mask=causal_mask(1, args.cache_len, np.array([pos])), **kv)
      for k in kv:
        kv[k] = last[k]
    print(f"decode replay of {len(ids)} prompt tokens: {time.time()-t0:.1f}s")
    pl = last["logits"][:, -1]
    d = np.abs(pl - ref["prefill_last_logits"]).max()
    print(f"prefill-last logits (via decode replay): max abs diff={d:.4e} "
          f"cos={cos(pl, ref['prefill_last_logits']):.6f} "
          f"argmax tflite={pl.argmax()} "
          f"ref={ref['prefill_last_logits'].argmax()}")

  print(f"peak RSS: "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB")


if __name__ == "__main__":
  main()
