"""Builds a sliced local model dir (first N layers of Moonlight, REAL weights).

Creates <out_dir> with:
  - config.json rewritten: num_hidden_layers=N, auto_map removed (native class)
  - model.safetensors.index.json filtered to keys for layers < N + embed/norm/head
  - symlinks to the needed shard files in the HF cache snapshot
"""

import argparse
import json
import os
import re
import sys

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--moonshotai--Moonlight-16B-A3B-Instruct/"
    "snapshots/4e735b07a89f73647dfab71ab91b840f362ede5b"
)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--layers", type=int, required=True)
  p.add_argument("--out", required=True)
  args = p.parse_args()

  os.makedirs(args.out, exist_ok=True)

  cfg = json.load(open(os.path.join(SNAP, "config.json")))
  cfg["num_hidden_layers"] = args.layers
  cfg.pop("auto_map", None)  # force native transformers deepseek_v3
  json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

  idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
  wm = {}
  for k, v in idx["weight_map"].items():
    m = re.match(r"model\.layers\.(\d+)\.", k)
    if m and int(m.group(1)) >= args.layers:
      continue
    wm[k] = v
  shards = sorted(set(wm.values()))
  for s in shards:
    src = os.path.join(SNAP, s)
    if not os.path.exists(src):
      print(f"MISSING shard: {s}", file=sys.stderr)
      sys.exit(1)
    dst = os.path.join(args.out, s)
    if not os.path.exists(dst):
      os.symlink(os.path.realpath(src), dst)
  json.dump(
      {"metadata": idx.get("metadata", {}), "weight_map": wm},
      open(os.path.join(args.out, "model.safetensors.index.json"), "w"),
  )
  print(f"slice ready: {args.out} layers={args.layers} shards={shards}")


if __name__ == "__main__":
  main()
