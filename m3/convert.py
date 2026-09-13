"""Converts REAL-weight Moonlight (full or sliced) to .tflite via export_hf.

Mirrors model_ext/deepseek_v3/convert_test.py but loads real weights.

Memory strategies (--load):
  fp32   : from_pretrained(torch_dtype=float32) — simple, 4x bytes of bf16.
  mmapf32: load on meta device, stream safetensors shards into ONE big
           file-backed fp32 tensor (torch.from_file, shared=True). Keeps
           weights out of anonymous RAM: pages are clean/evictable, so the
           kernel drops them under pressure instead of swapping.
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
import memlog  # noqa: E402

import litert_torch.generative.export_hf  # noqa: F401,E402  registers moe fns
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.export_hf.core import (  # noqa: E402
    exportable_module_config,
)
from litert_torch.generative.export_hf.model_ext import (  # noqa: E402
    patches as patches_lib,
)
from transformers.models.deepseek_v3 import modeling_deepseek_v3  # noqa: E402


def load_fp32(model_dir, config):
  return modeling_deepseek_v3.DeepseekV3ForCausalLM.from_pretrained(
      model_dir, config=config, torch_dtype=torch.float32
  )


def load_mmap_fp32(model_dir, config, backing_file):
  """Meta-device model + safetensors streamed into a file-backed fp32 blob."""
  from safetensors import safe_open

  with torch.device("meta"):
    model = modeling_deepseek_v3.DeepseekV3ForCausalLM(config)

  # Plan the layout of every state-dict entry (params + persistent buffers)
  # in one big file-backed fp32 tensor.
  sd_meta = model.state_dict()
  offsets = {}
  total = 0
  for name, t in sorted(sd_meta.items()):
    offsets[name] = (total, t.numel(), tuple(t.shape))
    total += t.numel()
  print(f"[mmapf32] total param elems={total} ({total*4/2**30:.1f} GiB)")
  blob = torch.from_file(
      backing_file, shared=True, size=total, dtype=torch.float32
  )

  views = {}
  for name, (off, numel, shape) in offsets.items():
    views[name] = blob[off : off + numel].view(shape)

  # Map checkpoint keys -> model params, incl. legacy per-expert layout.
  idx = json.load(
      open(os.path.join(model_dir, "model.safetensors.index.json"))
  )
  shard_keys = {}
  for k, shard in idx["weight_map"].items():
    shard_keys.setdefault(shard, []).append(k)

  n_written = 0
  t0 = time.time()
  for shard in sorted(shard_keys):
    path = os.path.join(model_dir, shard)
    with safe_open(path, framework="pt") as f:
      for key in shard_keys[shard]:
        t = f.get_tensor(key)
        m = re.match(
            r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.weight",
            key,
        )
        if m:
          base, eidx, proj = m.group(1), int(m.group(2)), m.group(3)
          if proj == "down_proj":
            dst = views[f"{base}.down_proj"][eidx]
            dst.copy_(t.to(torch.float32))
          else:
            gup = views[f"{base}.gate_up_proj"][eidx]
            inter = gup.shape[0] // 2
            half = gup[:inter] if proj == "gate_proj" else gup[inter:]
            half.copy_(t.to(torch.float32))
        else:
          if key not in views:
            print(f"[mmapf32] SKIP unmapped key {key}")
            continue
          views[key].copy_(t.to(torch.float32))
        n_written += 1
        del t
    print(
        f"[mmapf32] {shard} done ({n_written} tensors,"
        f" {time.time()-t0:.0f}s)",
        flush=True,
    )

  # Attach the views as real parameters / persistent buffers.
  for name, (off, numel, shape) in offsets.items():
    mod = model
    parts = name.split(".")
    for pt in parts[:-1]:
      mod = getattr(mod, pt)
    leaf = parts[-1]
    if leaf in mod._buffers:
      mod._buffers[leaf] = views[name]
    else:
      setattr(mod, leaf, nn.Parameter(views[name], requires_grad=False))
  # Rebuild non-persistent buffers (rotary inv_freq) on cpu.
  for mname, mod in model.named_modules():
    for bname, buf in list(mod.named_buffers(recurse=False)):
      if buf.is_meta:
        if "inv_freq" in bname:
          rope_cls = type(mod)
          new = rope_cls(config=config)
          mod.register_buffer(
              bname, new.inv_freq.clone(), persistent=False
          )
        else:
          raise RuntimeError(f"meta buffer left: {mname}.{bname}")
  return model


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model-dir", required=True)
  p.add_argument("--workdir", required=True)
  p.add_argument("--prefill-len", type=int, default=8)
  p.add_argument("--cache-len", type=int, default=64)
  p.add_argument("--load", choices=["fp32", "mmapf32"], default="fp32")
  p.add_argument("--lightweight", action="store_true")
  p.add_argument("--quantize", default=None, help="e.g. dynamic_wi8_afp32")
  p.add_argument("--skip-fp32-convert", action="store_true",
                 help="only quantize an existing workdir/model.tflite")
  p.add_argument("--purge-shards", action="store_true",
                 help="after weights are loaded, delete the source "
                 ".safetensors blobs (and symlinks) to free disk")
  args = p.parse_args()
  os.makedirs(args.workdir, exist_ok=True)
  memlog.start("convert", logfile=os.path.join(args.workdir, "mem.log"))

  if args.skip_fp32_convert:
    path = os.path.join(args.workdir, "model.tflite")
    qpath = export_lib.maybe_quantize_model(path, args.quantize)
    print(f"quantized: {qpath} size={os.path.getsize(qpath)}")
    return

  config = modeling_deepseek_v3.DeepseekV3Config.from_pretrained(
      args.model_dir
  )
  config._attn_implementation = "lrt_transposed_attention"
  config._experts_implementation = "litert_moe_sequential"

  t0 = time.time()
  with patches_lib.get_patch_context("deepseek_v3"):
    if args.load == "fp32":
      model = load_fp32(args.model_dir, config)
    else:
      model = load_mmap_fp32(
          args.model_dir, config, os.path.join(args.workdir, "weights.f32")
      )
  model.eval()
  for pa in model.parameters():
    pa.requires_grad_(False)
  print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
  model = export_lib.pre_split_model_experts(model)
  gc.collect()

  export_cfg = exportable_module_config.ExportableModuleConfig(
      model="deepseek_v3",
      batch_size=1,
      cache_length=args.cache_len,
      prefill_lengths=[args.prefill_len],
      cache_implementation="LiteRTLMCache",
      k_ts_idx=2,
      v_ts_idx=3,
      moe_exports_implementation="litert_moe_sequential",
      quantization_recipe=None,
      experimental_lightweight_conversion=args.lightweight,
      work_dir=args.workdir,
  )
  artifacts = export_lib.SourceModelArtifacts(
      model=model,
      model_config=config,
      text_model_config=config,
      tokenizer=None,
  )
  export_cfg = export_lib.update_export_config(export_cfg, artifacts)
  t0 = time.time()
  exported = export_lib.export_text_prefill_decode_model(
      artifacts, export_cfg, export_lib.ExportedModelArtifacts()
  )
  path = exported.prefill_decode_model_path
  print(
      f"converted in {time.time()-t0:.0f}s: {path}"
      f" size={os.path.getsize(path)}",
      flush=True,
  )

  if args.quantize:
    del model, artifacts
    gc.collect()
    t0 = time.time()
    qpath = export_lib.maybe_quantize_model(path, args.quantize)
    print(
        f"quantized in {time.time()-t0:.0f}s: {qpath}"
        f" size={os.path.getsize(qpath)}",
        flush=True,
    )


if __name__ == "__main__":
  main()
