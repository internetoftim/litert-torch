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
"""Converts all Qwen3-TTS models and tables to GPU-compatible LiteRT TFLite files.

Usage:
  bazel run //third_party/py/litert_torch/generative/export_hf/model_ext/qwen3_tts:export_qwen3_tts_gpu -- \
    --model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base" \
    --output_dir="/tmp/qwen3_tts_gpu"
"""

import copy
import importlib
import json
import os
import shutil
import sys
import tempfile

from absl import app
from absl import flags
import huggingface_hub
import numpy as np
import safetensors
import safetensors.torch
import tensorflow as tf
import torch

import litert_torch
from litert_torch.generative.export_hf import export as export_hf
from litert_torch.generative.export_hf.model_ext.qwen3_tts import exportable_module
from litert_torch.generative.export_hf.model_ext.qwen3_tts import patch as qwen3_tts_patch
from litert_torch.generative.export_hf.model_ext.qwen3_tts import speaker_encoder as qwen3_tts_speaker_encoder
from litert_torch.generative.quantize import quant_recipes

from ai_edge_quantizer import recipe as recipe_lib

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir",
    "/tmp/qwen3_tts_gpu",
    "Directory to save exported GPU TFLite models.",
)
flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "HuggingFace model ID or local checkpoint directory.",
)
flags.DEFINE_enum(
    "quantization",
    "int4_weight_only",
    ["fp16", "int4_weight_only", "fp32"],
    "Quantization scheme for GPU delegate execution.",
)
flags.DEFINE_string(
    "litert_samples_conversion_dir",
    "/tmp/litert_samples/compiled_model_api/text_to_speech_lm/conversion",
    "Path to litert-samples conversion directory containing qtok12.",
)
flags.DEFINE_list(
    "targets",
    ["all"],
    "List of components to export (all, embeddings, codec, mtp, talker).",
)

_TALKER_SYNTH_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2149,
    "eos_token_id": 2150,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 4096,
}


def setup_gpu_talker_recipe():
  """Sets up blockwise-32 INT4 weight-only quantization with INT8 embedding lookup."""
  wo_recipe = recipe_lib.dynamic_wi4_afp32()[0]
  emb_recipe = copy.deepcopy(wo_recipe)
  emb_recipe["op_config"]["weight_tensor_config"]["num_bits"] = 8
  emb_recipe["operation"] = "EMBEDDING_LOOKUP"
  block_recipe = copy.deepcopy(wo_recipe)
  if hasattr(recipe_lib, "AlgorithmName") and hasattr(
      recipe_lib.AlgorithmName, "OCTAV"
  ):
    block_recipe["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  block_recipe["op_config"]["weight_tensor_config"][
      "granularity"
  ] = "BLOCKWISE_32"
  setattr(recipe_lib, "GPU_BOCTAV4", lambda: [block_recipe, emb_recipe])


def get_mtp_quant_config():
  """Returns the LiteRT GPU-compatible quantization recipe for MTP."""
  return quant_recipes.full_fp16_recipe()


def _save_tflite_with_signatures(mod: tf.Module, out_path: str):
  """Exports a tf.Module to TFLite via SavedModel to preserve 'serving_default' signature definitions."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    tf.saved_model.save(
        mod,
        tmp_dir,
        signatures={"serving_default": mod.__call__.get_concrete_function()},
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
    with open(out_path, "wb") as f:
      f.write(converter.convert())


def export_embedding_and_projection_tables(reader, output_dir: str):
  """Exports embedding tables and text projection directly to TFLite models with signature definitions."""
  print("Converting text_embedding.tflite...")
  text_emb_weights = (
      reader.get_tensor("talker.model.text_embedding.weight")
      .to(torch.float32)
      .numpy()
  )

  class TextEmbeddingModule(tf.Module):

    def __init__(self, emb_matrix):
      self.emb_matrix = tf.Variable(
          emb_matrix, dtype=tf.float32, trainable=False
      )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="token_ids")
        ]
    )
    def __call__(self, token_ids):
      return tf.nn.embedding_lookup(self.emb_matrix, token_ids)

  mod_text_emb = TextEmbeddingModule(text_emb_weights)
  _save_tflite_with_signatures(
      mod_text_emb, os.path.join(output_dir, "text_embedding.tflite")
  )

  print("Converting codec_embedding.tflite...")
  codec_emb_weights = (
      reader.get_tensor("talker.model.codec_embedding.weight")
      .to(torch.float32)
      .numpy()
  )

  class CodecEmbeddingModule(tf.Module):

    def __init__(self, emb_matrix):
      self.emb_matrix = tf.Variable(
          emb_matrix, dtype=tf.float32, trainable=False
      )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="codec_ids")
        ]
    )
    def __call__(self, codec_ids):
      return tf.nn.embedding_lookup(self.emb_matrix, codec_ids)

  mod_codec_emb = CodecEmbeddingModule(codec_emb_weights)
  _save_tflite_with_signatures(
      mod_codec_emb, os.path.join(output_dir, "codec_embedding.tflite")
  )

  print("Converting mtp_embedding.tflite...")
  mtp_embs = [
      reader.get_tensor(
          f"talker.code_predictor.model.codec_embedding.{i}.weight"
      )
      .to(torch.float32)
      .numpy()
      for i in range(15)
  ]
  mtp_embs_flat = np.stack(mtp_embs).reshape(-1, mtp_embs[0].shape[-1])

  class MtpEmbeddingModule(tf.Module):

    def __init__(self, emb_matrix):
      self.emb_matrix = tf.Variable(
          emb_matrix, dtype=tf.float32, trainable=False
      )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="mtp_ids")
        ]
    )
    def __call__(self, mtp_ids):
      return tf.nn.embedding_lookup(self.emb_matrix, mtp_ids)

  mod_mtp_emb = MtpEmbeddingModule(mtp_embs_flat)
  _save_tflite_with_signatures(
      mod_mtp_emb, os.path.join(output_dir, "mtp_embedding.tflite")
  )

  print("Converting text_projection.tflite...")
  w1 = (
      reader.get_tensor("talker.text_projection.linear_fc1.weight")
      .to(torch.float32)
      .numpy()
  )
  b1 = (
      reader.get_tensor("talker.text_projection.linear_fc1.bias")
      .to(torch.float32)
      .numpy()
  )
  w2 = (
      reader.get_tensor("talker.text_projection.linear_fc2.weight")
      .to(torch.float32)
      .numpy()
  )
  b2 = (
      reader.get_tensor("talker.text_projection.linear_fc2.bias")
      .to(torch.float32)
      .numpy()
  )

  class TextProjectionModule(tf.Module):

    def __init__(self, w1, b1, w2, b2):
      self.w1 = tf.Variable(w1, dtype=tf.float32, trainable=False)
      self.b1 = tf.Variable(b1, dtype=tf.float32, trainable=False)
      self.w2 = tf.Variable(w2, dtype=tf.float32, trainable=False)
      self.b2 = tf.Variable(b2, dtype=tf.float32, trainable=False)

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, 2048], dtype=tf.float32, name="text_embeds"
            )
        ]
    )
    def __call__(self, text_embeds):
      h = tf.nn.silu(
          tf.matmul(text_embeds, self.w1, transpose_b=True) + self.b1
      )
      return tf.matmul(h, self.w2, transpose_b=True) + self.b2

  mod_text_proj = TextProjectionModule(w1, b1, w2, b2)
  _save_tflite_with_signatures(
      mod_text_proj, os.path.join(output_dir, "text_projection.tflite")
  )


def export_codec_decoder(model_dir: str, output_dir: str):
  """Exports codec_decoder_fp32.tflite using qtok12 reference modeling."""
  print("Exporting codec_decoder_fp32.tflite...")
  conv_dir = FLAGS.litert_samples_conversion_dir
  if not os.path.exists(conv_dir):
    raise FileNotFoundError(
        f"Expected conversion reference directory at {conv_dir} not found."
    )
  if conv_dir not in sys.path:
    sys.path.insert(0, conv_dir)

  config_mod = importlib.import_module(
      "qtok12.configuration_qwen3_tts_tokenizer_v2"
  )
  model_mod = importlib.import_module("qtok12.modeling_qwen3_tts_tokenizer_v2")
  cfg_cls = config_mod.Qwen3TTSTokenizerV2Config
  model_cls = model_mod.Qwen3TTSTokenizerV2Model

  speech_tok_dir = os.path.join(model_dir, "speech_tokenizer")
  if not os.path.exists(speech_tok_dir):
    speech_tok_dir = huggingface_hub.snapshot_download(
        FLAGS.model_id, allow_patterns=["speech_tokenizer/*"]
    )
    speech_tok_dir = os.path.join(speech_tok_dir, "speech_tokenizer")

  config = cfg_cls.from_pretrained(speech_tok_dir)
  model = model_cls.from_pretrained(
      speech_tok_dir, config=config, torch_dtype=torch.float32
  ).eval()
  decoder = model.decoder
  decoder.pre_transformer.config.use_cache = False

  rotary = decoder.pre_transformer.rotary_emb
  dim = decoder.pre_transformer.config.head_dim
  theta = decoder.pre_transformer.config.rope_theta
  with torch.no_grad():
    rotary.inv_freq.copy_(
        1.0
        / (
            theta
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                    device=rotary.inv_freq.device,
                )
                / dim
            )
        )
    )
  rotary.attention_scaling = 1.0

  class CodecDecode(torch.nn.Module):

    def __init__(self, dec):
      super().__init__()
      self.dec = dec

    def forward(self, codes):
      return self.dec(codes)

  out_path = os.path.join(output_dir, "codec_decoder_fp32.tflite")
  sample_input = (torch.zeros(1, 16, 64, dtype=torch.int32),)
  litert_torch.convert(CodecDecode(decoder).eval(), sample_input).export(
      out_path
  )
  print(f"Codec decoder exported to: {out_path}")


def export_mtp_gpu(reader, output_dir: str):
  """Authors and exports the static rank-4 GPU-compatible MTP step model."""
  print("Exporting GPU-compatible MTP model...")
  prefix = "talker.code_predictor."
  weights = {}
  for key in reader.keys():
    if (
        key.startswith(prefix + "model.layers.")
        or key == prefix + "model.norm.weight"
    ):
      weights[key[len(prefix + "model.") :]] = reader.get_tensor(key).to(
          torch.float32
      )
  heads = [
      reader.get_tensor(f"{prefix}lm_head.{i}.weight").to(torch.float32)
      for i in range(15)
  ]
  weights["heads"] = torch.stack(heads)

  mtp_module = exportable_module.MtpStepGpu(weights).eval()
  sample_inputs = {
      "embeddings": torch.zeros(1, 1, 1024, dtype=torch.float32),
      "input_ids": torch.zeros(1, dtype=torch.int32),
      "mask": torch.zeros(1, 1, 1, 32, dtype=torch.float32),
      **{
          f"kv_cache_k_{i}": torch.zeros(1, 32, 8, 128, dtype=torch.float32)
          for i in range(5)
      },
      **{
          f"kv_cache_v_{i}": torch.zeros(1, 32, 8, 128, dtype=torch.float32)
          for i in range(5)
      },
  }

  out_path = os.path.join(output_dir, "mtp_fp32.tflite")
  quant_cfg = get_mtp_quant_config()
  litert_torch.convert(
      mtp_module, sample_kwargs=sample_inputs, quant_config=quant_cfg
  ).export(out_path)
  print(f"MTP GPU model exported to: {out_path}")


def export_talker_gpu(reader, model_dir: str, output_dir: str):
  """Synthesizes causal checkpoint and exports Talker LLM with GPU recipe."""
  print("Exporting GPU-compatible Talker model...")
  synth_dir = "/tmp/synth_talker_ckpt"
  os.makedirs(synth_dir, exist_ok=True)

  out_tensors = {}
  for key in reader.keys():
    if (
        not key.startswith("talker.model.")
        and key != "talker.codec_head.weight"
    ):
      continue
    if key.startswith("talker.model.text_embedding"):
      continue
    tensor = reader.get_tensor(key).to(torch.float32)
    if key == "talker.model.codec_embedding.weight":
      pad = torch.zeros(4096 - tensor.shape[0], tensor.shape[1])
      out_tensors["model.embed_tokens.weight"] = torch.cat([tensor, pad], 0)
    elif key == "talker.codec_head.weight":
      eye = torch.eye(1024)
      eye = eye + 1e-6 * (1.0 - eye)
      out_tensors["lm_head.weight"] = torch.cat([tensor, eye], 0)
    else:
      out_tensors[key.replace("talker.model.", "model.")] = tensor

  safetensors.torch.save_file(
      out_tensors,
      os.path.join(synth_dir, "model.safetensors"),
      metadata={"format": "pt"},
  )
  with open(os.path.join(synth_dir, "config.json"), "w") as f:
    json.dump(_TALKER_SYNTH_CONFIG, f, indent=1)
  with open(os.path.join(synth_dir, "generation_config.json"), "w") as f:
    json.dump({"bos_token_id": 2149, "eos_token_id": 2150}, f)

  for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
    src_path = os.path.join(model_dir, name)
    if os.path.exists(src_path):
      shutil.copy(src_path, os.path.join(synth_dir, name))

  talker_out_dir = "/tmp/talker_gpu_export"
  os.makedirs(talker_out_dir, exist_ok=True)

  recipe_name = None
  if FLAGS.quantization == "int4_weight_only":
    setup_gpu_talker_recipe()
    recipe_name = "GPU_BOCTAV4"

  export_hf.export(
      model=synth_dir,
      output_dir=talker_out_dir,
      quantization_recipe=recipe_name,
      externalize_embedder=True,
      single_token_embedder=True,
      cache_length=1024,
      prefill_lengths=[32, 128],
      bundle_litert_lm=False,
      keep_temporary_files=True,
      use_jinja_template=False,
      trust_remote_code=True,
  )

  quant_path = os.path.join(talker_out_dir, "model_quantized.tflite")
  unquant_path = os.path.join(talker_out_dir, "model.tflite")
  dest_path = os.path.join(
      output_dir,
      "talker_int4.tflite"
      if FLAGS.quantization == "int4_weight_only"
      else "talker_fp32.tflite",
  )

  if os.path.exists(quant_path):
    shutil.copy(quant_path, dest_path)
  elif os.path.exists(unquant_path):
    shutil.copy(unquant_path, dest_path)
  else:
    # Fallback if named differently by export_hf
    tflite_files = [
        f for f in os.listdir(talker_out_dir) if f.endswith(".tflite")
    ]
    if tflite_files:
      shutil.copy(os.path.join(talker_out_dir, tflite_files[0]), dest_path)

  print(f"Talker model exported to: {dest_path}")

  # Copy tokenizer.json to output directory if present
  for tok_src in (
      os.path.join(talker_out_dir, "tokenizer.json"),
      os.path.join(model_dir, "tokenizer.json"),
  ):
    if os.path.exists(tok_src):
      shutil.copy(tok_src, os.path.join(output_dir, "tokenizer.json"))
      break


def export_speaker_encoder(reader, output_dir: str):
  """Authors and exports the GPU-compatible ECAPA-TDNN speaker encoder."""
  print("Exporting GPU-compatible speaker encoder...")
  model = qwen3_tts_speaker_encoder.Qwen3TTSSpeakerEncoder().eval()
  state_dict = {}
  prefix = "speaker_encoder."
  for k in reader.keys():
    if k.startswith(prefix):
      state_dict[k[len(prefix) :]] = reader.get_tensor(k).to(torch.float32)
  model.load_state_dict(state_dict)

  out_path = os.path.join(output_dir, "speaker_encoder_fp32.tflite")
  sample_input = (torch.zeros(1, 300, 128, dtype=torch.float32),)
  litert_torch.convert(model, sample_input).export(out_path)
  print(f"Speaker encoder exported to: {out_path}")


def main(_):
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  if os.path.exists(FLAGS.model_id):
    model_dir = FLAGS.model_id
  else:
    print(f"Downloading checkpoint for {FLAGS.model_id}...")
    model_dir = huggingface_hub.snapshot_download(FLAGS.model_id)

  reader = safetensors.safe_open(
      os.path.join(model_dir, "model.safetensors"), framework="pt"
  )

  targets = set(FLAGS.targets)
  export_all = "all" in targets

  if export_all or "speaker_encoder" in targets:
    export_speaker_encoder(reader, FLAGS.output_dir)
  if export_all or "embeddings" in targets:
    export_embedding_and_projection_tables(reader, FLAGS.output_dir)
  if export_all or "codec" in targets:
    export_codec_decoder(model_dir, FLAGS.output_dir)

  with qwen3_tts_patch.qwen3_tts_litert_patch():
    if export_all or "mtp" in targets:
      export_mtp_gpu(reader, FLAGS.output_dir)
    if export_all or "talker" in targets:
      export_talker_gpu(reader, model_dir, FLAGS.output_dir)

  print("\n=======================================================")
  print("Successfully exported all GPU-compatible TFLite models!")
  print("=======================================================")
  for root, _, files in os.walk(FLAGS.output_dir):
    for name in sorted(files):
      path = os.path.join(root, name)
      print(f"{os.path.getsize(path) / 1e6:9.2f} MB  {name}")


if __name__ == "__main__":
  app.run(main)
