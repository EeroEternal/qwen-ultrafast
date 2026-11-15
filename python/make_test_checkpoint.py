#!/usr/bin/env python3
import argparse
import json
import shutil
import struct
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/qwen-test"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {}

    def add(name, dtype, shape, payload):
        tensors[name] = (dtype, shape, payload)

    def bf16(name, shape):
        value = (torch.randn(shape, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        add(name, "BF16", shape, value.view(torch.uint16).numpy().tobytes())

    def f32(name, shape, value):
        add(name, "F32", shape, torch.tensor(value, dtype=torch.float32).numpy().tobytes())

    def nvfp4(name, n, k):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
        scales = torch.full((n, k // 16), 0x38, dtype=torch.uint8)
        add(f"{name}.weight", "U8", [n, k // 2], codes.numpy().tobytes())
        add(f"{name}.weight_scale", "F8_E4M3", [n, k // 16], scales.numpy().tobytes())
        f32(f"{name}.weight_scale_2", [], 0.1)

    hidden = 2560
    wide = hidden * 4
    key_dim = 2 * 128
    value_dim = 4 * 128
    vocab = 32768
    experts = 16
    bf16("model.language_model.embed_tokens.weight", [vocab, hidden])
    bf16("lm_head.weight", [vocab, hidden])
    for prefix in ("model.language_model.hyper_connection_mixer",):
        bf16(f"{prefix}.hc_norm.weight", [wide])
        bf16(f"{prefix}.input_mix_weight_down.weight", [320, wide])
        bf16(f"{prefix}.input_mix_weight_up.weight", [wide, 320])
    prefix = "model.language_model.layers.0"
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        current = f"{prefix}.{name}"
        bf16(f"{current}.hc_norm.weight", [wide])
        bf16(f"{current}.input_mix_weight_down.weight", [320, wide])
        bf16(f"{current}.input_mix_weight_up.weight", [wide, 320])
        bf16(f"{current}.block_inject_weight.weight", [4, wide])
    linear = f"{prefix}.linear_attn"
    bf16(f"{linear}.in_proj_qkv.weight", [2 * key_dim + value_dim, hidden])
    bf16(f"{linear}.in_proj_z.weight", [value_dim, hidden])
    bf16(f"{linear}.in_proj_a.weight", [4, hidden])
    bf16(f"{linear}.in_proj_b.weight", [4, hidden])
    bf16(f"{linear}.out_proj.weight", [hidden, value_dim])
    bf16(f"{linear}.conv1d.weight", [2 * key_dim + value_dim, 1, 4])
    bf16(f"{linear}.norm.weight", [128])
    f32(f"{linear}.A_log", [4], [0.0, 0.1, 0.2, 0.3])
    f32(f"{linear}.dt_bias", [4], [0.1, 0.1, 0.1, 0.1])
    mlp = f"{prefix}.mlp"
    bf16(f"{mlp}.gate.weight", [experts, hidden])
    for name in ("gate_proj", "up_proj"):
        bf16(f"{mlp}.shared_expert.{name}.weight", [640, hidden])
    bf16(f"{mlp}.shared_expert.down_proj.weight", [hidden, 640])
    bf16(f"{mlp}.shared_expert_gate.weight", [1, hidden])
    for expert in range(experts):
        for name, n, k in (("gate_proj", 640, hidden), ("up_proj", 640, hidden), ("down_proj", hidden, 640)):
            nvfp4(f"{mlp}.experts.{expert}.{name}", n, k)
    config = {
        "model_type": "qwen4_exp",
        "tie_word_embeddings": False,
        "text_config": {
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "vocab_size": vocab,
            "eos_token_id": 248044,
            "layer_types": ["linear_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 128,
            "moe_intermediate_size": 640,
            "num_experts": experts,
            "num_experts_per_tok": 10,
            "shared_expert_intermediate_size": 640,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "heads_per_ngram": 8,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": hidden,
            "ple_layer_ids": [],
            "seed": 1234,
            "norm_topk_prob": True,
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    header = {}
    offset = 0
    payloads = []
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        payloads.append(payload)
        offset += len(payload)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with (args.output / "model.safetensors").open("wb") as output:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        for payload in payloads:
            output.write(payload)
    (args.output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": offset},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }, indent=2) + "\n")
    shutil.copyfile(args.tokenizer, args.output / "tokenizer.json")
    print(f"wrote {len(tensors)} tensors, {offset} bytes to {args.output}")


if __name__ == "__main__":
    main()
