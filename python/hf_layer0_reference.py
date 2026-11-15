#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen_impl
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer


FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class Checkpointer:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.handles = {}

    def get(self, name):
        filename = self.weight_map[name]
        if filename not in self.handles:
            self.handles[filename] = safe_open(self.root / filename, framework="pt", device="cpu")
        return self.handles[filename].get_tensor(name)


def dequantize_nvfp4(checkpoint, prefix, device, fp4_lut):
    packed = checkpoint.get(f"{prefix}.weight").to(device)
    scales = checkpoint.get(f"{prefix}.weight_scale").to(device).float()
    global_scale = checkpoint.get(f"{prefix}.weight_scale_2").to(device)
    values = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = fp4_lut[values].flatten(-2).float()
    scales = scales.repeat_interleave(16, dim=-1)
    return (values * scales * global_scale).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.bfloat16)
    config = AutoConfig.from_pretrained(args.model).text_config
    layer = Qwen4ExpTextDecoderLayer(config, 0).to(device).eval()
    checkpoint = Checkpointer(args.model)
    prefix = "model.language_model.layers.0."
    state = layer.state_dict()
    with torch.inference_mode():
        for name in checkpoint.weight_map:
            if name.startswith(prefix) and ".experts." not in name:
                state[name[len(prefix):]].copy_(checkpoint.get(name).to(device))
        fp4_lut = torch.tensor(FP4_VALUES, device=device)
        for expert in range(config.num_experts):
            expert_prefix = f"{prefix}mlp.experts.{expert}"
            gate = dequantize_nvfp4(checkpoint, f"{expert_prefix}.gate_proj", device, fp4_lut)
            up = dequantize_nvfp4(checkpoint, f"{expert_prefix}.up_proj", device, fp4_lut)
            down = dequantize_nvfp4(checkpoint, f"{expert_prefix}.down_proj", device, fp4_lut)
            layer.mlp.experts.gate_up_proj[expert, : config.moe_intermediate_size].copy_(gate)
            layer.mlp.experts.gate_up_proj[expert, config.moe_intermediate_size :].copy_(up)
            layer.mlp.experts.down_proj[expert].copy_(down)
            if expert % 64 == 63:
                print(f"dequantized {expert + 1}/{config.num_experts}", flush=True)
        def report(stage, tensor):
            values = tensor.flatten().float()
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "shape": list(tensor.shape),
                        "square_sum": values.square().sum().item(),
                        "first": values[:8].tolist(),
                    }
                ),
                flush=True,
            )

        embedding = checkpoint.get("model.language_model.embed_tokens.weight")[9419].to(device)
        hidden = embedding.reshape(1, 1, 1, -1).repeat(1, 1, config.hc_count, 1).flatten(-2)
        cache = DynamicCache(config=config)
        mixed, hyper_input, injection = layer.attn_hyper_connection(hidden)
        report("attention_hyper_connection", mixed)
        report("gdn_norm_weight", layer.linear_attn.norm.weight)
        layer.linear_attn.in_proj_qkv.register_forward_hook(
            lambda _module, _args, output: report("gdn_projections", output)
        )
        layer.linear_attn.in_proj_z.register_forward_hook(
            lambda _module, _args, output: report("gdn_z", output)
        )
        layer.linear_attn.in_proj_a.register_forward_hook(
            lambda _module, _args, output: report("gdn_a", output)
        )
        layer.linear_attn.in_proj_b.register_forward_hook(
            lambda _module, _args, output: report("gdn_b", output)
        )
        layer.linear_attn.out_proj.register_forward_pre_hook(
            lambda _module, args: report("gdn_recurrence", args[0])
        )
        original_conv = qwen_impl.causal_conv1d_fn

        def traced_conv(*conv_args, **conv_kwargs):
            output = original_conv(*conv_args, **conv_kwargs)
            report("gdn_convolution", output)
            return output

        qwen_impl.causal_conv1d_fn = traced_conv
        original_chunk = qwen_impl.torch_chunk_gated_delta_rule

        def traced_chunk(query, key, value, g, beta, **chunk_kwargs):
            report("gdn_query", query)
            report("gdn_key", key)
            report("gdn_value", value)
            report("gdn_decay_log", g)
            report("gdn_beta", beta)
            q = query[0, 0].float()
            k = key[0, 0].float()
            q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) / math.sqrt(128)
            k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
            manual = (q * k).sum(dim=-1, keepdim=True) * value[0, 0].float()
            manual = manual * beta[0, 0].float().unsqueeze(-1)
            report("gdn_manual_raw", manual)
            output, state = original_chunk(query, key, value, g, beta, **chunk_kwargs)
            report("gdn_raw", output)
            return output, state

        qwen_impl.torch_chunk_gated_delta_rule = traced_chunk
        attention = layer.linear_attn(mixed, cache=cache)
        report("mixer", attention)
        hidden = hyper_input + (attention.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        mixed, hyper_input, injection = layer.mlp_hyper_connection(hidden)
        report("mlp_hyper_connection", mixed)
        _, routing_weights, selected_experts = layer.mlp.gate(mixed)
        print(f"router ids={selected_experts.flatten().tolist()}", flush=True)
        print(f"router weights={routing_weights.flatten().tolist()}", flush=True)
        moe = layer.mlp(mixed)
        report("moe", moe)
        output = hyper_input + (moe.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        report("layer_complete", output)
        torch.save(output.cpu(), "/tmp/qwen-hf-layer0.pt")


if __name__ == "__main__":
    main()
