#include "common.cuh"

extern "C" __global__ void __launch_bounds__(512) qwen_router_topk(
    const bf16* logits,
    unsigned int* expert_ids,
    float* expert_weights,
    unsigned int experts,
    unsigned int top_k
) {
    __shared__ float probabilities[512];
    __shared__ float reduction[32];
    const unsigned int expert = threadIdx.x;
    const float logit = expert < experts ? to_float(logits[expert]) : -INFINITY;
    const float maximum = block_max(logit, reduction);
    const float probability = expert < experts ? expf(logit - maximum) : 0.0f;
    probabilities[expert] = probability;
    const float sum = block_sum(probability, reduction);
    probabilities[expert] = probability / sum;
    __shared__ unsigned int selected;
    __shared__ float selected_sum;
    if (expert == 0) {
        selected_sum = 0.0f;
    }
    __syncthreads();
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        const float best = block_max(probabilities[expert], reduction);
        if (expert == 0) {
            selected = experts;
        }
        __syncthreads();
        if (probabilities[expert] == best) {
            atomicMin(&selected, expert);
        }
        __syncthreads();
        if (expert == 0) {
            expert_ids[slot] = selected;
            expert_weights[slot] = best;
            probabilities[selected] = -1.0f;
            selected_sum += best;
        }
        __syncthreads();
    }
    if (expert < top_k) {
        expert_weights[expert] /= selected_sum;
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_gemv(
    const bf16* input,
    const uint8_t* weight,
    const uint8_t* scales,
    const float* weight_scale_2,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    if (column >= n) {
        return;
    }
    const unsigned int groups = k / 16;
    const uint64_t* weight_row = reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(column) * (k / 2)
    );
    float sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t packed = weight_row[group];
        const bf16* values = input + group * 16;
        float dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const unsigned int code = (packed >> (4 * j)) & 15;
            dot = fmaf(to_float(values[j]), fp4_to_float(code), dot);
        }
        sum = fmaf(fp8_to_float(scales[column * groups + group]), dot, sum);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        output[column] = from_float(sum * weight_scale_2[0]);
    }
}

extern "C" __global__ void qwen_nvfp4_quantize(
    const bf16* input,
    float input_scale_inv,
    uint8_t* packed,
    uint8_t* scales,
    unsigned int count
) {
    const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= count / 16) {
        return;
    }
    const bf16* values = input + group * 16;
    float maximum = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        maximum = fmaxf(maximum, fabsf(to_float(values[i])));
    }
    const float scale = maximum * input_scale_inv * (1.0f / 6.0f);
    uint16_t encoded_scale;
    asm(
        "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(encoded_scale)
        : "f"(scale), "f"(scale)
    );
    const uint8_t raw_scale = static_cast<uint8_t>(encoded_scale & 0xff);
    scales[group] = raw_scale;
    const float inverse = raw_scale == 0 ? 0.0f : input_scale_inv / fp8_to_float(raw_scale);
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        const float pair[2] = {
            to_float(values[2 * i]) * inverse,
            to_float(values[2 * i + 1]) * inverse,
        };
        uint8_t encoded = 0;
#pragma unroll
        for (unsigned int j = 0; j < 2; ++j) {
            const float value = pair[j];
            const float magnitude = fabsf(value);
            unsigned int code;
            if (magnitude <= 0.25f) {
                code = 0;
            } else if (magnitude < 0.75f) {
                code = 1;
            } else if (magnitude <= 1.25f) {
                code = 2;
            } else if (magnitude < 1.75f) {
                code = 3;
            } else if (magnitude <= 2.5f) {
                code = 4;
            } else if (magnitude < 3.5f) {
                code = 5;
            } else if (magnitude <= 5.0f) {
                code = 6;
            } else {
                code = 7;
            }
            if (signbit(value)) {
                code |= 8;
            }
            encoded |= code << (4 * j);
        }
        packed[group * 8 + i] = encoded;
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_gemv_w4a4(
    const uint8_t* input,
    const uint8_t* input_scales,
    const uint8_t* weight,
    const uint8_t* weight_scales,
    const float* weight_scale_2,
    float input_scale,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    if (column >= n) {
        return;
    }
    const unsigned int groups = k / 16;
    const uint64_t* input_row = reinterpret_cast<const uint64_t*>(input);
    const uint64_t* weight_row = reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(column) * (k / 2)
    );
    float sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t input_packed = input_row[group];
        const uint64_t weight_packed = weight_row[group];
        float dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const float left = fp4_to_float((input_packed >> (4 * j)) & 15);
            const float right = fp4_to_float((weight_packed >> (4 * j)) & 15);
            dot = fmaf(left, right, dot);
        }
        const float scale = fp8_to_float(input_scales[group]) *
            fp8_to_float(weight_scales[column * groups + group]);
        sum = fmaf(scale, dot, sum);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        output[column] = from_float(sum * weight_scale_2[0] * input_scale);
    }
}

extern "C" __global__ void qwen_clear_f32(float* output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = 0.0f;
    }
}

extern "C" __global__ void qwen_expert_accumulate(
    const bf16* input,
    float weight,
    float* accumulation,
    unsigned int count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float rounded_weight = to_float(from_float(weight));
        accumulation[i] = fmaf(to_float(input[i]), rounded_weight, accumulation[i]);
    }
}

extern "C" __global__ void qwen_moe_finish(
    const float* routed,
    const bf16* shared,
    const float* shared_gate,
    bf16* output,
    unsigned int count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = from_float(routed[i] + to_float(shared[i]) * sigmoid(shared_gate[0]));
    }
}

struct ExpertDeviceMatrix {
    unsigned long long weight;
    unsigned long long scales;
    unsigned long long global_scale;
};

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_gemv_indirect(
    const bf16* input,
    const ExpertDeviceMatrix* table,
    const unsigned int* expert_ids,
    unsigned int slot,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    const ExpertDeviceMatrix matrix = table[expert_ids[slot]];
    if (column >= n) {
        return;
    }
    if (matrix.weight == 0) {
        if (threadIdx.x == 0) {
            output[column] = from_float(0.0f);
        }
        return;
    }
    const uint8_t* weight = reinterpret_cast<const uint8_t*>(matrix.weight);
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(matrix.scales);
    const float* global_scale = reinterpret_cast<const float*>(matrix.global_scale);
    const unsigned int groups = k / 16;
    const uint64_t* weight_row = reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(column) * (k / 2)
    );
    float sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t packed = weight_row[group];
        const bf16* values = input + group * 16;
        float dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const unsigned int code = (packed >> (4 * j)) & 15;
            dot = fmaf(to_float(values[j]), fp4_to_float(code), dot);
        }
        sum = fmaf(fp8_to_float(scales[column * groups + group]), dot, sum);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        output[column] = from_float(sum * global_scale[0]);
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_gemv_w4a4_indirect(
    const uint8_t* input,
    const uint8_t* input_scales,
    const ExpertDeviceMatrix* table,
    const unsigned int* expert_ids,
    unsigned int slot,
    float input_scale,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    const ExpertDeviceMatrix matrix = table[expert_ids[slot]];
    if (column >= n) {
        return;
    }
    if (matrix.weight == 0) {
        if (threadIdx.x == 0) {
            output[column] = from_float(0.0f);
        }
        return;
    }
    const uint8_t* weight = reinterpret_cast<const uint8_t*>(matrix.weight);
    const uint8_t* weight_scales = reinterpret_cast<const uint8_t*>(matrix.scales);
    const float* global_scale = reinterpret_cast<const float*>(matrix.global_scale);
    const unsigned int groups = k / 16;
    const uint64_t* input_row = reinterpret_cast<const uint64_t*>(input);
    const uint64_t* weight_row = reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(column) * (k / 2)
    );
    float sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t input_packed = input_row[group];
        const uint64_t weight_packed = weight_row[group];
        float dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const float left = fp4_to_float((input_packed >> (4 * j)) & 15);
            const float right = fp4_to_float((weight_packed >> (4 * j)) & 15);
            dot = fmaf(left, right, dot);
        }
        const float scale = fp8_to_float(input_scales[group]) *
            fp8_to_float(weight_scales[column * groups + group]);
        sum = fmaf(scale, dot, sum);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        output[column] = from_float(sum * global_scale[0] * input_scale);
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_gate_up_indirect(
    const bf16* input,
    const ExpertDeviceMatrix* gate_table,
    const ExpertDeviceMatrix* up_table,
    const unsigned int* expert_ids,
    unsigned int slot,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    const ExpertDeviceMatrix gate = gate_table[expert_ids[slot]];
    const ExpertDeviceMatrix up = up_table[expert_ids[slot]];
    if (column >= n) {
        return;
    }
    if (gate.weight == 0) {
        if (threadIdx.x == 0) {
            output[column] = from_float(0.0f);
        }
        return;
    }
    const unsigned int groups = k / 16;
    const uint64_t* gate_row = reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const uint8_t*>(gate.weight) + static_cast<size_t>(column) * (k / 2)
    );
    const uint64_t* up_row = reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const uint8_t*>(up.weight) + static_cast<size_t>(column) * (k / 2)
    );
    const uint8_t* gate_scales = reinterpret_cast<const uint8_t*>(gate.scales);
    const uint8_t* up_scales = reinterpret_cast<const uint8_t*>(up.scales);
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t gate_packed = gate_row[group];
        const uint64_t up_packed = up_row[group];
        const bf16* values = input + group * 16;
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const float value = to_float(values[j]);
            gate_dot = fmaf(value, fp4_to_float((gate_packed >> (4 * j)) & 15), gate_dot);
            up_dot = fmaf(value, fp4_to_float((up_packed >> (4 * j)) & 15), up_dot);
        }
        gate_sum = fmaf(fp8_to_float(gate_scales[column * groups + group]), gate_dot, gate_sum);
        up_sum = fmaf(fp8_to_float(up_scales[column * groups + group]), up_dot, up_sum);
    }
    __shared__ float reduction[32];
    gate_sum = block_sum(gate_sum, reduction);
    up_sum = block_sum(up_sum, reduction);
    if (threadIdx.x == 0) {
        const float gate_global = reinterpret_cast<const float*>(gate.global_scale)[0];
        const float up_global = reinterpret_cast<const float*>(up.global_scale)[0];
        const bf16 rounded_gate = from_float(gate_sum * gate_global);
        const bf16 rounded_up = from_float(up_sum * up_global);
        const float g = to_float(rounded_gate);
        output[column] = from_float(g / (1.0f + expf(-g)) * to_float(rounded_up));
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_nvfp4_down_accumulate_indirect(
    const bf16* input,
    const ExpertDeviceMatrix* table,
    const unsigned int* expert_ids,
    unsigned int slot,
    const float* expert_weights,
    float* accumulation,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    const ExpertDeviceMatrix matrix = table[expert_ids[slot]];
    if (column >= n || matrix.weight == 0) {
        return;
    }
    const uint8_t* weight = reinterpret_cast<const uint8_t*>(matrix.weight);
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(matrix.scales);
    const float* global_scale = reinterpret_cast<const float*>(matrix.global_scale);
    const unsigned int groups = k / 16;
    const uint64_t* weight_row = reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(column) * (k / 2)
    );
    float sum = 0.0f;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint64_t packed = weight_row[group];
        const bf16* values = input + group * 16;
        float dot = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            const unsigned int code = (packed >> (4 * j)) & 15;
            dot = fmaf(to_float(values[j]), fp4_to_float(code), dot);
        }
        sum = fmaf(fp8_to_float(scales[column * groups + group]), dot, sum);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        const bf16 rounded = from_float(sum * global_scale[0]);
        const float rounded_weight = to_float(from_float(expert_weights[slot]));
        accumulation[column] = fmaf(to_float(rounded), rounded_weight, accumulation[column]);
    }
}

extern "C" __global__ void qwen_expert_accumulate_indirect(
    const bf16* input,
    const float* weights,
    unsigned int slot,
    float* accumulation,
    unsigned int count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float rounded_weight = to_float(from_float(weights[slot]));
        accumulation[i] = fmaf(to_float(input[i]), rounded_weight, accumulation[i]);
    }
}


extern "C" __global__ void qwen_peer_sum(
    unsigned long long p0,
    unsigned long long p1,
    unsigned long long p2,
    unsigned long long p3,
    unsigned long long p4,
    unsigned long long p5,
    unsigned long long p6,
    unsigned long long p7,
    unsigned int world_size,
    float* output,
    unsigned int count
) {
    const unsigned long long peers[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float value = 0.0f;
    for (unsigned int rank = 0; rank < world_size; ++rank) {
        value += reinterpret_cast<const float*>(peers[rank])[i];
    }
    output[i] = value;
}
