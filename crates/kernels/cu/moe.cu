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
    __syncthreads();
    if (expert == 0) {
        float selected_sum = 0.0f;
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            float best = -1.0f;
            unsigned int best_expert = experts;
            for (unsigned int candidate = 0; candidate < experts; ++candidate) {
                if (probabilities[candidate] > best) {
                    best = probabilities[candidate];
                    best_expert = candidate;
                }
            }
            expert_ids[slot] = best_expert;
            expert_weights[slot] = best;
            probabilities[best_expert] = -1.0f;
            selected_sum += best;
        }
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            expert_weights[slot] /= selected_sum;
        }
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
