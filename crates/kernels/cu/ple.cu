#include "common.cuh"

extern "C" __global__ void qwen_hc_expand(
    const bf16* input,
    bf16* output,
    unsigned int hidden_size,
    unsigned int streams
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) {
        return;
    }
    for (unsigned int stream = 0; stream < streams; ++stream) {
        output[stream * hidden_size + i] = input[i];
    }
}

extern "C" __global__ void qwen_ple_hash(
    unsigned int token,
    unsigned int* history,
    const unsigned long long* multipliers,
    const unsigned long long* vocabulary_sizes,
    const unsigned long long* offsets,
    unsigned long long* row_ids,
    unsigned int eos_token
) {
    const unsigned int head = threadIdx.x;
    if (head >= 16) {
        return;
    }
    __shared__ unsigned int previous;
    __shared__ unsigned int older;
    if (head == 0) {
        const uint2 old = *reinterpret_cast<const uint2*>(history);
        previous = old.x;
        older = old.y;
        const uint2 updated = make_uint2(token, old.x);
        *reinterpret_cast<uint2*>(history) = updated;
    }
    __syncthreads();
    const unsigned int shifted_one = previous == eos_token ? eos_token : previous;
    const unsigned int shifted_two = previous == eos_token ? eos_token : older;
    unsigned long long mixed = static_cast<unsigned long long>(token) * multipliers[0];
    mixed ^= static_cast<unsigned long long>(shifted_one) * multipliers[1];
    if (head >= 8) {
        mixed ^= static_cast<unsigned long long>(shifted_two) * multipliers[2];
    }
    const long long signed_mixed = static_cast<long long>(mixed);
    const long long vocabulary_size = static_cast<long long>(vocabulary_sizes[head]);
    long long remainder = signed_mixed % vocabulary_size;
    if (remainder < 0) {
        remainder += vocabulary_size;
    }
    row_ids[head] = static_cast<unsigned long long>(remainder) + offsets[head];
}

extern "C" __global__ void __launch_bounds__(256) qwen_ple_gate(
    const bf16* key,
    const bf16* query,
    const bf16* value,
    bf16* output,
    unsigned int hidden_size
) {
    __shared__ float reduction[32];
    const unsigned int stream = blockIdx.x;
    const bf16* stream_key = key + stream * hidden_size;
    const bf16* stream_query = query + stream * hidden_size;
    float dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dot = fmaf(to_float(stream_key[i]), to_float(stream_query[i]), dot);
    }
    float gate = block_sum(dot, reduction) * rsqrtf(static_cast<float>(hidden_size));
    gate = sqrtf(fmaxf(fabsf(gate), 1e-6f)) * (gate < 0.0f ? -1.0f : 1.0f);
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        output[stream * hidden_size + i] = from_float(sigmoid(gate) * to_float(value[i]));
    }
}

extern "C" __global__ void qwen_ple_conv(
    const bf16* gated,
    const bf16* normed,
    const bf16* weight,
    bf16* convolution_state,
    bf16* output,
    unsigned int channels
) {
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    bf16* history = convolution_state + channel * 9;
    float value = to_float(weight[channel * 4 + 3]) * to_float(normed[channel]);
    value += to_float(weight[channel * 4]) * to_float(history[0]);
    value += to_float(weight[channel * 4 + 1]) * to_float(history[3]);
    value += to_float(weight[channel * 4 + 2]) * to_float(history[6]);
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        history[i] = history[i + 1];
    }
    history[8] = normed[channel];
    output[channel] = from_float(to_float(gated[channel]) + silu(value));
}
