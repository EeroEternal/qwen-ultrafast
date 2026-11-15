#include "common.cuh"

extern "C" __global__ void qwen_hc_down(
    const bf16* input,
    bf16* output,
    unsigned int count,
    unsigned int streams
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = from_float(silu(to_float(input[i]) / static_cast<float>(streams)));
    }
}

extern "C" __global__ void qwen_hc_mix(
    const bf16* normed,
    const bf16* up,
    bf16* output,
    unsigned int hidden_size,
    unsigned int streams
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) {
        return;
    }
    float value = 0.0f;
    for (unsigned int stream = 0; stream < streams; ++stream) {
        const unsigned int offset = stream * hidden_size + i;
        value += sigmoid(to_float(up[offset])) * to_float(normed[offset]);
    }
    output[i] = from_float(value / static_cast<float>(streams));
}

extern "C" __global__ void qwen_hc_inject(
    bf16* hyper_input,
    const bf16* block_output,
    const bf16* injection_weights,
    unsigned int hidden_size,
    unsigned int streams
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) {
        return;
    }
    for (unsigned int stream = 0; stream < streams; ++stream) {
        const float weight = 2.0f * sigmoid(
            to_float(injection_weights[stream]) / static_cast<float>(streams)
        );
        const unsigned int offset = stream * hidden_size + i;
        hyper_input[offset] = from_float(
            to_float(hyper_input[offset]) + to_float(block_output[i]) * weight
        );
    }
}
