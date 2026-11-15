#include "common.cuh"

extern "C" __global__ void qwen_gdn_conv(
    const bf16* input,
    const bf16* weight,
    bf16* convolution_state,
    bf16* output,
    unsigned int channels
) {
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    bf16* history = convolution_state + channel * 3;
    const float h0 = to_float(history[0]);
    const float h1 = to_float(history[1]);
    const float h2 = to_float(history[2]);
    const float h3 = to_float(input[channel]);
    const bf16* w = weight + channel * 4;
    const float value = h0 * to_float(w[0]) + h1 * to_float(w[1]) +
        h2 * to_float(w[2]) + h3 * to_float(w[3]);
    history[0] = history[1];
    history[1] = history[2];
    history[2] = input[channel];
    output[channel] = from_float(silu(value));
}

extern "C" __global__ void __launch_bounds__(1024) qwen_gdn_recurrence(
    const bf16* qkv,
    const bf16* z,
    const bf16* a,
    const bf16* b,
    const float* a_log,
    const float* dt_bias,
    const bf16* norm,
    float* recurrent_state,
    bf16* output,
    bf16* raw_output,
    unsigned int key_heads,
    unsigned int value_heads,
    unsigned int head_dim,
    float epsilon
) {
    __shared__ float query[128];
    __shared__ float key[128];
    __shared__ float reduction[32];
    __shared__ float delta_values[128];
    __shared__ float raw_values[128];
    __shared__ float decay;
    __shared__ float beta;
    const unsigned int head = blockIdx.x;
    const unsigned int key_head = head / (value_heads / key_heads);
    const unsigned int key_dim = key_heads * head_dim;
    const unsigned int tid = threadIdx.x;
    const unsigned int value_index = tid & 127;
    const unsigned int group = tid >> 7;
    if (tid < head_dim) {
        query[tid] = to_float(qkv[key_head * head_dim + tid]);
        key[tid] = to_float(qkv[key_dim + key_head * head_dim + tid]);
    }
    float q_square = 0.0f;
    float k_square = 0.0f;
    if (tid < head_dim) {
        q_square = query[tid] * query[tid];
        k_square = key[tid] * key[tid];
    }
    const float q_norm = rsqrtf(block_sum(q_square, reduction) + epsilon);
    const float k_norm = rsqrtf(block_sum(k_square, reduction) + epsilon);
    if (tid < head_dim) {
        query[tid] *= q_norm * rsqrtf(static_cast<float>(head_dim));
        key[tid] *= k_norm;
    }
    if (tid == 0) {
        decay = expf(-expf(a_log[head]) * log1pf(expf(to_float(a[head]) + dt_bias[head])));
        beta = sigmoid(to_float(b[head]));
    }
    __syncthreads();
    float* state = recurrent_state + static_cast<size_t>(head) * head_dim * head_dim;
    for (unsigned int k = group; k < head_dim; k += 8) {
        state[k * head_dim + value_index] *= decay;
    }
    __syncthreads();
    if (group == 0) {
        float memory = 0.0f;
        for (unsigned int k = 0; k < head_dim; ++k) {
            memory = fmaf(state[k * head_dim + value_index], key[k], memory);
        }
        const float value = to_float(qkv[2 * key_dim + head * head_dim + value_index]);
        delta_values[value_index] = (value - memory) * beta;
    }
    __syncthreads();
    const float delta = delta_values[value_index];
    for (unsigned int k = group; k < head_dim; k += 8) {
        state[k * head_dim + value_index] = fmaf(key[k], delta, state[k * head_dim + value_index]);
    }
    __syncthreads();
    if (group == 0) {
        float result = 0.0f;
        for (unsigned int k = 0; k < head_dim; ++k) {
            result = fmaf(state[k * head_dim + value_index], query[k], result);
        }
        const bf16 rounded = from_float(result);
        raw_values[value_index] = to_float(rounded);
        if (raw_output != nullptr) {
            raw_output[head * head_dim + value_index] = rounded;
        }
    }
    const float variance_input = group == 0 ? raw_values[value_index] : 0.0f;
    const float inverse_rms = rsqrtf(block_sum(variance_input * variance_input, reduction) /
        static_cast<float>(head_dim) + epsilon);
    if (group == 0) {
        const bf16 normalized = from_float(variance_input * inverse_rms);
        const float gated = to_float(normalized) * to_float(norm[value_index]) *
            sigmoid(to_float(z[head * head_dim + value_index]));
        output[head * head_dim + value_index] = from_float(gated);
    }
}
