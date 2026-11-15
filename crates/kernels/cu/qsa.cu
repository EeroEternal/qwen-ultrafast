#include "common.cuh"

extern "C" __global__ void __launch_bounds__(256) qwen_qsa_prepare(
    const bf16* projected_query,
    const bf16* projected_key,
    const bf16* projected_value,
    const bf16* query_norm,
    const bf16* key_norm,
    bf16* query,
    bf16* gate,
    bf16* key_cache,
    bf16* value_cache,
    const State* state,
    const float* cosines,
    const float* sines,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int rotary_dim,
    unsigned int max_sequence,
    float epsilon
) {
    __shared__ float values[256];
    __shared__ float reduction[32];
    const unsigned int head = blockIdx.x;
    const unsigned int position = state->position;
    if (head >= query_heads + 2 * kv_heads) {
        return;
    }
    if (head >= query_heads + kv_heads) {
        const unsigned int kv_head = head - query_heads - kv_heads;
        const bf16* source = projected_value + kv_head * head_dim;
        bf16* destination = value_cache + (kv_head * max_sequence + position) * head_dim;
        for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            destination[d] = source[d];
        }
        return;
    }
    const bool is_query = head < query_heads;
    const bf16* source = is_query ? projected_query + head * 2 * head_dim : projected_key + (head - query_heads) * head_dim;
    const bf16* norm = is_query ? query_norm : key_norm;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = to_float(source[d]);
    }
    float square_sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        square_sum = fmaf(values[d], values[d], square_sum);
    }
    const float inverse_rms = rsqrtf(block_sum(square_sum, reduction) / head_dim + epsilon);
    const unsigned int kv_head = head - query_heads;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = values[d] * inverse_rms * (1.0f + to_float(norm[d]));
        if (d < rotary_dim) {
            const unsigned int pair = d < rotary_dim / 2 ? d + rotary_dim / 2 : d - rotary_dim / 2;
            const float other = values[pair] * inverse_rms * (1.0f + to_float(norm[pair]));
            const float cosine = cosines[position * (rotary_dim / 2) + d % (rotary_dim / 2)];
            const float sine = sines[position * (rotary_dim / 2) + d % (rotary_dim / 2)];
            value = d < rotary_dim / 2 ? value * cosine - other * sine : value * cosine + other * sine;
        }
        if (is_query) {
            query[head * head_dim + d] = from_float(value);
        } else {
            key_cache[(kv_head * max_sequence + position) * head_dim + d] = from_float(value);
        }
    }
    if (is_query) {
        const bf16* source_gate = projected_query + head * 2 * head_dim + head_dim;
        for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            gate[head * head_dim + d] = source_gate[d];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) qwen_qsa_attention_reduce(
    const float* partial_values,
    const float* partial_max,
    const float* partial_sum,
    const bf16* gate,
    bf16* output,
    unsigned int query_heads,
    unsigned int head_dim,
    unsigned int splits
) {
    __shared__ float reduction[32];
    const unsigned int query_head = blockIdx.x;
    float local_max = -INFINITY;
    for (unsigned int split = threadIdx.x; split < splits; split += blockDim.x) {
        const unsigned int index = split * query_heads + query_head;
        if (partial_sum[index] > 0.0f) {
            local_max = fmaxf(local_max, partial_max[index]);
        }
    }
    const float maximum = block_max(local_max, reduction);
    float sum = 0.0f;
    for (unsigned int split = 0; split < splits; ++split) {
        const unsigned int index = split * query_heads + query_head;
        if (partial_sum[index] > 0.0f) {
            sum += partial_sum[index] * expf(partial_max[index] - maximum);
        }
    }
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (unsigned int split = 0; split < splits; ++split) {
            const unsigned int index = split * query_heads + query_head;
            if (partial_sum[index] > 0.0f) {
                const float weight = expf(partial_max[index] - maximum);
                const size_t offset = (split * query_heads + query_head) * head_dim + d;
                value = fmaf(partial_values[offset], weight, value);
            }
        }
        const bf16 rounded = from_float(value / sum);
        output[query_head * head_dim + d] = from_float(
            to_float(rounded) * sigmoid(to_float(gate[query_head * head_dim + d]))
        );
    }
}
