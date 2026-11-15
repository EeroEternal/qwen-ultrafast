#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

struct State {
    unsigned int token;
    unsigned int position;
    unsigned int step;
    unsigned int padding;
};

__device__ __forceinline__ float to_float(bf16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ bf16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        value = lane < warps ? shared[lane] : 0.0f;
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        value = lane < warps ? shared[lane] : -INFINITY;
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
        }
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float dot8(const uint4& lhs, const uint4& rhs) {
    const bf16* a = reinterpret_cast<const bf16*>(&lhs);
    const bf16* b = reinterpret_cast<const bf16*>(&rhs);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum = fmaf(to_float(a[i]), to_float(b[i]), sum);
    }
    return sum;
}

extern "C" __global__ void qwen_embedding(
    const bf16* embedding,
    const State* state,
    bf16* output,
    unsigned int hidden_size
) {
    const uint4* source = reinterpret_cast<const uint4*>(
        embedding + static_cast<size_t>(state->token) * hidden_size
    );
    uint4* destination = reinterpret_cast<uint4*>(output);
    for (unsigned int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
        destination[i] = source[i];
    }
}

template <bool FLOAT_OUTPUT>
__device__ __forceinline__ void gemv(
    const bf16* input,
    const bf16* weight,
    void* output,
    unsigned int n,
    unsigned int k
) {
    const unsigned int column = blockIdx.x;
    if (column >= n) {
        return;
    }
    const uint4* input_vec = reinterpret_cast<const uint4*>(input);
    const uint4* weight_vec = reinterpret_cast<const uint4*>(
        weight + static_cast<size_t>(column) * k
    );
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < k / 8; i += blockDim.x) {
        sum += dot8(input_vec[i], weight_vec[i]);
    }
    __shared__ float reduction[32];
    sum = block_sum(sum, reduction);
    if (threadIdx.x == 0) {
        if constexpr (FLOAT_OUTPUT) {
            reinterpret_cast<float*>(output)[column] = sum;
        } else {
            reinterpret_cast<bf16*>(output)[column] = from_float(sum);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) qwen_gemv_bf16(
    const bf16* input,
    const bf16* weight,
    bf16* output,
    unsigned int n,
    unsigned int k
) {
    gemv<false>(input, weight, output, n, k);
}

extern "C" __global__ void __launch_bounds__(128) qwen_gemv_f32(
    const bf16* input,
    const bf16* weight,
    float* output,
    unsigned int n,
    unsigned int k
) {
    gemv<true>(input, weight, output, n, k);
}

extern "C" __global__ void __launch_bounds__(256) qwen_rmsnorm(
    const bf16* input,
    const bf16* weight,
    bf16* output,
    unsigned int columns,
    float epsilon
) {
    const uint4* input_vec = reinterpret_cast<const uint4*>(input);
    float square_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < columns / 8; i += blockDim.x) {
        const uint4 packed = input_vec[i];
        const bf16* values = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float value = to_float(values[j]);
            square_sum = fmaf(value, value, square_sum);
        }
    }
    __shared__ float reduction[32];
    const float inverse_rms = rsqrtf(
        block_sum(square_sum, reduction) / static_cast<float>(columns) + epsilon
    );
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) {
        output[i] = from_float(to_float(input[i]) * inverse_rms * (1.0f + to_float(weight[i])));
    }
}

extern "C" __global__ void qwen_add(
    const bf16* lhs,
    const bf16* rhs,
    bf16* output,
    unsigned int count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / 8) {
        return;
    }
    const uint4 a = reinterpret_cast<const uint4*>(lhs)[i];
    const uint4 b = reinterpret_cast<const uint4*>(rhs)[i];
    uint4 result;
    const bf16* av = reinterpret_cast<const bf16*>(&a);
    const bf16* bv = reinterpret_cast<const bf16*>(&b);
    bf16* values = reinterpret_cast<bf16*>(&result);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        values[j] = from_float(to_float(av[j]) + to_float(bv[j]));
    }
    reinterpret_cast<uint4*>(output)[i] = result;
}

extern "C" __global__ void qwen_swiglu(
    const bf16* gate_up,
    bf16* output,
    unsigned int intermediate_size
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= intermediate_size / 8) {
        return;
    }
    const uint4 gate = reinterpret_cast<const uint4*>(gate_up)[i];
    const uint4 up = reinterpret_cast<const uint4*>(gate_up + intermediate_size)[i];
    uint4 result;
    const bf16* gv = reinterpret_cast<const bf16*>(&gate);
    const bf16* uv = reinterpret_cast<const bf16*>(&up);
    bf16* values = reinterpret_cast<bf16*>(&result);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = to_float(gv[j]);
        values[j] = from_float(g / (1.0f + expf(-g)) * to_float(uv[j]));
    }
    reinterpret_cast<uint4*>(output)[i] = result;
}

extern "C" __global__ void __launch_bounds__(256) qwen_qkv_prepare(
    const bf16* qkv,
    const bf16* q_norm,
    const bf16* k_norm,
    bf16* query,
    bf16* key_cache,
    bf16* value_cache,
    const State* state,
    const float* cosines,
    const float* sines,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
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
        const bf16* source = qkv + (query_heads + kv_heads + kv_head) * head_dim;
        bf16* destination = value_cache + (kv_head * max_sequence + position) * head_dim;
        for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            destination[d] = source[d];
        }
        return;
    }
    const bool is_query = head < query_heads;
    const bf16* source = qkv + head * head_dim;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        values[d] = to_float(source[d]);
    }
    float square_sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        square_sum = fmaf(values[d], values[d], square_sum);
    }
    const float inverse_rms = rsqrtf(
        block_sum(square_sum, reduction) / static_cast<float>(head_dim) + epsilon
    );
    const bf16* norm = is_query ? q_norm : k_norm;
    const unsigned int half = head_dim / 2;
    const unsigned int kv_head = head - query_heads;
    for (unsigned int d = threadIdx.x; d < half; d += blockDim.x) {
        const float first = values[d] * inverse_rms * to_float(norm[d]);
        const float second = values[d + half] * inverse_rms * to_float(norm[d + half]);
        const float cosine = cosines[position * half + d];
        const float sine = sines[position * half + d];
        const float rotated_first = first * cosine - second * sine;
        const float rotated_second = second * cosine + first * sine;
        if (is_query) {
            query[head * head_dim + d] = from_float(rotated_first);
            query[head * head_dim + d + half] = from_float(rotated_second);
        } else {
            bf16* destination = key_cache + (kv_head * max_sequence + position) * head_dim;
            destination[d] = from_float(rotated_first);
            destination[d + half] = from_float(rotated_second);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) qwen_attention_partial(
    const bf16* query,
    const bf16* key_cache,
    const bf16* value_cache,
    const State* state,
    float* partial_values,
    float* partial_max,
    float* partial_sum,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int max_sequence,
    unsigned int splits,
    float scale
) {
    extern __shared__ float scores[];
    __shared__ float query_values[256];
    __shared__ float reduction[32];
    const unsigned int query_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int sequence = state->position + 1;
    const unsigned int chunk = (sequence + splits - 1) / splits;
    const unsigned int begin = split * chunk;
    const unsigned int end = min(sequence, begin + chunk);
    const unsigned int local_length = begin < end ? end - begin : 0;
    const unsigned int kv_head = query_head / (query_heads / kv_heads);
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        query_values[d] = to_float(query[query_head * head_dim + d]);
    }
    __syncthreads();
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warps = blockDim.x >> 5;
    for (unsigned int local_key = warp; local_key < local_length; local_key += warps) {
        const unsigned int key = begin + local_key;
        const bf16* key_row = key_cache + (kv_head * max_sequence + key) * head_dim;
        float score = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            score = fmaf(query_values[d], to_float(key_row[d]), score);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xffffffffu, score, offset);
        }
        if (lane == 0) {
            scores[local_key] = score * scale;
        }
    }
    __syncthreads();
    float local_max = -INFINITY;
    for (unsigned int key = threadIdx.x; key < local_length; key += blockDim.x) {
        local_max = fmaxf(local_max, scores[key]);
    }
    const float maximum = block_max(local_max, reduction);
    float local_sum = 0.0f;
    for (unsigned int key = threadIdx.x; key < local_length; key += blockDim.x) {
        const float weight = expf(scores[key] - maximum);
        scores[key] = weight;
        local_sum += weight;
    }
    const float sum = block_sum(local_sum, reduction);
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (unsigned int local_key = 0; local_key < local_length; ++local_key) {
            const unsigned int key = begin + local_key;
            const size_t offset = (kv_head * max_sequence + key) * head_dim + d;
            value = fmaf(scores[local_key], to_float(value_cache[offset]), value);
        }
        partial_values[(split * query_heads + query_head) * head_dim + d] = value;
    }
    if (threadIdx.x == 0) {
        partial_max[split * query_heads + query_head] = maximum;
        partial_sum[split * query_heads + query_head] = sum;
    }
}

extern "C" __global__ void __launch_bounds__(256) qwen_attention_reduce(
    const float* partial_values,
    const float* partial_max,
    const float* partial_sum,
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
        output[query_head * head_dim + d] = from_float(value / sum);
    }
}

extern "C" __global__ void __launch_bounds__(1024) qwen_argmax(
    const float* logits,
    unsigned int* next_token,
    unsigned int vocabulary
) {
    __shared__ float values[1024];
    __shared__ unsigned int indices[1024];
    float best = -INFINITY;
    unsigned int index = vocabulary;
    for (unsigned int i = threadIdx.x; i < vocabulary; i += blockDim.x) {
        if (logits[i] > best) {
            best = logits[i];
            index = i;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = index;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const unsigned int other = threadIdx.x + stride;
            if (values[other] > values[threadIdx.x] ||
                (values[other] == values[threadIdx.x] && indices[other] < indices[threadIdx.x])) {
                values[threadIdx.x] = values[other];
                indices[threadIdx.x] = indices[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        next_token[0] = indices[0];
    }
}

extern "C" __global__ void qwen_advance(
    State* state,
    const unsigned int* next_token,
    unsigned int* output_tokens
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned int token = next_token[0];
        output_tokens[state->step] = token;
        state->token = token;
        state->position += 1;
        state->step += 1;
    }
}
