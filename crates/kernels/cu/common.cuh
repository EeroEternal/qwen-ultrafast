#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ float fp8_to_float(uint8_t value) {
    const uint16_t packed = value;
    uint32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(packed));
    return __half2float(__ushort_as_half(static_cast<unsigned short>(halves & 0xffff)));
}

__device__ __forceinline__ float fp4_to_float(unsigned int value) {
    const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return values[value & 15];
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

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float silu(float value) {
    return value * sigmoid(value);
}
