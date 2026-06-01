#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <int BITS>
__device__ __forceinline__ int unpack_value(const uint8_t* __restrict__ packed, int idx) {
    constexpr int qmin = -(1 << (BITS - 1));
    int bit_pos = idx * BITS;
    unsigned int encoded = 0u;
    #pragma unroll
    for (int b = 0; b < BITS; ++b) {
        int src = bit_pos + b;
        if ((packed[src >> 3] & (uint8_t)(1u << (src & 7))) != 0) {
            encoded |= 1u << b;
        }
    }
    return (int)encoded + qmin;
}

template <int BITS>
__device__ __forceinline__ void pack_signed_per_row_body(
    const int8_t* __restrict__ values,
    uint8_t* __restrict__ packed,
    int count
) {
    constexpr int qmin = -(1 << (BITS - 1));
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bytes = (count * BITS + 7) >> 3;
    if (byte_idx >= bytes) return;
    uint8_t out = 0;
    #pragma unroll
    for (int dst_bit = 0; dst_bit < 8; ++dst_bit) {
        int global_bit = (byte_idx << 3) + dst_bit;
        int value_idx = global_bit / BITS;
        if (value_idx < count) {
            int value_bit = global_bit - value_idx * BITS;
            unsigned int encoded = (unsigned int)((int)values[value_idx] - qmin);
            if (((encoded >> value_bit) & 1u) != 0) {
                out |= (uint8_t)(1u << dst_bit);
            }
        }
    }
    packed[byte_idx] = out;
}

template <int BITS>
__device__ __forceinline__ void dequant_per_row_f16_scale_body(
    const uint8_t* __restrict__ packed,
    const uint16_t* __restrict__ scales,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    float scale = __half2float(__ushort_as_half(scales[row]));
    out[idx] = (float)unpack_value<BITS>(packed, idx) * scale;
}

#define DECLARE_QUANT_KERNELS(BITS) \
extern "C" __global__ void pack_signed_i##BITS##_per_row_sm90( \
    const int8_t* values, uint8_t* packed, int count \
) { \
    pack_signed_per_row_body<BITS>(values, packed, count); \
} \
extern "C" __global__ void dequant_i##BITS##_per_row_f16_scale_sm90( \
    const uint8_t* packed, const uint16_t* scales, float* out, int rows, int cols \
) { \
    dequant_per_row_f16_scale_body<BITS>(packed, scales, out, rows, cols); \
}

DECLARE_QUANT_KERNELS(4)
DECLARE_QUANT_KERNELS(5)
DECLARE_QUANT_KERNELS(6)
DECLARE_QUANT_KERNELS(7)
DECLARE_QUANT_KERNELS(8)
