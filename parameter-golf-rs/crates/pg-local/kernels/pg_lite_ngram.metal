#include <metal_stdlib>
using namespace metal;

struct PgLiteEvalConfig {
    uint token_count;
    uint residual_buckets;
    uint context;
    float residual_weight;
};

static inline uint pg_lite_context_bucket(
    device const uchar* bytes,
    uint idx,
    uint context,
    uint buckets
) {
    uint start = idx > min(context, 32u) ? idx - min(context, 32u) : 0u;
    uint hash = 2166136261u;
    for (uint pos = start; pos < idx; ++pos) {
        hash ^= uint(bytes[pos]);
        hash *= 16777619u;
    }
    return buckets == 0u ? 0u : hash % buckets;
}

kernel void pg_lite_ngram_loss_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const ushort* residual_counts [[buffer(2)]],
    constant PgLiteEvalConfig& cfg [[buffer(3)]],
    device float* token_losses [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cfg.token_count) {
        return;
    }

    uint byte = uint(bytes[gid]);
    uint prev = gid == 0u ? 256u : uint(bytes[gid - 1u]);
    uint bigram_offset = prev * 256u;
    uint bigram_sum = 0u;
    for (uint col = 0u; col < 256u; ++col) {
        bigram_sum += bigram_counts[bigram_offset + col];
    }
    float p_bigram = float(bigram_counts[bigram_offset + byte]) / max(float(bigram_sum), 1.0f);

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    uint residual_offset = bucket * 256u;
    uint residual_sum = 0u;
    for (uint col = 0u; col < 256u; ++col) {
        residual_sum += uint(residual_counts[residual_offset + col]);
    }
    float p_residual = float(residual_counts[residual_offset + byte]) / max(float(residual_sum), 1.0f);
    float p = mix(p_bigram, p_residual, cfg.residual_weight);
    token_losses[gid] = -log(max(p, 1.0e-12f));
}

kernel void pg_lite_ngram_loss_u32_residual_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const uint* residual_counts [[buffer(2)]],
    constant PgLiteEvalConfig& cfg [[buffer(3)]],
    device float* token_losses [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cfg.token_count) {
        return;
    }

    uint byte = uint(bytes[gid]);
    uint prev = gid == 0u ? 256u : uint(bytes[gid - 1u]);
    uint bigram_offset = prev * 256u;
    uint bigram_sum = 0u;
    for (uint col = 0u; col < 256u; ++col) {
        bigram_sum += bigram_counts[bigram_offset + col];
    }
    float p_bigram = float(bigram_counts[bigram_offset + byte]) / max(float(bigram_sum), 1.0f);

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    uint residual_offset = bucket * 256u;
    uint residual_sum = 0u;
    for (uint col = 0u; col < 256u; ++col) {
        residual_sum += residual_counts[residual_offset + col];
    }
    float p_residual = float(residual_counts[residual_offset + byte]) / max(float(residual_sum), 1.0f);
    float p = mix(p_bigram, p_residual, cfg.residual_weight);
    token_losses[gid] = -log(max(p, 1.0e-12f));
}

kernel void pg_lite_ngram_train_kernel(
    device const uchar* bytes [[buffer(0)]],
    device atomic_uint* bigram_counts [[buffer(1)]],
    device atomic_uint* residual_counts [[buffer(2)]],
    constant PgLiteEvalConfig& cfg [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cfg.token_count) {
        return;
    }

    uint byte = uint(bytes[gid]);
    uint prev = gid == 0u ? 256u : uint(bytes[gid - 1u]);
    atomic_fetch_add_explicit(
        &bigram_counts[prev * 256u + byte],
        1u,
        memory_order_relaxed
    );

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    atomic_fetch_add_explicit(
        &residual_counts[bucket * 256u + byte],
        1u,
        memory_order_relaxed
    );
}

kernel void pg_lite_reduce_loss_kernel(
    device const float* token_losses [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint groups_per_grid [[threadgroups_per_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (tid >= 256u) {
        return;
    }
    threadgroup float scratch[256];
    float sum = 0.0f;
    for (uint idx = gid * threads_per_group + tid; idx < token_count; idx += threads_per_group * groups_per_grid) {
        sum += token_losses[idx];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_sums[gid] = scratch[0];
    }
}
