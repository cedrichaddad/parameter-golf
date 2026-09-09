#include <metal_stdlib>
using namespace metal;

constant uint PG_LITE_BYTE_VALUES = 256u;
constant uint PG_LITE_BIGRAM_ROWS = 257u;
constant uint PG_LITE_BOS_ROW = 256u;
constant uint PG_LITE_THREADS = 256u;
constant ulong PG_LITE_FNV_OFFSET = 0xcbf29ce484222325UL;
constant ulong PG_LITE_FNV_PRIME = 0x100000001b3UL;

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
    uint window = min(context, 32u);
    uint start = idx > window ? idx - window : 0u;
    ulong hash = PG_LITE_FNV_OFFSET;
    for (uint pos = start; pos < idx; ++pos) {
        hash ^= ulong(bytes[pos]);
        hash *= PG_LITE_FNV_PRIME;
    }
    return buckets == 0u ? 0u : uint(hash % ulong(buckets));
}

static inline uint pg_lite_prev_byte(device const uchar* bytes, uint idx) {
    return idx == 0u ? PG_LITE_BOS_ROW : uint(bytes[idx - 1u]);
}

static inline float pg_lite_loss_from_prob(float p) {
    return -log(max(p, 1.0e-12f));
}

static inline float pg_lite_mix_prob(float p_bigram, float p_residual, float residual_weight) {
    float w = clamp(residual_weight, 0.0f, 0.95f);
    return p_bigram * (1.0f - w) + p_residual * w;
}

static inline float pg_lite_prob_u16_residual(
    uint byte,
    uint prev,
    uint bucket,
    device const uint* bigram_counts,
    device const uint* bigram_sums,
    device const ushort* residual_counts,
    device const uint* residual_sums,
    float residual_weight
) {
    uint bigram_offset = prev * PG_LITE_BYTE_VALUES;
    uint residual_offset = bucket * PG_LITE_BYTE_VALUES;
    float p_bigram = float(bigram_counts[bigram_offset + byte]) /
        max(float(bigram_sums[prev]), 1.0f);
    float p_residual = float(residual_counts[residual_offset + byte]) /
        max(float(residual_sums[bucket]), 1.0f);
    return pg_lite_mix_prob(p_bigram, p_residual, residual_weight);
}

static inline float pg_lite_prob_u32_residual(
    uint byte,
    uint prev,
    uint bucket,
    device const uint* bigram_counts,
    device const uint* bigram_sums,
    device const uint* residual_counts,
    device const uint* residual_sums,
    float residual_weight
) {
    uint bigram_offset = prev * PG_LITE_BYTE_VALUES;
    uint residual_offset = bucket * PG_LITE_BYTE_VALUES;
    float p_bigram = float(bigram_counts[bigram_offset + byte]) /
        max(float(bigram_sums[prev]), 1.0f);
    float p_residual = float(residual_counts[residual_offset + byte]) /
        max(float(residual_sums[bucket]), 1.0f);
    return pg_lite_mix_prob(p_bigram, p_residual, residual_weight);
}

kernel void pg_lite_init_bigram_counts_kernel(
    device uint* bigram_counts [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < PG_LITE_BIGRAM_ROWS * PG_LITE_BYTE_VALUES) {
        bigram_counts[gid] = 1u;
    }
}

kernel void pg_lite_init_residual_counts_u16_kernel(
    device ushort* residual_counts [[buffer(0)]],
    constant uint& residual_buckets [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < residual_buckets * PG_LITE_BYTE_VALUES) {
        residual_counts[gid] = ushort(1u);
    }
}

kernel void pg_lite_init_residual_counts_u32_kernel(
    device uint* residual_counts [[buffer(0)]],
    constant uint& residual_buckets [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < residual_buckets * PG_LITE_BYTE_VALUES) {
        residual_counts[gid] = 1u;
    }
}

kernel void pg_lite_bigram_row_sums_kernel(
    device const uint* bigram_counts [[buffer(0)]],
    device uint* bigram_sums [[buffer(1)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= PG_LITE_BIGRAM_ROWS) {
        return;
    }
    uint offset = row * PG_LITE_BYTE_VALUES;
    uint sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        sum += bigram_counts[offset + col];
    }
    bigram_sums[row] = max(sum, 1u);
}

kernel void pg_lite_residual_row_sums_u16_kernel(
    device const ushort* residual_counts [[buffer(0)]],
    device uint* residual_sums [[buffer(1)]],
    constant uint& residual_buckets [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= residual_buckets) {
        return;
    }
    uint offset = row * PG_LITE_BYTE_VALUES;
    uint sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        sum += uint(residual_counts[offset + col]);
    }
    residual_sums[row] = max(sum, 1u);
}

kernel void pg_lite_residual_row_sums_u32_kernel(
    device const uint* residual_counts [[buffer(0)]],
    device uint* residual_sums [[buffer(1)]],
    constant uint& residual_buckets [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= residual_buckets) {
        return;
    }
    uint offset = row * PG_LITE_BYTE_VALUES;
    uint sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        sum += residual_counts[offset + col];
    }
    residual_sums[row] = max(sum, 1u);
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
    uint prev = pg_lite_prev_byte(bytes, gid);
    atomic_fetch_add_explicit(
        &bigram_counts[prev * PG_LITE_BYTE_VALUES + byte],
        1u,
        memory_order_relaxed
    );

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    atomic_fetch_add_explicit(
        &residual_counts[bucket * PG_LITE_BYTE_VALUES + byte],
        1u,
        memory_order_relaxed
    );
}

kernel void pg_lite_downcast_residual_u32_to_u16_kernel(
    device const uint* residual_u32 [[buffer(0)]],
    device ushort* residual_u16 [[buffer(1)]],
    constant uint& residual_entries [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < residual_entries) {
        residual_u16[gid] = ushort(min(residual_u32[gid], 65535u));
    }
}

kernel void pg_lite_ngram_loss_presummed_u16_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const uint* bigram_sums [[buffer(2)]],
    device const ushort* residual_counts [[buffer(3)]],
    device const uint* residual_sums [[buffer(4)]],
    constant PgLiteEvalConfig& cfg [[buffer(5)]],
    device float* token_losses [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cfg.token_count) {
        return;
    }
    uint byte = uint(bytes[gid]);
    uint prev = pg_lite_prev_byte(bytes, gid);
    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    float p = pg_lite_prob_u16_residual(
        byte, prev, bucket, bigram_counts, bigram_sums, residual_counts, residual_sums,
        cfg.residual_weight
    );
    token_losses[gid] = pg_lite_loss_from_prob(p);
}

kernel void pg_lite_ngram_loss_presummed_u32_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const uint* bigram_sums [[buffer(2)]],
    device const uint* residual_counts [[buffer(3)]],
    device const uint* residual_sums [[buffer(4)]],
    constant PgLiteEvalConfig& cfg [[buffer(5)]],
    device float* token_losses [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cfg.token_count) {
        return;
    }
    uint byte = uint(bytes[gid]);
    uint prev = pg_lite_prev_byte(bytes, gid);
    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    float p = pg_lite_prob_u32_residual(
        byte, prev, bucket, bigram_counts, bigram_sums, residual_counts, residual_sums,
        cfg.residual_weight
    );
    token_losses[gid] = pg_lite_loss_from_prob(p);
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
    uint prev = pg_lite_prev_byte(bytes, gid);
    uint bigram_offset = prev * PG_LITE_BYTE_VALUES;
    uint bigram_sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        bigram_sum += bigram_counts[bigram_offset + col];
    }

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    uint residual_offset = bucket * PG_LITE_BYTE_VALUES;
    uint residual_sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        residual_sum += uint(residual_counts[residual_offset + col]);
    }
    float p_bigram = float(bigram_counts[bigram_offset + byte]) / max(float(bigram_sum), 1.0f);
    float p_residual = float(residual_counts[residual_offset + byte]) /
        max(float(residual_sum), 1.0f);
    token_losses[gid] = pg_lite_loss_from_prob(
        pg_lite_mix_prob(p_bigram, p_residual, cfg.residual_weight)
    );
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
    uint prev = pg_lite_prev_byte(bytes, gid);
    uint bigram_offset = prev * PG_LITE_BYTE_VALUES;
    uint bigram_sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        bigram_sum += bigram_counts[bigram_offset + col];
    }

    uint bucket = pg_lite_context_bucket(bytes, gid, cfg.context, cfg.residual_buckets);
    uint residual_offset = bucket * PG_LITE_BYTE_VALUES;
    uint residual_sum = 0u;
    for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
        residual_sum += residual_counts[residual_offset + col];
    }
    float p_bigram = float(bigram_counts[bigram_offset + byte]) / max(float(bigram_sum), 1.0f);
    float p_residual = float(residual_counts[residual_offset + byte]) /
        max(float(residual_sum), 1.0f);
    token_losses[gid] = pg_lite_loss_from_prob(
        pg_lite_mix_prob(p_bigram, p_residual, cfg.residual_weight)
    );
}

kernel void pg_lite_ngram_loss_reduce_u16_presummed_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const uint* bigram_sums [[buffer(2)]],
    device const ushort* residual_counts [[buffer(3)]],
    device const uint* residual_sums [[buffer(4)]],
    constant PgLiteEvalConfig& cfg [[buffer(5)]],
    device float* partial_sums [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
    if (tid >= PG_LITE_THREADS) {
        return;
    }
    threadgroup float scratch[PG_LITE_THREADS];
    float sum = 0.0f;
    uint gid = tgid.x;
    uint groups = max(threadgroups_per_grid.x, 1u);
    for (uint idx = gid * PG_LITE_THREADS + tid; idx < cfg.token_count; idx += PG_LITE_THREADS * groups) {
        uint byte = uint(bytes[idx]);
        uint prev = pg_lite_prev_byte(bytes, idx);
        uint bucket = pg_lite_context_bucket(bytes, idx, cfg.context, cfg.residual_buckets);
        float p = pg_lite_prob_u16_residual(
            byte, prev, bucket, bigram_counts, bigram_sums, residual_counts, residual_sums,
            cfg.residual_weight
        );
        sum += pg_lite_loss_from_prob(p);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = PG_LITE_THREADS >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_sums[gid] = scratch[0];
    }
}

kernel void pg_lite_ngram_loss_reduce_u32_presummed_kernel(
    device const uchar* bytes [[buffer(0)]],
    device const uint* bigram_counts [[buffer(1)]],
    device const uint* bigram_sums [[buffer(2)]],
    device const uint* residual_counts [[buffer(3)]],
    device const uint* residual_sums [[buffer(4)]],
    constant PgLiteEvalConfig& cfg [[buffer(5)]],
    device float* partial_sums [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
    if (tid >= PG_LITE_THREADS) {
        return;
    }
    threadgroup float scratch[PG_LITE_THREADS];
    float sum = 0.0f;
    uint gid = tgid.x;
    uint groups = max(threadgroups_per_grid.x, 1u);
    for (uint idx = gid * PG_LITE_THREADS + tid; idx < cfg.token_count; idx += PG_LITE_THREADS * groups) {
        uint byte = uint(bytes[idx]);
        uint prev = pg_lite_prev_byte(bytes, idx);
        uint bucket = pg_lite_context_bucket(bytes, idx, cfg.context, cfg.residual_buckets);
        float p = pg_lite_prob_u32_residual(
            byte, prev, bucket, bigram_counts, bigram_sums, residual_counts, residual_sums,
            cfg.residual_weight
        );
        sum += pg_lite_loss_from_prob(p);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = PG_LITE_THREADS >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_sums[gid] = scratch[0];
    }
}

kernel void pg_lite_reduce_loss_kernel(
    device const float* token_losses [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
    if (tid >= PG_LITE_THREADS) {
        return;
    }
    threadgroup float scratch[PG_LITE_THREADS];
    float sum = 0.0f;
    uint gid = tgid.x;
    uint groups = max(threadgroups_per_grid.x, 1u);
    for (uint idx = gid * PG_LITE_THREADS + tid; idx < token_count; idx += PG_LITE_THREADS * groups) {
        sum += token_losses[idx];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = PG_LITE_THREADS >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_sums[gid] = scratch[0];
    }
}

kernel void pg_lite_score_first_eval_u32_kernel(
    device const uchar* bytes [[buffer(0)]],
    device uint* bigram_counts [[buffer(1)]],
    device uint* residual_counts [[buffer(2)]],
    constant PgLiteEvalConfig& cfg [[buffer(3)]],
    device float* loss_out [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u) {
        return;
    }
    float loss = 0.0f;
    for (uint idx = 0u; idx < cfg.token_count; ++idx) {
        uint byte = uint(bytes[idx]);
        uint prev = pg_lite_prev_byte(bytes, idx);
        uint bucket = pg_lite_context_bucket(bytes, idx, cfg.context, cfg.residual_buckets);
        uint bigram_offset = prev * PG_LITE_BYTE_VALUES;
        uint residual_offset = bucket * PG_LITE_BYTE_VALUES;
        uint bigram_sum = 0u;
        uint residual_sum = 0u;
        for (uint col = 0u; col < PG_LITE_BYTE_VALUES; ++col) {
            bigram_sum += bigram_counts[bigram_offset + col];
            residual_sum += residual_counts[residual_offset + col];
        }
        float p_bigram = float(bigram_counts[bigram_offset + byte]) /
            max(float(bigram_sum), 1.0f);
        float p_residual = float(residual_counts[residual_offset + byte]) /
            max(float(residual_sum), 1.0f);
        loss += pg_lite_loss_from_prob(
            pg_lite_mix_prob(p_bigram, p_residual, cfg.residual_weight)
        );
        bigram_counts[bigram_offset + byte] = bigram_counts[bigram_offset + byte] + 1u;
        residual_counts[residual_offset + byte] = residual_counts[residual_offset + byte] + 1u;
    }
    loss_out[0] = loss;
}
