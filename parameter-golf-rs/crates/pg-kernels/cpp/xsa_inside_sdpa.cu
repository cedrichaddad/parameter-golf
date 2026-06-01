#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr float kXsaNormEps = 1.0e-8f;

__device__ __forceinline__ size_t q_offset(
    int batch,
    int seq,
    int head,
    int seq_len,
    int num_heads,
    int head_dim
) {
    return (static_cast<size_t>(batch) * seq_len * num_heads
            + static_cast<size_t>(seq) * num_heads
            + head)
           * head_dim;
}

__device__ __forceinline__ size_t kv_offset(
    int batch,
    int seq,
    int kv_head,
    int seq_len,
    int num_kv_heads,
    int head_dim
) {
    return (static_cast<size_t>(batch) * seq_len * num_kv_heads
            + static_cast<size_t>(seq) * num_kv_heads
            + kv_head)
           * head_dim;
}

__global__ void xsa_inside_sdpa_forward_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale
) {
    const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = batch * seq_len * num_heads;
    if (query_index >= total_queries) {
        return;
    }

    const int head = query_index % num_heads;
    const int token_index = query_index / num_heads;
    const int seq = token_index % seq_len;
    const int batch_idx = token_index / seq_len;
    const int group = num_heads / num_kv_heads;
    const int kv_head = head / group;

    const size_t q_base = q_offset(batch_idx, seq, head, seq_len, num_heads, head_dim);
    const size_t self_v_base = kv_offset(batch_idx, seq, kv_head, seq_len, num_kv_heads, head_dim);
    const size_t out_base = q_base;

    float max_score = -INFINITY;
    for (int src = 0; src <= seq; ++src) {
        const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_base + dim] * k[k_base + dim];
        }
        const float score = dot * softmax_scale;
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    for (int src = 0; src <= seq; ++src) {
        const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_base + dim] * k[k_base + dim];
        }
        denom += expf(dot * softmax_scale - max_score);
    }

    float v_norm_sq = kXsaNormEps;
    float y_dot_v = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        const float v_self = v[self_v_base + dim];
        v_norm_sq += v_self * v_self;

        float y = 0.0f;
        for (int src = 0; src <= seq; ++src) {
            const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
            const size_t v_base = k_base;
            float dot = 0.0f;
            for (int inner = 0; inner < head_dim; ++inner) {
                dot += q[q_base + inner] * k[k_base + inner];
            }
            const float prob = expf(dot * softmax_scale - max_score) / denom;
            y += prob * v[v_base + dim];
        }
        y_dot_v += y * v_self;
    }

    const float coeff = y_dot_v / v_norm_sq;
    for (int dim = 0; dim < head_dim; ++dim) {
        float y = 0.0f;
        for (int src = 0; src <= seq; ++src) {
            const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
            const size_t v_base = k_base;
            float dot = 0.0f;
            for (int inner = 0; inner < head_dim; ++inner) {
                dot += q[q_base + inner] * k[k_base + inner];
            }
            const float prob = expf(dot * softmax_scale - max_score) / denom;
            y += prob * v[v_base + dim];
        }
        out[out_base + dim] = y - coeff * v[self_v_base + dim];
    }
}

__global__ void xsa_inside_sdpa_backward_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale
) {
    const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = batch * seq_len * num_heads;
    if (query_index >= total_queries) {
        return;
    }

    const int head = query_index % num_heads;
    const int token_index = query_index / num_heads;
    const int seq = token_index % seq_len;
    const int batch_idx = token_index / seq_len;
    const int group = num_heads / num_kv_heads;
    const int kv_head = head / group;

    const size_t q_base = q_offset(batch_idx, seq, head, seq_len, num_heads, head_dim);
    const size_t go_base = q_base;
    const size_t self_v_base = kv_offset(batch_idx, seq, kv_head, seq_len, num_kv_heads, head_dim);

    float max_score = -INFINITY;
    for (int src = 0; src <= seq; ++src) {
        const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_base + dim] * k[k_base + dim];
        }
        max_score = fmaxf(max_score, dot * softmax_scale);
    }

    float denom = 0.0f;
    for (int src = 0; src <= seq; ++src) {
        const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += q[q_base + dim] * k[k_base + dim];
        }
        denom += expf(dot * softmax_scale - max_score);
    }

    float v_norm_sq = kXsaNormEps;
    float go_dot_v = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        const float v_self = v[self_v_base + dim];
        v_norm_sq += v_self * v_self;
        go_dot_v += grad_out[go_base + dim] * v_self;
    }
    const float go_coeff = go_dot_v / v_norm_sq;

    float y_dot_v = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        float y = 0.0f;
        for (int src = 0; src <= seq; ++src) {
            const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
            const size_t v_base = k_base;
            float dot = 0.0f;
            for (int inner = 0; inner < head_dim; ++inner) {
                dot += q[q_base + inner] * k[k_base + inner];
            }
            const float prob = expf(dot * softmax_scale - max_score) / denom;
            y += prob * v[v_base + dim];
        }
        y_dot_v += y * v[self_v_base + dim];
    }
    const float coeff = y_dot_v / v_norm_sq;

    for (int dim = 0; dim < head_dim; ++dim) {
        float y = 0.0f;
        for (int src = 0; src <= seq; ++src) {
            const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
            const size_t v_base = k_base;
            float dot = 0.0f;
            for (int inner = 0; inner < head_dim; ++inner) {
                dot += q[q_base + inner] * k[k_base + inner];
            }
            const float prob = expf(dot * softmax_scale - max_score) / denom;
            y += prob * v[v_base + dim];
        }
        const float direct_grad_v =
            -coeff * grad_out[go_base + dim]
            - go_coeff * y
            + (2.0f * y_dot_v * go_dot_v / (v_norm_sq * v_norm_sq)) * v[self_v_base + dim];
        atomicAdd(&grad_v[self_v_base + dim], direct_grad_v);
    }

    float grad_score_sum = 0.0f;
    for (int src = 0; src <= seq; ++src) {
        const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
        const size_t v_base = k_base;
        float dot_qk = 0.0f;
        float dot_grad_y_v = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot_qk += q[q_base + dim] * k[k_base + dim];
            const float grad_y = grad_out[go_base + dim] - go_coeff * v[self_v_base + dim];
            dot_grad_y_v += grad_y * v[v_base + dim];
        }
        const float prob = expf(dot_qk * softmax_scale - max_score) / denom;
        grad_score_sum += prob * dot_grad_y_v;
    }

    for (int dim = 0; dim < head_dim; ++dim) {
        float grad_q_acc = 0.0f;
        const float grad_y = grad_out[go_base + dim] - go_coeff * v[self_v_base + dim];
        for (int src = 0; src <= seq; ++src) {
            const size_t k_base = kv_offset(batch_idx, src, kv_head, seq_len, num_kv_heads, head_dim);
            const size_t v_base = k_base;
            float dot_qk = 0.0f;
            float dot_grad_y_v = 0.0f;
            for (int inner = 0; inner < head_dim; ++inner) {
                dot_qk += q[q_base + inner] * k[k_base + inner];
                const float grad_y_inner =
                    grad_out[go_base + inner] - go_coeff * v[self_v_base + inner];
                dot_grad_y_v += grad_y_inner * v[v_base + inner];
            }
            const float prob = expf(dot_qk * softmax_scale - max_score) / denom;
            const float grad_pre_softmax = prob * (dot_grad_y_v - grad_score_sum);

            grad_q_acc += grad_pre_softmax * softmax_scale * k[k_base + dim];
            atomicAdd(&grad_k[k_base + dim], grad_pre_softmax * softmax_scale * q[q_base + dim]);
            atomicAdd(&grad_v[v_base + dim], prob * grad_y);
        }
        grad_q[q_base + dim] = grad_q_acc;
    }
}

int validate_args(
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t first_out_ptr,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (batch <= 0 || seq_len <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return 1;
    }
    if (num_heads % num_kv_heads != 0) {
        return 2;
    }
    if (q_ptr == 0 || k_ptr == 0 || v_ptr == 0 || first_out_ptr == 0) {
        return 3;
    }
    return 0;
}

}  // namespace

extern "C" {

int run_naive_xsa_inside_sdpa_f32_forward(
    void* stream,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t out_ptr,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale
) {
    const int validation =
        validate_args(q_ptr, k_ptr, v_ptr, out_ptr, batch, seq_len, num_heads, num_kv_heads, head_dim);
    if (validation != 0) {
        return validation;
    }

    const float* q = reinterpret_cast<const float*>(q_ptr);
    const float* k = reinterpret_cast<const float*>(k_ptr);
    const float* v = reinterpret_cast<const float*>(v_ptr);
    float* out = reinterpret_cast<float*>(out_ptr);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    const int total_queries = batch * seq_len * num_heads;
    const int threads = 128;
    const int blocks = (total_queries + threads - 1) / threads;
    xsa_inside_sdpa_forward_kernel<<<blocks, threads, 0, cuda_stream>>>(
        q,
        k,
        v,
        out,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        softmax_scale
    );
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

int run_naive_xsa_inside_sdpa_f32_backward(
    void* stream,
    uint64_t q_ptr,
    uint64_t k_ptr,
    uint64_t v_ptr,
    uint64_t grad_out_ptr,
    uint64_t grad_q_ptr,
    uint64_t grad_k_ptr,
    uint64_t grad_v_ptr,
    int batch,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale
) {
    int validation = validate_args(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_out_ptr,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim
    );
    if (validation != 0) {
        return validation;
    }
    if (grad_q_ptr == 0 || grad_k_ptr == 0 || grad_v_ptr == 0) {
        return 3;
    }

    const float* q = reinterpret_cast<const float*>(q_ptr);
    const float* k = reinterpret_cast<const float*>(k_ptr);
    const float* v = reinterpret_cast<const float*>(v_ptr);
    const float* grad_out = reinterpret_cast<const float*>(grad_out_ptr);
    float* grad_q = reinterpret_cast<float*>(grad_q_ptr);
    float* grad_k = reinterpret_cast<float*>(grad_k_ptr);
    float* grad_v = reinterpret_cast<float*>(grad_v_ptr);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    const size_t q_bytes = static_cast<size_t>(batch) * seq_len * num_heads * head_dim * sizeof(float);
    const size_t kv_bytes = static_cast<size_t>(batch) * seq_len * num_kv_heads * head_dim * sizeof(float);
    cudaError_t err = cudaMemsetAsync(grad_q, 0, q_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMemsetAsync(grad_k, 0, kv_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMemsetAsync(grad_v, 0, kv_bytes, cuda_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const int total_queries = batch * seq_len * num_heads;
    const int threads = 128;
    const int blocks = (total_queries + threads - 1) / threads;
    xsa_inside_sdpa_backward_kernel<<<blocks, threads, 0, cuda_stream>>>(
        q,
        k,
        v,
        grad_out,
        grad_q,
        grad_k,
        grad_v,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        softmax_scale
    );
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

}
