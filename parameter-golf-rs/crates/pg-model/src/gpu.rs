/// GPU model runner — orchestrates forward/backward on CUDA.
///
/// Maps the CPU-verified model logic to GPU kernels:
///   - cuBLASLt for all GEMM (QKV projections, MLP, output, Newton-Schulz)
///   - cuDNN Flash Attention 3 for fused attention forward+backward
///   - CubeCL fused element-wise kernels (RMSNorm, RoPE, activations, residuals)
///   - NCCL for multi-GPU (reduce-scatter/all-gather/all-reduce)
///
/// Memory layout: all parameter banks are contiguous BF16 on device.
/// Activations are allocated from BufferPool (zero runtime malloc).
///
/// This module requires the `cuda` feature.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use pg_core::tensor::GpuTensor;

use crate::config::ModelConfig;

/// GPU weight storage — all parameters resident on a single device.
#[cfg(feature = "cuda")]
pub struct GpuWeights {
    // Parameter banks — contiguous BF16
    pub qo_bank: GpuTensor,       // [2*n, d, d]
    pub kv_bank: GpuTensor,       // [2*n, kv, d]
    pub mlp_up_bank: GpuTensor,   // [n, mlp, d]
    pub mlp_down_bank: GpuTensor, // [n, d, mlp]

    // Embeddings — BF16
    pub tok_emb: GpuTensor,       // [vocab, d]

    // Scalar params — F32 (small, precision-sensitive)
    pub attn_scales: Vec<GpuTensor>,   // per-layer [d]
    pub mlp_scales: Vec<GpuTensor>,    // per-layer [d]
    pub resid_mix: Vec<GpuTensor>,     // per-layer [2, d]
    pub q_gains: Vec<GpuTensor>,       // per-layer [h]

    // Misc
    pub bigram_embed: GpuTensor,
    pub bigram_proj: GpuTensor,
    pub smear_gate: GpuTensor,
    pub skip_weights: GpuTensor,

    // RoPE tables — precomputed F32
    pub rope_cos: GpuTensor,
    pub rope_sin: GpuTensor,
}

/// GPU activation buffers — pre-allocated from pool.
#[cfg(feature = "cuda")]
pub struct GpuActivations {
    pub x: GpuTensor,             // [B*T, d] main hidden state
    pub x0: GpuTensor,            // [B*T, d] residual anchor
    pub attn_norm: GpuTensor,     // [B*T, d]
    pub mlp_norm: GpuTensor,      // [B*T, d]
    pub q: GpuTensor,             // [B*T, h, hd]
    pub k: GpuTensor,             // [B*T, hkv, hd]
    pub v: GpuTensor,             // [B*T, hkv, hd]
    pub attn_out: GpuTensor,      // [B*T, h, hd]
    pub proj_out: GpuTensor,      // [B*T, d]
    pub mlp_up: GpuTensor,        // [B*T, mlp]
    pub mlp_act: GpuTensor,       // [B*T, mlp]
    pub mlp_out: GpuTensor,       // [B*T, d]
    pub logits: GpuTensor,        // [B*T, vocab]

    // Checkpointed layer states for backward
    pub layer_checkpoints: Vec<GpuTensor>, // layers 3-8 recomputed
}

/// Training step phases on GPU.
///
/// Single step timeline on 8×H100:
///
/// ```text
/// |-- Data load (async memcpy) --|
/// |-- Forward pass (compute stream) --|
/// |-- Backward pass (compute stream) --|
/// |-- Phase 1: Async RS all banks (NCCL stream) --|
/// |-- Phase 2: AllReduce scalars + AdamW step (overlapped) --|
/// |-- Phase 3: Wait RS → NS5 → Async AG (per bank, pipelined) --|
/// |-- EMA update (compute stream) --|
/// ```
///
/// Key optimizations:
/// - Activation checkpointing layers 3-8 (saves ~40% memory)
/// - Fused attention (cuDNN FlashAttn3): QKV→softmax→output in one kernel
/// - Fused element-wise: RMSNorm+residual, LeakyReLU²+scale, RoPE+QKnorm
/// - 3-stream overlap: compute, NCCL, memcpy never block each other

/// Bank shapes for the Muon optimizer.
pub fn bank_shapes(config: &ModelConfig) -> Vec<[usize; 3]> {
    let n = config.num_layers;
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    vec![
        [2 * n, d, d],    // qo_bank
        [2 * n, kv, d],   // kv_bank
        [n, mlp, d],      // mlp_up_bank
        [n, d, mlp],      // mlp_down_bank
    ]
}

/// Activation checkpointing config.
/// Layers 3-8 (0-indexed) recompute forward during backward.
/// Layers 0-2 and 9-10 save full activations.
pub fn checkpoint_layers(config: &ModelConfig) -> Vec<bool> {
    (0..config.num_layers)
        .map(|i| i >= 3 && i <= 8)
        .collect()
}

/// Estimate peak GPU memory for training (bytes).
pub fn estimate_memory(config: &ModelConfig, batch_tokens: usize) -> usize {
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    let n = config.num_layers;
    let vocab = config.vocab_size;

    // Parameters (BF16 = 2 bytes)
    let param_bytes = (2 * n * d * d     // qo_bank
        + 2 * n * kv * d   // kv_bank
        + n * mlp * d       // mlp_up
        + n * d * mlp       // mlp_down
        + vocab * d         // tok_emb
    ) * 2;

    // Gradients (same size as params)
    let grad_bytes = param_bytes;

    // Optimizer state: Muon momentum (same as banks) + AdamW m/v (2× scalars)
    let muon_state = param_bytes; // momentum buffer same size as banks
    let adamw_state = vocab * d * 2 * 2 * 4; // m + v for embeddings, F32

    // NS5 workspace: for each bank [B,M,N], need a_buf [B,rows,rows], aa, b_buf, new_x [B,rows,cols]
    // rows = min(M,N), cols = max(M,N)
    let ns5_workspace: usize = bank_shapes(config).iter().map(|s| {
        let (b, m, n_dim) = (s[0], s[1], s[2]);
        let (rows, cols) = if m > n_dim { (n_dim, m) } else { (m, n_dim) };
        (3 * b * rows * rows + b * rows * cols) * 2 // BF16
    }).sum();

    // Activations (BF16)
    let bt = batch_tokens;
    let act_per_layer = bt * d * 2 // x + attn_norm
        + bt * d * 2       // mlp_norm + proj_out
        + bt * config.num_heads as usize * config.head_dim * 2 // q, attn_out
        + bt * kv * 2       // k, v
        + bt * mlp * 2      // mlp_up, mlp_act
        + bt * d;           // mlp_out
    let act_bytes = act_per_layer * 2 * 2; // BF16, keep ~2 layers live

    // Logits
    let logit_bytes = bt * vocab * 4; // F32 for numerical stability

    param_bytes + grad_bytes + muon_state + adamw_state + ns5_workspace + act_bytes + logit_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimate() {
        let config = ModelConfig::sota();
        let mem = estimate_memory(&config, config.train_seq_len);
        let gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("Estimated peak GPU memory: {:.2} GB (per device)", gb);
        // Should fit in H100 80GB
        assert!(gb < 80.0, "exceeds H100 memory: {:.2} GB", gb);
    }

    #[test]
    fn test_checkpoint_layers() {
        let config = ModelConfig::sota();
        let ckpt = checkpoint_layers(&config);
        assert_eq!(ckpt.len(), 11);
        assert!(!ckpt[0]); // layer 0: not checkpointed
        assert!(!ckpt[2]); // layer 2: not checkpointed
        assert!(ckpt[3]);  // layer 3: checkpointed
        assert!(ckpt[8]);  // layer 8: checkpointed
        assert!(!ckpt[9]); // layer 9: not checkpointed
    }

    #[test]
    fn test_bank_shapes() {
        let config = ModelConfig::sota();
        let shapes = bank_shapes(&config);
        assert_eq!(shapes.len(), 4);
        assert_eq!(shapes[0], [22, 512, 512]); // qo_bank
        assert_eq!(shapes[1], [22, 256, 512]); // kv_bank
        assert_eq!(shapes[2], [11, 1536, 512]); // mlp_up
        assert_eq!(shapes[3], [11, 512, 1536]); // mlp_down
    }
}
