/// GPU model runner — orchestrates forward/backward on CUDA.
///
/// Maps the CPU-verified model logic to GPU kernels:
///   - cuBLASLt for all GEMM (QKV projections, MLP, output, Newton-Schulz)
///   - F32 CUDA SDPA parity kernels today; production BF16 fused SDPA is gated
///   - CUDA element-wise kernels (RMSNorm, RoPE, activations, residuals)
///   - NCCL all-reduce multi-GPU today; sharded Parallel Muon is gated
///
/// Memory layout: all parameter banks are contiguous F32 on device today.
/// Activations are allocated from BufferPool (zero runtime malloc).
///
/// This module requires the `cuda` feature.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgResult};

use crate::config::ModelConfig;
#[cfg(feature = "cuda")]
use crate::{ExecutionPlan, GptModel};

/// GPU weight storage — all parameters resident on a single device.
#[cfg(feature = "cuda")]
pub struct GpuWeights {
    // Parameter banks — contiguous BF16
    pub qo_bank: GpuTensor,       // [2*n, d, d]
    pub kv_bank: GpuTensor,       // [2*n, kv, d]
    pub mlp_up_bank: GpuTensor,   // [n, mlp, d]
    pub mlp_down_bank: GpuTensor, // [n, d, mlp]

    // Embeddings — BF16
    pub tok_emb: GpuTensor, // [vocab, d]

    // Scalar params — F32 (small, precision-sensitive)
    pub attn_scales: Vec<GpuTensor>,       // per-layer [d]
    pub mlp_scales: Vec<GpuTensor>,        // per-layer [d]
    pub resid_mix: Vec<GpuTensor>,         // per-layer [2, d]
    pub q_gains: Vec<GpuTensor>,           // per-layer [h]
    pub attn_gate_weights: Vec<GpuTensor>, // per-layer [h, gate_width]
    pub attn_gate_biases: Vec<GpuTensor>,  // per-layer [h]

    // Misc
    pub bigram_embed: GpuTensor,
    pub bigram_proj: GpuTensor,
    pub bigram_scale: f32,
    pub smear_gate: GpuTensor,
    pub skip_weights: GpuTensor,

    // Value Embedding (shared)
    pub ve_embed: GpuTensor,
    pub ve_proj: GpuTensor,
    pub ve_scale: f32,
    pub ve_layer_scales: GpuTensor,
    pub ve_layer_scales_host: Vec<f32>,

    // RoPE tables — precomputed F32
    pub rope_cos: GpuTensor,
    pub rope_sin: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuWeights {
    pub fn from_cpu(cpu: &crate::model::GptModel, stream: Arc<CudaStream>) -> PgResult<Self> {
        let c = &cpu.config;
        let l = c.num_layers;
        let d = c.model_dim;
        let md = c.mlp_dim;
        let kv = c.kv_dim();

        fn to_gpu(s: &Arc<CudaStream>, data: &[f32], shape: &[usize]) -> PgResult<GpuTensor> {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            GpuTensor::from_host_data_gpu(s.clone(), bytes, shape, DType::F32)
        }

        let mut attn_scales = Vec::new();
        let mut mlp_scales = Vec::new();
        let mut resid_mix = Vec::new();
        let mut q_gains = Vec::new();
        let mut attn_gate_weights = Vec::new();
        let mut attn_gate_biases = Vec::new();

        for i in 0..l {
            let b = &cpu.blocks[i];
            attn_scales.push(to_gpu(&stream, &b.attn_scale, &[d])?);
            mlp_scales.push(to_gpu(&stream, &b.mlp_scale, &[d])?);
            resid_mix.push(to_gpu(&stream, &b.resid_mix, &[2, d])?);
            q_gains.push(to_gpu(&stream, &b.q_gain, &[c.num_heads])?);
            let gate_width = c.attn_out_gate_width.max(1);
            attn_gate_weights.push(to_gpu(
                &stream,
                &b.attn_gate_weight,
                &[c.num_heads, gate_width],
            )?);
            attn_gate_biases.push(to_gpu(&stream, &b.attn_gate_bias, &[c.num_heads])?);
        }

        Ok(Self {
            qo_bank: to_gpu(&stream, &cpu.qo_bank, &[2 * l, d, d])?,
            kv_bank: to_gpu(&stream, &cpu.kv_bank, &[2 * l, kv, d])?,
            mlp_up_bank: to_gpu(&stream, &cpu.mlp_up_bank, &[l, md, d])?,
            mlp_down_bank: to_gpu(&stream, &cpu.mlp_down_bank, &[l, d, md])?,
            tok_emb: to_gpu(&stream, &cpu.tok_emb, &[c.vocab_size, d])?,
            attn_scales,
            mlp_scales,
            resid_mix,
            q_gains,
            attn_gate_weights,
            attn_gate_biases,
            bigram_embed: to_gpu(
                &stream,
                &cpu.bigram_embed,
                &[c.bigram_vocab_size, c.bigram_dim],
            )?,
            bigram_proj: to_gpu(&stream, &cpu.bigram_proj, &[d, c.bigram_dim])?,
            bigram_scale: cpu.bigram_scale,
            smear_gate: to_gpu(&stream, &cpu.smear_gate, &[d])?,
            skip_weights: to_gpu(&stream, &cpu.skip_weights, &[c.num_skip_weights(), d])?,
            ve_embed: to_gpu(&stream, &cpu.ve_embed, &[c.vocab_size, c.ve_dim])?,
            ve_proj: to_gpu(&stream, &cpu.ve_proj, &[kv, c.ve_dim])?,
            ve_scale: cpu.ve_scale,
            ve_layer_scales: to_gpu(&stream, &cpu.ve_layer_scales, &[cpu.ve_layer_scales.len()])?,
            ve_layer_scales_host: cpu.ve_layer_scales.clone(),
            rope_cos: to_gpu(&stream, &cpu.rope_cos, &[c.train_seq_len, c.rope_dims / 2])?,
            rope_sin: to_gpu(&stream, &cpu.rope_sin, &[c.train_seq_len, c.rope_dims / 2])?,
        })
    }

    pub fn sync_from_cpu(&mut self, cpu: &crate::model::GptModel) -> PgResult<()> {
        fn sync_f32(tensor: &mut GpuTensor, data: &[f32]) -> PgResult<()> {
            tensor.copy_from_host_bytes(bytemuck::cast_slice(data))
        }

        let c = &cpu.config;
        let l = c.num_layers;
        if self.attn_scales.len() != l
            || self.mlp_scales.len() != l
            || self.resid_mix.len() != l
            || self.q_gains.len() != l
            || self.attn_gate_weights.len() != l
            || self.attn_gate_biases.len() != l
        {
            return Err(pg_core::PgError::InvalidOp(
                "GPU weight layout no longer matches CPU model layer count".into(),
            ));
        }

        sync_f32(&mut self.qo_bank, &cpu.qo_bank)?;
        sync_f32(&mut self.kv_bank, &cpu.kv_bank)?;
        sync_f32(&mut self.mlp_up_bank, &cpu.mlp_up_bank)?;
        sync_f32(&mut self.mlp_down_bank, &cpu.mlp_down_bank)?;
        sync_f32(&mut self.tok_emb, &cpu.tok_emb)?;
        for i in 0..l {
            let b = &cpu.blocks[i];
            sync_f32(&mut self.attn_scales[i], &b.attn_scale)?;
            sync_f32(&mut self.mlp_scales[i], &b.mlp_scale)?;
            sync_f32(&mut self.resid_mix[i], &b.resid_mix)?;
            sync_f32(&mut self.q_gains[i], &b.q_gain)?;
            sync_f32(&mut self.attn_gate_weights[i], &b.attn_gate_weight)?;
            sync_f32(&mut self.attn_gate_biases[i], &b.attn_gate_bias)?;
        }
        sync_f32(&mut self.bigram_embed, &cpu.bigram_embed)?;
        sync_f32(&mut self.bigram_proj, &cpu.bigram_proj)?;
        self.bigram_scale = cpu.bigram_scale;
        sync_f32(&mut self.smear_gate, &cpu.smear_gate)?;
        sync_f32(&mut self.skip_weights, &cpu.skip_weights)?;
        sync_f32(&mut self.ve_embed, &cpu.ve_embed)?;
        sync_f32(&mut self.ve_proj, &cpu.ve_proj)?;
        self.ve_scale = cpu.ve_scale;
        sync_f32(&mut self.ve_layer_scales, &cpu.ve_layer_scales)?;
        self.ve_layer_scales_host.clone_from(&cpu.ve_layer_scales);
        Ok(())
    }

    pub fn sync_to_cpu(&self, cpu: &mut crate::model::GptModel) -> PgResult<()> {
        fn download_f32(tensor: &GpuTensor) -> PgResult<Vec<f32>> {
            let bytes = tensor.to_host_bytes()?;
            Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
        }

        let c = &cpu.config;
        let l = c.num_layers;
        if self.attn_scales.len() != l
            || self.mlp_scales.len() != l
            || self.resid_mix.len() != l
            || self.q_gains.len() != l
            || self.attn_gate_weights.len() != l
            || self.attn_gate_biases.len() != l
        {
            return Err(pg_core::PgError::InvalidOp(
                "GPU weight layout no longer matches CPU model layer count".into(),
            ));
        }

        cpu.qo_bank = download_f32(&self.qo_bank)?;
        cpu.kv_bank = download_f32(&self.kv_bank)?;
        cpu.mlp_up_bank = download_f32(&self.mlp_up_bank)?;
        cpu.mlp_down_bank = download_f32(&self.mlp_down_bank)?;
        cpu.tok_emb = download_f32(&self.tok_emb)?;
        for i in 0..l {
            cpu.blocks[i].attn_scale = download_f32(&self.attn_scales[i])?;
            cpu.blocks[i].mlp_scale = download_f32(&self.mlp_scales[i])?;
            cpu.blocks[i].resid_mix = download_f32(&self.resid_mix[i])?;
            cpu.blocks[i].q_gain = download_f32(&self.q_gains[i])?;
            cpu.blocks[i].attn_gate_weight = download_f32(&self.attn_gate_weights[i])?;
            cpu.blocks[i].attn_gate_bias = download_f32(&self.attn_gate_biases[i])?;
        }
        cpu.bigram_embed = download_f32(&self.bigram_embed)?;
        cpu.bigram_proj = download_f32(&self.bigram_proj)?;
        cpu.bigram_scale = self.bigram_scale;
        cpu.smear_gate = download_f32(&self.smear_gate)?;
        cpu.skip_weights = download_f32(&self.skip_weights)?;
        cpu.ve_embed = download_f32(&self.ve_embed)?;
        cpu.ve_proj = download_f32(&self.ve_proj)?;
        cpu.ve_scale = self.ve_scale;
        cpu.ve_layer_scales = self.ve_layer_scales_host.clone();
        Ok(())
    }
}

/// GPU activation buffers — pre-allocated from pool.
#[cfg(feature = "cuda")]
pub struct GpuActivations {
    pub x: GpuTensor,
    pub x_in: GpuTensor,
    pub x0: GpuTensor,
    pub attn_norm: GpuTensor,
    pub mlp_norm: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub ve_out: GpuTensor,
    pub ve_embed_out: GpuTensor,
    pub attn_out: GpuTensor,
    pub xsa_out: GpuTensor,
    pub attn_gated: GpuTensor,
    pub attn_gate_values: GpuTensor,
    pub attn_gate_grad_input: GpuTensor,
    pub proj_out: GpuTensor,
    pub mlp_up: GpuTensor,
    pub mlp_act: GpuTensor,
    pub mlp_out: GpuTensor,
    pub bigram_out: GpuTensor,
    pub bigram_proj_out: GpuTensor,
    pub lora_tmp: GpuTensor,
    pub lora_delta: GpuTensor,
    pub lora_grad_tmp: GpuTensor,
    pub logits: GpuTensor,

    pub encoder_skips: Vec<GpuTensor>,
}

#[cfg(feature = "cuda")]
impl GpuActivations {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let vocab = config.vocab_size;
        let bigram_dim = config.bigram_dim.max(1);
        let ve_dim = config.ve_dim.max(1);

        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        let mut encoder_skips = Vec::with_capacity(config.num_encoder_layers());
        for _ in 0..config.num_encoder_layers() {
            encoder_skips.push(zeros(&[tokens, d])?);
        }

        Ok(Self {
            x: zeros(&[tokens, d])?,
            x_in: zeros(&[tokens, d])?,
            x0: zeros(&[tokens, d])?,
            attn_norm: zeros(&[tokens, d])?,
            mlp_norm: zeros(&[tokens, d])?,
            q: zeros(&[tokens, config.num_heads, config.head_dim])?,
            k: zeros(&[tokens, config.num_kv_heads, config.head_dim])?,
            v: zeros(&[tokens, config.num_kv_heads, config.head_dim])?,
            ve_out: zeros(&[tokens, kv])?,
            ve_embed_out: zeros(&[tokens, ve_dim])?,
            attn_out: zeros(&[tokens, config.num_heads, config.head_dim])?,
            xsa_out: zeros(&[tokens, config.num_heads, config.head_dim])?,
            attn_gated: zeros(&[tokens, config.num_heads, config.head_dim])?,
            attn_gate_values: zeros(&[tokens, config.num_heads])?,
            attn_gate_grad_input: zeros(&[tokens, d])?,
            proj_out: zeros(&[tokens, d])?,
            mlp_up: zeros(&[tokens, mlp])?,
            mlp_act: zeros(&[tokens, mlp])?,
            mlp_out: zeros(&[tokens, d])?,
            bigram_out: zeros(&[tokens, bigram_dim])?,
            bigram_proj_out: zeros(&[tokens, d])?,
            lora_tmp: zeros(&[tokens, d])?,
            lora_delta: zeros(&[tokens, d])?,
            lora_grad_tmp: zeros(&[tokens, d])?,
            logits: zeros(&[tokens, vocab])?,
            encoder_skips,
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        Self::new(&config, tokens, stream)
    }
}

/// GPU gradient buffers matching the CPU `GradBuffers` parameter layout.
///
/// This is intentionally allocation-only today. The record path must not use
/// it until `GpuModel::backward` writes these tensors with real CUDA kernels.
#[cfg(feature = "cuda")]
pub struct GpuGradBuffers {
    pub tok_emb: GpuTensor,
    pub bigram_embed: GpuTensor,
    pub bigram_proj: GpuTensor,
    pub bigram_scale: GpuTensor,
    pub smear_gate: GpuTensor,
    pub skip_weights: GpuTensor,
    pub qo_bank: GpuTensor,
    pub kv_bank: GpuTensor,
    pub mlp_up_bank: GpuTensor,
    pub mlp_down_bank: GpuTensor,
    pub block_attn_scale: Vec<GpuTensor>,
    pub block_mlp_scale: Vec<GpuTensor>,
    pub block_resid_mix: Vec<GpuTensor>,
    pub block_q_gain: Vec<GpuTensor>,
    pub block_attn_gate_weight: Vec<GpuTensor>,
    pub block_attn_gate_bias: Vec<GpuTensor>,
    pub ve_embed: GpuTensor,
    pub ve_proj: GpuTensor,
    pub ve_scale: GpuTensor,
    pub ve_layer_scales: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuGradBuffers {
    pub fn new(config: &ModelConfig, stream: Arc<CudaStream>) -> PgResult<Self> {
        let n = config.num_layers;
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        Ok(Self {
            tok_emb: zeros(&[config.vocab_size, d])?,
            bigram_embed: zeros(&[config.bigram_vocab_size, config.bigram_dim.max(1)])?,
            bigram_proj: zeros(&[d, config.bigram_dim.max(1)])?,
            bigram_scale: zeros(&[1])?,
            smear_gate: zeros(&[d])?,
            skip_weights: zeros(&[config.num_skip_weights(), d])?,
            qo_bank: zeros(&[2 * n, d, d])?,
            kv_bank: zeros(&[2 * n, kv, d])?,
            mlp_up_bank: zeros(&[n, mlp, d])?,
            mlp_down_bank: zeros(&[n, d, mlp])?,
            block_attn_scale: (0..n).map(|_| zeros(&[d])).collect::<PgResult<_>>()?,
            block_mlp_scale: (0..n).map(|_| zeros(&[d])).collect::<PgResult<_>>()?,
            block_resid_mix: (0..n).map(|_| zeros(&[2, d])).collect::<PgResult<_>>()?,
            block_q_gain: (0..n)
                .map(|_| zeros(&[config.num_heads]))
                .collect::<PgResult<_>>()?,
            block_attn_gate_weight: (0..n)
                .map(|_| zeros(&[config.num_heads, config.attn_out_gate_width.max(1)]))
                .collect::<PgResult<_>>()?,
            block_attn_gate_bias: (0..n)
                .map(|_| zeros(&[config.num_heads]))
                .collect::<PgResult<_>>()?,
            ve_embed: zeros(&[config.vocab_size, config.ve_dim.max(1)])?,
            ve_proj: zeros(&[kv, config.ve_dim.max(1)])?,
            ve_scale: zeros(&[1])?,
            ve_layer_scales: zeros(&[config.ve_layers.len().max(1)])?,
        })
    }

    pub fn zero(&self, kernels: &pg_kernels::gpu_kernels::GpuKernels) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let zero = |tensor: &GpuTensor| {
            kernels.scale_inplace(
                CudaPtr(tensor.cu_ptr(kernels.stream())?),
                0.0,
                tensor.numel() as u32,
            )
        };

        zero(&self.tok_emb)?;
        zero(&self.bigram_embed)?;
        zero(&self.bigram_proj)?;
        zero(&self.bigram_scale)?;
        zero(&self.smear_gate)?;
        zero(&self.skip_weights)?;
        zero(&self.qo_bank)?;
        zero(&self.kv_bank)?;
        zero(&self.mlp_up_bank)?;
        zero(&self.mlp_down_bank)?;
        for tensor in &self.block_attn_scale {
            zero(tensor)?;
        }
        for tensor in &self.block_mlp_scale {
            zero(tensor)?;
        }
        for tensor in &self.block_resid_mix {
            zero(tensor)?;
        }
        for tensor in &self.block_q_gain {
            zero(tensor)?;
        }
        for tensor in &self.block_attn_gate_weight {
            zero(tensor)?;
        }
        for tensor in &self.block_attn_gate_bias {
            zero(tensor)?;
        }
        zero(&self.ve_embed)?;
        zero(&self.ve_proj)?;
        zero(&self.ve_scale)?;
        zero(&self.ve_layer_scales)?;
        Ok(())
    }
}

/// Saved GPU forward boundary states required for correctness-first backward.
///
/// This mirrors the CPU `ForwardCache` at the layer-boundary level while still
/// allowing block-internal activations to be recomputed later.
#[cfg(feature = "cuda")]
pub struct GpuForwardCache {
    pub layer_x: Vec<GpuTensor>,
    pub x0: GpuTensor,
    pub x_final: GpuTensor,
    pub skips: Vec<GpuTensor>,
    pub x_post_embed: GpuTensor,
    pub x_post_norm: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuForwardCache {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        let d = config.model_dim;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        Ok(Self {
            layer_x: (0..config.num_layers)
                .map(|_| zeros(&[tokens, d]))
                .collect::<PgResult<_>>()?,
            x0: zeros(&[tokens, d])?,
            x_final: zeros(&[tokens, d])?,
            skips: (0..config.num_encoder_layers())
                .map(|_| zeros(&[tokens, d]))
                .collect::<PgResult<_>>()?,
            x_post_embed: zeros(&[tokens, d])?,
            x_post_norm: zeros(&[tokens, d])?,
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        Self::new(&config, tokens, stream)
    }
}

/// Block-local recompute cache for GPU backward.
#[cfg(feature = "cuda")]
struct GpuBlockBackwardCache {
    q_pre_norm: GpuTensor,
    k_pre_norm: GpuTensor,
    q_post_rope: GpuTensor,
    x_after_attn: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuBlockBackwardCache {
    fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);
        Ok(Self {
            q_pre_norm: zeros(&[tokens, config.num_heads, config.head_dim])?,
            k_pre_norm: zeros(&[tokens, config.num_kv_heads, config.head_dim])?,
            q_post_rope: zeros(&[tokens, config.num_heads, config.head_dim])?,
            x_after_attn: zeros(&[tokens, config.model_dim])?,
        })
    }
}

/// Persistent backward recompute state reused across training steps.
#[cfg(feature = "cuda")]
pub struct GpuBackwardState {
    cache: GpuForwardCache,
    block_cache: GpuBlockBackwardCache,
}

#[cfg(feature = "cuda")]
impl GpuBackwardState {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        Ok(Self {
            cache: GpuForwardCache::new(config, tokens, stream.clone())?,
            block_cache: GpuBlockBackwardCache::new(config, tokens, stream)?,
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        Self::new(&config, tokens, stream)
    }
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
/// - Production record gaps are explicit in `ModelSpec` / `TrainSpec`.
/// - Current distributed path prioritizes correctness over overlap.

/// Bank shapes for the Muon optimizer.
pub fn bank_shapes(config: &ModelConfig) -> Vec<[usize; 3]> {
    let n = config.num_layers;
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    vec![
        [2 * n, d, d],  // qo_bank
        [2 * n, kv, d], // kv_bank
        [n, mlp, d],    // mlp_up_bank
        [n, d, mlp],    // mlp_down_bank
    ]
}

/// Activation checkpointing config.
/// Layers 3-8 (0-indexed) recompute forward during backward.
/// Layers 0-2 and 9-10 save full activations.
pub fn checkpoint_layers(config: &ModelConfig) -> Vec<bool> {
    (0..config.num_layers).map(|i| i >= 3 && i <= 8).collect()
}

/// Estimate peak GPU memory for training (bytes).
pub fn estimate_memory(config: &ModelConfig, batch_tokens: usize) -> usize {
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    let n = config.num_layers;
    let vocab = config.vocab_size;

    // Parameters (BF16 = 2 bytes)
    let param_bytes = (
        2 * n * d * d     // qo_bank
        + 2 * n * kv * d   // kv_bank
        + n * mlp * d       // mlp_up
        + n * d * mlp       // mlp_down
        + vocab * d
        // tok_emb
    ) * 2;

    // Gradients (same size as params)
    let grad_bytes = param_bytes;

    // Optimizer state: Muon momentum (same as banks) + AdamW m/v (2× scalars)
    let muon_state = param_bytes; // momentum buffer same size as banks
    let adamw_state = vocab * d * 2 * 2 * 4; // m + v for embeddings, F32

    // NS5 workspace: for each bank [B,M,N], need a_buf [B,rows,rows], aa, b_buf, new_x [B,rows,cols]
    // rows = min(M,N), cols = max(M,N)
    let ns5_workspace: usize = bank_shapes(config)
        .iter()
        .map(|s| {
            let (b, m, n_dim) = (s[0], s[1], s[2]);
            let (rows, cols) = if m > n_dim { (n_dim, m) } else { (m, n_dim) };
            (3 * b * rows * rows + b * rows * cols) * 2 // BF16
        })
        .sum();

    // Activations (BF16)
    let bt = batch_tokens;
    let act_per_layer = bt * d * 2 // x + attn_norm
        + bt * d * 2       // mlp_norm + proj_out
        + bt * config.num_heads as usize * config.head_dim * 2 // q, attn_out
        + bt * kv * 2       // k, v
        + bt * mlp * 2      // mlp_up, mlp_act
        + bt * d; // mlp_out
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
        assert!(ckpt[3]); // layer 3: checkpointed
        assert!(ckpt[8]); // layer 8: checkpointed
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

#[cfg(feature = "cuda")]
pub struct GpuQProjectionLora {
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    /// A matrices are stored row-major as [rank, model_dim].
    pub a: Vec<GpuTensor>,
    /// B matrices are stored row-major as [model_dim, rank].
    pub b: Vec<GpuTensor>,
    pub grad_a: Vec<GpuTensor>,
    pub grad_b: Vec<GpuTensor>,
}

#[cfg(feature = "cuda")]
impl GpuQProjectionLora {
    fn new(
        config: &ModelConfig,
        stream: Arc<CudaStream>,
        rank: usize,
        alpha: f32,
    ) -> PgResult<Self> {
        if rank == 0 || rank > config.model_dim {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA rank must be in 1..={}, got {rank}",
                config.model_dim
            )));
        }
        let d = config.model_dim;
        let scale = alpha / rank as f32;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        let mut a = Vec::with_capacity(config.num_layers);
        let mut b = Vec::with_capacity(config.num_layers);
        let mut grad_a = Vec::with_capacity(config.num_layers);
        let mut grad_b = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let mut a_host = vec![0.0f32; rank * d];
            // Warm-start A deterministically and keep B at zero. This matches the
            // frontier LoRA-TTT convention: first update moves B, later updates
            // can move both factors without perturbing score-before-update logits.
            for r in 0..rank {
                for col in 0..d {
                    let x = (layer as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ (r as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                        ^ (col as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
                    let centered = ((x >> 40) as f32 / 16_777_216.0) - 0.5;
                    a_host[r * d + col] = centered * 0.01;
                }
            }
            a.push(GpuTensor::from_host_data_gpu(
                stream.clone(),
                bytemuck::cast_slice(&a_host),
                &[rank, d],
                DType::F32,
            )?);
            b.push(zeros(&[d, rank])?);
            grad_a.push(zeros(&[rank, d])?);
            grad_b.push(zeros(&[d, rank])?);
        }

        Ok(Self {
            rank,
            alpha,
            scale,
            a,
            b,
            grad_a,
            grad_b,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct GpuModel {
    pub config: ModelConfig,
    pub weights: GpuWeights,
    pub gemm: pg_kernels::gemm::GemmEngine,
    pub kernels: pg_kernels::gpu_kernels::GpuKernels,
    pub cuda_cpp_attention: Option<pg_kernels::flash_attn::CudaCppAttention>,
    pub q_lora: Option<GpuQProjectionLora>,
    pub _ctx: Arc<CudaContext>,
}

#[cfg(feature = "cuda")]
impl GpuModel {
    pub fn from_cpu_reference(
        cpu: &GptModel,
        plan: &ExecutionPlan,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        plan.validate_model_config(&cpu.config)?;
        let weights = GpuWeights::from_cpu(cpu, stream.clone())?;
        let gemm = pg_kernels::gemm::GemmEngine::new(stream.clone())?;
        let kernels = pg_kernels::gpu_kernels::GpuKernels::new(ctx.clone(), stream)?;
        let cuda_cpp_attention =
            pg_kernels::flash_attn::CudaCppAttention::new(gemm.stream().clone()).ok();
        Ok(Self {
            config: cpu.config.clone(),
            weights,
            gemm,
            kernels,
            cuda_cpp_attention,
            q_lora: None,
            _ctx: ctx,
        })
    }

    pub fn sync_from_cpu_reference(
        &mut self,
        cpu: &GptModel,
        plan: &ExecutionPlan,
    ) -> PgResult<()> {
        plan.validate_model_config(&cpu.config)?;
        self.weights.sync_from_cpu(cpu)
    }

    pub fn sync_to_cpu_reference(&self, cpu: &mut GptModel, plan: &ExecutionPlan) -> PgResult<()> {
        plan.validate_model_config(&cpu.config)?;
        self.weights.sync_to_cpu(cpu)
    }

    pub fn enable_q_lora(&mut self, rank: usize, alpha: f32) -> PgResult<()> {
        let stream = self.gemm.stream().clone();
        self.q_lora = Some(GpuQProjectionLora::new(&self.config, stream, rank, alpha)?);
        Ok(())
    }

    pub fn q_lora_enabled(&self) -> bool {
        self.q_lora.is_some()
    }

    pub fn zero_q_lora_grads(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for layer in 0..self.config.num_layers {
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_a[layer].cu_ptr(stream)?),
                    0.0,
                    lora.grad_a[layer].numel() as u32,
                )?;
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_b[layer].cu_ptr(stream)?),
                    0.0,
                    lora.grad_b[layer].numel() as u32,
                )?;
            }
        }
        Ok(())
    }

    pub fn reset_q_lora_b(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for b in &lora.b {
                self.kernels
                    .scale_inplace(CudaPtr(b.cu_ptr(stream)?), 0.0, b.numel() as u32)?;
            }
            self.zero_q_lora_grads()?;
        }
        Ok(())
    }

    pub fn step_q_lora_sgd(&self, lr: f32, weight_decay: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for layer in 0..self.config.num_layers {
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.a[layer].cu_ptr(stream)?),
                    CudaPtr(lora.grad_a[layer].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.a[layer].numel() as u32,
                )?;
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.b[layer].cu_ptr(stream)?),
                    CudaPtr(lora.grad_b[layer].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.b[layer].numel() as u32,
                )?;
            }
            self.zero_q_lora_grads()?;
        }
        Ok(())
    }

    fn ln_scale_factor(&self, layer: usize) -> f32 {
        if self.config.ln_scale {
            1.0 / ((layer + 1) as f32).sqrt()
        } else {
            1.0
        }
    }

    fn parallel_residual_enabled(&self) -> bool {
        self.config.parallel_residual
    }

    fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.config.is_recurrent_layer(layer)
    }

    fn run_attention_forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if std::env::var("PG_USE_CPP_NAIVE_ATTENTION")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
            && seq_len == tokens
        {
            if let Some(cuda_cpp_attention) = &self.cuda_cpp_attention {
                return cuda_cpp_attention.forward(
                    q,
                    k,
                    v,
                    out,
                    tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        }

        self.kernels.causal_attention_online_fwd(
            CudaPtr(q),
            CudaPtr(k),
            CudaPtr(v),
            CudaPtr(out),
            tokens as u32,
            seq_len as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
        )
    }

    fn copy_tensor(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.shape() != dst.shape() {
            return Err(pg_core::PgError::ShapeMismatch {
                expected: src.shape().to_vec(),
                got: dst.shape().to_vec(),
            });
        }

        let stream = self.gemm.stream();
        self.kernels.copy_fwd(
            CudaPtr(src.cu_ptr(stream)?),
            CudaPtr(dst.cu_ptr(stream)?),
            src.numel() as u32,
        )
    }

    fn add_inplace(&self, dst: &GpuTensor, src: &GpuTensor, alpha: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if dst.shape() != src.shape() {
            return Err(pg_core::PgError::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }

        let stream = self.gemm.stream();
        self.kernels.add_scaled_fwd(
            CudaPtr(dst.cu_ptr(stream)?),
            CudaPtr(src.cu_ptr(stream)?),
            alpha,
            dst.numel() as u32,
        )
    }

    fn apply_q_lora_forward(&self, layer: usize, buf: &mut GpuActivations) -> PgResult<()> {
        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            let t = buf.attn_norm.shape()[0];
            let d = self.config.model_dim;
            let r = lora.rank;
            unsafe {
                self.gemm.matmul_f32(
                    buf.attn_norm.cu_ptr(stream)?,
                    lora.a[layer].cu_ptr(stream)?,
                    buf.lora_tmp.cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.matmul_f32(
                    buf.lora_tmp.cu_ptr(stream)?,
                    lora.b[layer].cu_ptr(stream)?,
                    buf.lora_delta.cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    0.0,
                )?;
            }
            let lora_delta_q =
                buf.lora_delta
                    .reshape(&[t, self.config.num_heads, self.config.head_dim])?;
            self.add_inplace(&buf.q, &lora_delta_q, 1.0)?;
        }
        Ok(())
    }

    fn backward_q_lora(
        &self,
        layer: usize,
        grad_q_proj: &GpuTensor,
        grad_attn_norm: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            let t = buf.attn_norm.shape()[0];
            let d = self.config.model_dim;
            let r = lora.rank;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    grad_q_proj.cu_ptr(stream)?,
                    buf.lora_tmp.cu_ptr(stream)?,
                    lora.grad_b[layer].cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    1.0,
                )?;
                self.gemm.linear_backward_input_f32(
                    grad_q_proj.cu_ptr(stream)?,
                    lora.b[layer].cu_ptr(stream)?,
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_f32(
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    buf.attn_norm.cu_ptr(stream)?,
                    lora.grad_a[layer].cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    1.0,
                )?;
                self.gemm.linear_backward_input_f32(
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    lora.a[layer].cu_ptr(stream)?,
                    buf.lora_delta.cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            self.add_inplace(grad_attn_norm, &buf.lora_delta, 1.0)?;
        }
        Ok(())
    }

    fn zeros_f32(&self, shape: &[usize]) -> PgResult<GpuTensor> {
        GpuTensor::zeros_gpu(self.gemm.stream().clone(), shape, DType::F32)
    }

    fn mean_loss(&self, logits: &GpuTensor, targets: &GpuTensor, tokens: usize) -> PgResult<f32> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        let losses = self.zeros_f32(&[tokens])?;
        self.kernels.cross_entropy_fwd(
            CudaPtr(logits.cu_ptr(stream)?),
            CudaPtr(targets.cu_ptr(stream)?),
            CudaPtr(losses.cu_ptr(stream)?),
            self.config.vocab_size as u32,
            self.config.logit_softcap,
            tokens as u32,
        )?;
        stream
            .synchronize()
            .map_err(|e| pg_core::PgError::InvalidOp(format!("stream sync failed: {:?}", e)))?;
        let bytes = losses.to_host_bytes()?;
        let values = bytemuck::cast_slice::<u8, f32>(&bytes);
        Ok(values.iter().sum::<f32>() / tokens as f32)
    }

    pub fn cross_entropy_losses(
        &self,
        logits: &GpuTensor,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tokens: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        self.kernels.cross_entropy_fwd(
            CudaPtr(logits.cu_ptr(self.gemm.stream())?),
            CudaPtr(targets.cu_ptr(self.gemm.stream())?),
            CudaPtr(losses.cu_ptr(self.gemm.stream())?),
            self.config.vocab_size as u32,
            self.config.logit_softcap,
            tokens as u32,
        )
    }

    fn block_recompute_for_backward(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuBlockBackwardCache,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let stream = self.gemm.stream();

        self.copy_tensor(layer_x, &buf.x)?;
        self.kernels.residual_mix_fwd(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(x0.cu_ptr(stream)?),
            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
            CudaPtr(buf.x_in.cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        self.kernels.rms_norm_forward(
            CudaPtr(buf.x_in.cu_ptr(stream)?),
            CudaPtr(buf.attn_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;

        let q_w = self.weights.qo_bank.slice_first(layer)?;
        let k_w = self.weights.kv_bank.slice_first(layer)?;
        let v_w = self.weights.kv_bank.slice_first(n + layer)?;
        unsafe {
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                q_w.cu_ptr(stream)?,
                buf.q.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                k_w.cu_ptr(stream)?,
                buf.k.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                v_w.cu_ptr(stream)?,
                buf.v.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
        }
        self.apply_q_lora_forward(layer, buf)?;
        self.copy_tensor(&buf.q, &cache.q_pre_norm)?;
        self.copy_tensor(&buf.k, &cache.k_pre_norm)?;

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_embed.cu_ptr(stream)?),
                CudaPtr(buf.ve_embed_out.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.ve_embed_out.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    buf.ve_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_fwd(
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                self.weights.ve_scale * self.weights.ve_layer_scales_host[ve_idx],
                (t * kv) as u32,
            )?;
        }

        self.kernels.qk_norm_fwd(
            CudaPtr(buf.q.cu_ptr(stream)?),
            hd as u32,
            (t * h) as u32,
            1e-6,
        )?;
        self.kernels.qk_norm_fwd(
            CudaPtr(buf.k.cu_ptr(stream)?),
            hd as u32,
            (t * hkv) as u32,
            1e-6,
        )?;
        if self.config.rope_dims > 0 {
            self.kernels.partial_rope_fwd(
                CudaPtr(buf.q.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                h as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * h) as u32,
            )?;
            self.kernels.partial_rope_fwd(
                CudaPtr(buf.k.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
            )?;
        }
        self.copy_tensor(&buf.q, &cache.q_post_rope)?;

        self.kernels.q_gain_fwd(
            CudaPtr(buf.q.cu_ptr(stream)?),
            CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
            h as u32,
            hd as u32,
            (t * h) as u32,
        )?;
        self.run_attention_forward(
            buf.q.cu_ptr(stream)?,
            buf.k.cu_ptr(stream)?,
            buf.v.cu_ptr(stream)?,
            buf.attn_out.cu_ptr(stream)?,
            t,
            h,
            hkv,
            hd,
            runtime_seq_len,
        )?;

        let attn_src = if layer >= n.saturating_sub(self.config.xsa_last_n) {
            self.kernels.xsa_fwd(
                CudaPtr(buf.attn_out.cu_ptr(stream)?),
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(buf.xsa_out.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hkv as u32,
                hd as u32,
            )?;
            &buf.xsa_out
        } else {
            &buf.attn_out
        };
        let attn_src = if self.config.attn_out_gate_enabled {
            self.kernels.attn_out_gate_fwd(
                CudaPtr(attn_src.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_biases[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            &buf.attn_gated
        } else {
            attn_src
        };

        let o_w = self.weights.qo_bank.slice_first(n + layer)?;
        unsafe {
            self.gemm.matmul_f32(
                attn_src.cu_ptr(stream)?,
                o_w.cu_ptr(stream)?,
                buf.proj_out.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
        }

        self.copy_tensor(&buf.x_in, &buf.x)?;
        self.kernels.residual_add_scale_fwd(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.proj_out.cu_ptr(stream)?),
            CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;
        if self.parallel_residual_enabled() {
            self.copy_tensor(&buf.x_in, &cache.x_after_attn)?;
        } else {
            self.copy_tensor(&buf.x, &cache.x_after_attn)?;
        }

        self.kernels.rms_norm_forward(
            if self.parallel_residual_enabled() {
                CudaPtr(buf.x_in.cu_ptr(stream)?)
            } else {
                CudaPtr(buf.x.cu_ptr(stream)?)
            },
            CudaPtr(buf.mlp_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;

        let up_w = self.weights.mlp_up_bank.slice_first(layer)?;
        let down_w = self.weights.mlp_down_bank.slice_first(layer)?;
        unsafe {
            self.gemm.matmul_f32(
                buf.mlp_norm.cu_ptr(stream)?,
                up_w.cu_ptr(stream)?,
                buf.mlp_up.cu_ptr(stream)?,
                t,
                mlp,
                d,
                1.0,
                0.0,
            )?;
        }
        self.kernels.leaky_relu_sq_forward(
            CudaPtr(buf.mlp_up.cu_ptr(stream)?),
            CudaPtr(buf.mlp_act.cu_ptr(stream)?),
            (t * mlp) as u32,
        )?;
        unsafe {
            self.gemm.matmul_f32(
                buf.mlp_act.cu_ptr(stream)?,
                down_w.cu_ptr(stream)?,
                buf.mlp_out.cu_ptr(stream)?,
                t,
                d,
                mlp,
                1.0,
                0.0,
            )?;
        }
        self.kernels.residual_add_scale_fwd(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.mlp_out.cu_ptr(stream)?),
            CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        Ok(())
    }

    fn block_backward_single(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        block_cache: &mut GpuBlockBackwardCache,
        grad_x: &GpuTensor,
        grad_x0: &GpuTensor,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<GpuTensor> {
        use pg_kernels::gpu_kernels::CudaPtr;

        self.block_recompute_for_backward(
            layer,
            input_ids,
            layer_x,
            x0,
            buf,
            block_cache,
            runtime_seq_len,
        )?;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let stream = self.gemm.stream();

        let grad_x_after_attn = self.zeros_f32(&[t, d])?;
        let grad_mlp_out = self.zeros_f32(&[t, d])?;
        self.kernels.residual_add_scale_bwd(
            CudaPtr(buf.mlp_out.cu_ptr(stream)?),
            CudaPtr(grad_x.cu_ptr(stream)?),
            CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
            CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
            CudaPtr(grad_mlp_out.cu_ptr(stream)?),
            CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        let grad_mlp_act = self.zeros_f32(&[t, mlp])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_mlp_out.cu_ptr(stream)?,
                self.weights
                    .mlp_down_bank
                    .slice_first(layer)?
                    .cu_ptr(stream)?,
                grad_mlp_act.cu_ptr(stream)?,
                t,
                d,
                mlp,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_mlp_out.cu_ptr(stream)?,
                buf.mlp_act.cu_ptr(stream)?,
                grads.mlp_down_bank.slice_first(layer)?.cu_ptr(stream)?,
                t,
                d,
                mlp,
                1.0,
                1.0,
            )?;
        }

        let grad_mlp_up = self.zeros_f32(&[t, mlp])?;
        self.kernels.leaky_relu_sq_backward(
            CudaPtr(buf.mlp_up.cu_ptr(stream)?),
            CudaPtr(grad_mlp_act.cu_ptr(stream)?),
            CudaPtr(grad_mlp_up.cu_ptr(stream)?),
            (t * mlp) as u32,
        )?;

        let grad_mlp_norm = self.zeros_f32(&[t, d])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_mlp_up.cu_ptr(stream)?,
                self.weights
                    .mlp_up_bank
                    .slice_first(layer)?
                    .cu_ptr(stream)?,
                grad_mlp_norm.cu_ptr(stream)?,
                t,
                mlp,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_mlp_up.cu_ptr(stream)?,
                buf.mlp_norm.cu_ptr(stream)?,
                grads.mlp_up_bank.slice_first(layer)?.cu_ptr(stream)?,
                t,
                mlp,
                d,
                1.0,
                1.0,
            )?;
        }

        let grad_x_pre_mlp_norm = self.zeros_f32(&[t, d])?;
        self.kernels.rms_norm_backward(
            CudaPtr(block_cache.x_after_attn.cu_ptr(stream)?),
            CudaPtr(grad_mlp_norm.cu_ptr(stream)?),
            CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;
        if !self.parallel_residual_enabled() {
            self.add_inplace(&grad_x_after_attn, &grad_x_pre_mlp_norm, 1.0)?;
        }

        let grad_x_in = self.zeros_f32(&[t, d])?;
        let grad_proj_out = self.zeros_f32(&[t, d])?;
        self.kernels.residual_add_scale_bwd(
            CudaPtr(buf.proj_out.cu_ptr(stream)?),
            CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
            CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
            CudaPtr(grad_x_in.cu_ptr(stream)?),
            CudaPtr(grad_proj_out.cu_ptr(stream)?),
            CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;
        if self.parallel_residual_enabled() {
            self.add_inplace(&grad_x_in, &grad_x_pre_mlp_norm, 1.0)?;
        }

        let grad_attn_result = self.zeros_f32(&[t, h, hd])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_proj_out.cu_ptr(stream)?,
                self.weights
                    .qo_bank
                    .slice_first(n + layer)?
                    .cu_ptr(stream)?,
                grad_attn_result.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_proj_out.cu_ptr(stream)?,
                if self.config.attn_out_gate_enabled {
                    buf.attn_gated.cu_ptr(stream)?
                } else if layer >= n.saturating_sub(self.config.xsa_last_n) {
                    buf.xsa_out.cu_ptr(stream)?
                } else {
                    buf.attn_out.cu_ptr(stream)?
                },
                grads.qo_bank.slice_first(n + layer)?.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                1.0,
            )?;
        }

        let grad_attn_pre_gate = if self.config.attn_out_gate_enabled {
            let raw_src = if layer >= n.saturating_sub(self.config.xsa_last_n) {
                &buf.xsa_out
            } else {
                &buf.attn_out
            };
            let grad_raw = self.zeros_f32(&[t, h, hd])?;
            self.kernels.scale_inplace(
                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                0.0,
                (t * d) as u32,
            )?;
            self.kernels.attn_out_gate_bwd(
                CudaPtr(raw_src.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(grad_raw.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                CudaPtr(grads.block_attn_gate_weight[layer].cu_ptr(stream)?),
                CudaPtr(grads.block_attn_gate_bias[layer].cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            grad_raw
        } else {
            grad_attn_result
        };

        let grad_attn_out = self.zeros_f32(&[t, h, hd])?;
        let grad_v_xsa = self.zeros_f32(&[t, hkv, hd])?;
        if layer >= n.saturating_sub(self.config.xsa_last_n) {
            self.kernels.xsa_bwd(
                CudaPtr(buf.attn_out.cu_ptr(stream)?),
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(grad_attn_pre_gate.cu_ptr(stream)?),
                CudaPtr(grad_attn_out.cu_ptr(stream)?),
                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hkv as u32,
                hd as u32,
            )?;
        } else {
            self.copy_tensor(&grad_attn_pre_gate, &grad_attn_out)?;
        }

        let grad_q_post_gain = self.zeros_f32(&[t, h, hd])?;
        let grad_k_attn = self.zeros_f32(&[t, hkv, hd])?;
        let grad_v_projection = self.zeros_f32(&[t, hkv, hd])?;
        self.kernels.causal_attention_online_bwd(
            CudaPtr(buf.q.cu_ptr(stream)?),
            CudaPtr(buf.k.cu_ptr(stream)?),
            CudaPtr(buf.v.cu_ptr(stream)?),
            CudaPtr(grad_attn_out.cu_ptr(stream)?),
            CudaPtr(grad_q_post_gain.cu_ptr(stream)?),
            CudaPtr(grad_k_attn.cu_ptr(stream)?),
            CudaPtr(grad_v_projection.cu_ptr(stream)?),
            t as u32,
            runtime_seq_len as u32,
            h as u32,
            hkv as u32,
            hd as u32,
        )?;
        if layer >= n.saturating_sub(self.config.xsa_last_n) {
            self.add_inplace(&grad_v_projection, &grad_v_xsa, 1.0)?;
        }

        let grad_q_post_rope = self.zeros_f32(&[t, h, hd])?;
        self.kernels.q_gain_bwd(
            CudaPtr(block_cache.q_post_rope.cu_ptr(stream)?),
            CudaPtr(grad_q_post_gain.cu_ptr(stream)?),
            CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
            CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
            CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
            h as u32,
            hd as u32,
            (t * h) as u32,
        )?;
        if self.config.rope_dims > 0 {
            self.kernels.partial_rope_bwd(
                CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                h as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * h) as u32,
            )?;
            self.kernels.partial_rope_bwd(
                CudaPtr(grad_k_attn.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
            )?;
        }

        let grad_q_proj = self.zeros_f32(&[t, h, hd])?;
        let grad_k_proj = self.zeros_f32(&[t, hkv, hd])?;
        self.kernels.qk_norm_bwd(
            CudaPtr(block_cache.q_pre_norm.cu_ptr(stream)?),
            CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
            CudaPtr(grad_q_proj.cu_ptr(stream)?),
            hd as u32,
            (t * h) as u32,
            1e-6,
        )?;
        self.kernels.qk_norm_bwd(
            CudaPtr(block_cache.k_pre_norm.cu_ptr(stream)?),
            CudaPtr(grad_k_attn.cu_ptr(stream)?),
            CudaPtr(grad_k_proj.cu_ptr(stream)?),
            hd as u32,
            (t * hkv) as u32,
            1e-6,
        )?;

        let grad_attn_norm = self.zeros_f32(&[t, d])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_q_proj.cu_ptr(stream)?,
                self.weights.qo_bank.slice_first(layer)?.cu_ptr(stream)?,
                grad_attn_norm.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_q_proj.cu_ptr(stream)?,
                buf.attn_norm.cu_ptr(stream)?,
                grads.qo_bank.slice_first(layer)?.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                1.0,
            )?;
        }
        self.backward_q_lora(layer, &grad_q_proj, &grad_attn_norm, buf)?;

        let grad_attn_norm_k = self.zeros_f32(&[t, d])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_k_proj.cu_ptr(stream)?,
                self.weights.kv_bank.slice_first(layer)?.cu_ptr(stream)?,
                grad_attn_norm_k.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_k_proj.cu_ptr(stream)?,
                buf.attn_norm.cu_ptr(stream)?,
                grads.kv_bank.slice_first(layer)?.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                1.0,
            )?;
        }
        self.add_inplace(&grad_attn_norm, &grad_attn_norm_k, 1.0)?;

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            let layer_scale = self.weights.ve_layer_scales_host[ve_idx];
            self.kernels.dot_accumulate(
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                CudaPtr(grads.ve_scale.cu_ptr(stream)?),
                layer_scale,
                (t * kv) as u32,
            )?;
            self.kernels.dot_accumulate(
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                CudaPtr(grads.ve_layer_scales.slice_first(ve_idx)?.cu_ptr(stream)?),
                self.weights.ve_scale,
                (t * kv) as u32,
            )?;

            let grad_projected = self.zeros_f32(&[t, kv])?;
            self.kernels.add_scaled_fwd(
                CudaPtr(grad_projected.cu_ptr(stream)?),
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                self.weights.ve_scale * layer_scale,
                (t * kv) as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    grad_projected.cu_ptr(stream)?,
                    buf.ve_embed_out.cu_ptr(stream)?,
                    grads.ve_proj.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    1.0,
                )?;
            }
            let grad_ve_embed_out = self.zeros_f32(&[t, self.config.ve_dim])?;
            unsafe {
                self.gemm.linear_backward_input_f32(
                    grad_projected.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    grad_ve_embed_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.embedding_gather_bwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(grad_ve_embed_out.cu_ptr(stream)?),
                CudaPtr(grads.ve_embed.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
        }

        let grad_attn_norm_v = self.zeros_f32(&[t, d])?;
        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_v_projection.cu_ptr(stream)?,
                self.weights
                    .kv_bank
                    .slice_first(n + layer)?
                    .cu_ptr(stream)?,
                grad_attn_norm_v.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_v_projection.cu_ptr(stream)?,
                buf.attn_norm.cu_ptr(stream)?,
                grads.kv_bank.slice_first(n + layer)?.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                1.0,
            )?;
        }
        self.add_inplace(&grad_attn_norm, &grad_attn_norm_v, 1.0)?;
        if self.config.attn_out_gate_enabled {
            self.add_inplace(&grad_attn_norm, &buf.attn_gate_grad_input, 1.0)?;
        }

        let grad_x_in_from_norm = self.zeros_f32(&[t, d])?;
        self.kernels.rms_norm_backward(
            CudaPtr(buf.x_in.cu_ptr(stream)?),
            CudaPtr(grad_attn_norm.cu_ptr(stream)?),
            CudaPtr(grad_x_in_from_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;
        self.add_inplace(&grad_x_in, &grad_x_in_from_norm, 1.0)?;

        let grad_x_out = self.zeros_f32(&[t, d])?;
        self.kernels.residual_mix_bwd(
            CudaPtr(layer_x.cu_ptr(stream)?),
            CudaPtr(x0.cu_ptr(stream)?),
            CudaPtr(grad_x_in.cu_ptr(stream)?),
            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
            CudaPtr(grad_x_out.cu_ptr(stream)?),
            CudaPtr(grad_x0.cu_ptr(stream)?),
            CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        Ok(grad_x_out)
    }

    fn block_backward(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        block_cache: &mut GpuBlockBackwardCache,
        grad_x: &GpuTensor,
        grad_x0: &GpuTensor,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<GpuTensor> {
        if self.is_recurrent_layer(layer) {
            let t = input_ids.shape().iter().product::<usize>();
            let pass1_out = self.zeros_f32(&[t, self.config.model_dim])?;
            self.block_recompute_for_backward(
                layer,
                input_ids,
                layer_x,
                x0,
                buf,
                block_cache,
                runtime_seq_len,
            )?;
            self.copy_tensor(&buf.x, &pass1_out)?;
            let grad_mid = self.block_backward_single(
                layer,
                input_ids,
                &pass1_out,
                x0,
                buf,
                block_cache,
                grad_x,
                grad_x0,
                grads,
                runtime_seq_len,
            )?;
            return self.block_backward_single(
                layer,
                input_ids,
                layer_x,
                x0,
                buf,
                block_cache,
                &grad_mid,
                grad_x0,
                grads,
                runtime_seq_len,
            );
        }

        self.block_backward_single(
            layer,
            input_ids,
            layer_x,
            x0,
            buf,
            block_cache,
            grad_x,
            grad_x0,
            grads,
            runtime_seq_len,
        )
    }

    fn block_forward_once(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);
        let attn_norm = CudaPtr(buf.attn_norm.cu_ptr(stream)?);
        let mlp_norm = CudaPtr(buf.mlp_norm.cu_ptr(stream)?);
        let q = CudaPtr(buf.q.cu_ptr(stream)?);
        let k = CudaPtr(buf.k.cu_ptr(stream)?);
        let v = CudaPtr(buf.v.cu_ptr(stream)?);
        let attn_out = CudaPtr(buf.attn_out.cu_ptr(stream)?);
        let xsa_out = CudaPtr(buf.xsa_out.cu_ptr(stream)?);
        let proj_out = CudaPtr(buf.proj_out.cu_ptr(stream)?);
        let mlp_up = CudaPtr(buf.mlp_up.cu_ptr(stream)?);
        let mlp_act = CudaPtr(buf.mlp_act.cu_ptr(stream)?);
        let mlp_out = CudaPtr(buf.mlp_out.cu_ptr(stream)?);

        self.kernels.residual_mix_fwd(
            x,
            x0,
            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
            x_in,
            d as u32,
            (t * d) as u32,
        )?;

        self.kernels.rms_norm_forward(
            x_in,
            attn_norm,
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;

        let q_w = self.weights.qo_bank.slice_first(layer)?.cu_ptr(stream)?;
        let k_w = self.weights.kv_bank.slice_first(layer)?.cu_ptr(stream)?;
        let v_w = self
            .weights
            .kv_bank
            .slice_first(n + layer)?
            .cu_ptr(stream)?;
        unsafe {
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                q_w,
                buf.q.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                k_w,
                buf.k.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.matmul_f32(
                buf.attn_norm.cu_ptr(stream)?,
                v_w,
                buf.v.cu_ptr(stream)?,
                t,
                kv,
                d,
                1.0,
                0.0,
            )?;
        }
        self.apply_q_lora_forward(layer, buf)?;

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_embed.cu_ptr(stream)?),
                CudaPtr(buf.ve_embed_out.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.ve_embed_out.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    buf.ve_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_fwd(
                v,
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                self.weights.ve_scale * self.weights.ve_layer_scales_host[ve_idx],
                (t * kv) as u32,
            )?;
        }

        self.kernels
            .qk_norm_fwd(q, hd as u32, (t * h) as u32, 1e-6)?;
        self.kernels
            .qk_norm_fwd(k, hd as u32, (t * hkv) as u32, 1e-6)?;
        if self.config.rope_dims > 0 {
            self.kernels.partial_rope_fwd(
                q,
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                h as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * h) as u32,
            )?;
            self.kernels.partial_rope_fwd(
                k,
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
            )?;
        }
        self.kernels.q_gain_fwd(
            q,
            CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
            h as u32,
            hd as u32,
            (t * h) as u32,
        )?;

        self.run_attention_forward(
            buf.q.cu_ptr(stream)?,
            buf.k.cu_ptr(stream)?,
            buf.v.cu_ptr(stream)?,
            buf.attn_out.cu_ptr(stream)?,
            t,
            h,
            hkv,
            hd,
            runtime_seq_len,
        )?;

        let attn_src_tensor = if layer >= n.saturating_sub(self.config.xsa_last_n) {
            self.kernels.xsa_fwd(
                attn_out, v, xsa_out, t as u32, h as u32, hkv as u32, hd as u32,
            )?;
            &buf.xsa_out
        } else {
            &buf.attn_out
        };
        let attn_src_tensor = if self.config.attn_out_gate_enabled {
            self.kernels.attn_out_gate_fwd(
                CudaPtr(attn_src_tensor.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_biases[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            &buf.attn_gated
        } else {
            attn_src_tensor
        };

        let o_w = self
            .weights
            .qo_bank
            .slice_first(n + layer)?
            .cu_ptr(stream)?;
        unsafe {
            self.gemm.matmul_f32(
                attn_src_tensor.cu_ptr(stream)?,
                o_w,
                buf.proj_out.cu_ptr(stream)?,
                t,
                d,
                d,
                1.0,
                0.0,
            )?;
        }

        self.kernels.copy_fwd(x_in, x, (t * d) as u32)?;
        self.kernels.residual_add_scale_fwd(
            x,
            proj_out,
            CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        let mlp_norm_input = if self.parallel_residual_enabled() {
            x_in
        } else {
            x
        };
        self.kernels.rms_norm_forward(
            mlp_norm_input,
            mlp_norm,
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;

        let up_w = self
            .weights
            .mlp_up_bank
            .slice_first(layer)?
            .cu_ptr(stream)?;
        let down_w = self
            .weights
            .mlp_down_bank
            .slice_first(layer)?
            .cu_ptr(stream)?;
        unsafe {
            self.gemm.matmul_f32(
                buf.mlp_norm.cu_ptr(stream)?,
                up_w,
                buf.mlp_up.cu_ptr(stream)?,
                t,
                mlp,
                d,
                1.0,
                0.0,
            )?;
        }
        self.kernels
            .leaky_relu_sq_forward(mlp_up, mlp_act, (t * mlp) as u32)?;
        unsafe {
            self.gemm.matmul_f32(
                buf.mlp_act.cu_ptr(stream)?,
                down_w,
                buf.mlp_out.cu_ptr(stream)?,
                t,
                d,
                mlp,
                1.0,
                0.0,
            )?;
        }
        self.kernels.residual_add_scale_fwd(
            x,
            mlp_out,
            CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        Ok(())
    }

    fn block_forward(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        self.block_forward_once(layer, input_ids, buf, runtime_seq_len)?;
        if self.is_recurrent_layer(layer) {
            self.block_forward_once(layer, input_ids, buf, runtime_seq_len)?;
        }
        Ok(())
    }

    pub fn forward(&self, input_ids: &GpuTensor, buf: &mut GpuActivations) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.forward_with_seq_len(input_ids, buf, t)
    }

    pub fn forward_with_seq_len(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            x,
            d as u32,
            t as u32,
        )?;

        if self.config.bigram_vocab_size > 0 {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_fwd(
                x,
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                self.weights.bigram_scale,
                (t * d) as u32,
            )?;
        }

        self.kernels
            .rms_norm_forward(x, x_in, t as u32, d as u32, 1.0, 1e-6)?;
        self.kernels.smear_gate_fwd(
            x_in,
            CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
            x,
            t as u32,
            runtime_seq_len as u32,
            d as u32,
        )?;
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        for layer in 0..n_enc {
            self.block_forward(layer, input_ids, buf, runtime_seq_len)?;
            self.kernels.copy_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.encoder_skips[layer].cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        for i in 0..n_dec {
            if i < self.config.num_skip_weights() {
                let skip_idx = self.config.num_skip_weights() - 1 - i;
                self.kernels.residual_add_scale_fwd(
                    CudaPtr(buf.x.cu_ptr(stream)?),
                    CudaPtr(buf.encoder_skips[skip_idx].cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
            self.block_forward(n_enc + i, input_ids, buf, runtime_seq_len)?;
        }

        self.kernels.rms_norm_forward(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.x_in.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;
        unsafe {
            self.gemm.matmul_f32(
                buf.x_in.cu_ptr(stream)?,
                self.weights.tok_emb.cu_ptr(stream)?,
                buf.logits.cu_ptr(stream)?,
                t,
                vocab,
                d,
                1.0,
                0.0,
            )?;
        }
        Ok(())
    }

    pub fn forward_with_cache(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuForwardCache,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.forward_with_cache_seq_len(input_ids, buf, cache, t)
    }

    pub fn forward_with_cache_seq_len(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuForwardCache,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            x,
            d as u32,
            t as u32,
        )?;

        if self.config.bigram_vocab_size > 0 {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_fwd(
                x,
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                self.weights.bigram_scale,
                (t * d) as u32,
            )?;
        }

        self.copy_tensor(&buf.x, &cache.x_post_embed)?;

        self.kernels
            .rms_norm_forward(x, x_in, t as u32, d as u32, 1.0, 1e-6)?;
        self.copy_tensor(&buf.x_in, &cache.x_post_norm)?;

        self.kernels.smear_gate_fwd(
            x_in,
            CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
            x,
            t as u32,
            runtime_seq_len as u32,
            d as u32,
        )?;
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;
        self.copy_tensor(&buf.x, &cache.x0)?;

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        for layer in 0..n_enc {
            self.copy_tensor(&buf.x, &cache.layer_x[layer])?;
            self.block_forward(layer, input_ids, buf, runtime_seq_len)?;
            self.kernels.copy_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.encoder_skips[layer].cu_ptr(stream)?),
                (t * d) as u32,
            )?;
            self.copy_tensor(&buf.x, &cache.skips[layer])?;
        }

        for i in 0..n_dec {
            if i < self.config.num_skip_weights() {
                let skip_idx = self.config.num_skip_weights() - 1 - i;
                self.kernels.residual_add_scale_fwd(
                    CudaPtr(buf.x.cu_ptr(stream)?),
                    CudaPtr(buf.encoder_skips[skip_idx].cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
            self.copy_tensor(&buf.x, &cache.layer_x[n_enc + i])?;
            self.block_forward(n_enc + i, input_ids, buf, runtime_seq_len)?;
        }

        self.copy_tensor(&buf.x, &cache.x_final)?;

        self.kernels.rms_norm_forward(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.x_in.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;
        unsafe {
            self.gemm.matmul_f32(
                buf.x_in.cu_ptr(stream)?,
                self.weights.tok_emb.cu_ptr(stream)?,
                buf.logits.cu_ptr(stream)?,
                t,
                vocab,
                d,
                1.0,
                0.0,
            )?;
        }
        Ok(())
    }

    pub fn backward_output_loss_only(
        &self,
        cache: &GpuForwardCache,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<GpuTensor> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();

        let grad_logits = GpuTensor::zeros_gpu(stream.clone(), &[t, vocab], DType::F32)?;
        let grad_x = GpuTensor::zeros_gpu(stream.clone(), &[t, d], DType::F32)?;
        let grad_pre_norm = GpuTensor::zeros_gpu(stream.clone(), &[t, d], DType::F32)?;

        self.kernels.cross_entropy_bwd(
            CudaPtr(buf.logits.cu_ptr(stream)?),
            CudaPtr(targets.cu_ptr(stream)?),
            CudaPtr(grad_logits.cu_ptr(stream)?),
            vocab as u32,
            self.config.logit_softcap,
            1.0 / t as f32,
            t as u32,
        )?;

        unsafe {
            self.gemm.linear_backward_input_f32(
                grad_logits.cu_ptr(stream)?,
                self.weights.tok_emb.cu_ptr(stream)?,
                grad_x.cu_ptr(stream)?,
                t,
                vocab,
                d,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_logits.cu_ptr(stream)?,
                buf.x_in.cu_ptr(stream)?,
                grads.tok_emb.cu_ptr(stream)?,
                t,
                vocab,
                d,
                1.0,
                1.0,
            )?;
        }

        self.kernels.rms_norm_backward(
            CudaPtr(cache.x_final.cu_ptr(stream)?),
            CudaPtr(grad_x.cu_ptr(stream)?),
            CudaPtr(grad_pre_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;

        Ok(grad_pre_norm)
    }

    pub fn backward_with_state(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<f32> {
        let t = input_ids.shape().iter().product::<usize>();
        self.backward_with_state_seq_len_loss_mode(input_ids, targets, buf, state, grads, true, t)
    }

    pub fn backward_with_state_no_loss(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.backward_with_state_seq_len_loss_mode(input_ids, targets, buf, state, grads, false, t)
            .map(|_| ())
    }

    pub fn backward_with_state_seq_len(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<f32> {
        self.backward_with_state_seq_len_loss_mode(
            input_ids,
            targets,
            buf,
            state,
            grads,
            true,
            runtime_seq_len,
        )
    }

    pub fn backward_with_state_seq_len_no_loss(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        self.backward_with_state_seq_len_loss_mode(
            input_ids,
            targets,
            buf,
            state,
            grads,
            false,
            runtime_seq_len,
        )
        .map(|_| ())
    }

    fn backward_with_state_seq_len_loss_mode(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        compute_loss: bool,
        runtime_seq_len: usize,
    ) -> PgResult<f32> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        let n_skip = self.config.num_skip_weights();
        let stream = self.gemm.stream();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);

        self.forward_with_cache_seq_len(input_ids, buf, &mut state.cache, runtime_seq_len)?;
        let loss = if compute_loss {
            self.mean_loss(&buf.logits, targets, t)?
        } else {
            0.0
        };

        let mut grad_x = self.backward_output_loss_only(&state.cache, buf, targets, grads)?;
        let grad_x0 = self.zeros_f32(&[t, d])?;
        let grad_encoder_skips: Vec<GpuTensor> = (0..n_skip)
            .map(|_| self.zeros_f32(&[t, d]))
            .collect::<PgResult<_>>()?;

        for i in (0..n_dec).rev() {
            let bi = n_enc + i;
            grad_x = self.block_backward(
                bi,
                input_ids,
                &state.cache.layer_x[bi],
                &state.cache.x0,
                buf,
                &mut state.block_cache,
                &grad_x,
                &grad_x0,
                grads,
                runtime_seq_len,
            )?;

            if i < n_skip {
                let enc_layer = n_enc - 1 - i;
                let grad_x_post_skip = self.zeros_f32(&[t, d])?;
                self.kernels.residual_add_scale_bwd(
                    CudaPtr(state.cache.skips[enc_layer].cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    CudaPtr(grad_x_post_skip.cu_ptr(stream)?),
                    CudaPtr(grad_encoder_skips[enc_layer].cu_ptr(stream)?),
                    CudaPtr(grads.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
                grad_x = grad_x_post_skip;
            }
        }

        for i in (0..n_enc).rev() {
            if i < n_skip {
                self.add_inplace(&grad_x, &grad_encoder_skips[i], 1.0)?;
            }
            grad_x = self.block_backward(
                i,
                input_ids,
                &state.cache.layer_x[i],
                &state.cache.x0,
                buf,
                &mut state.block_cache,
                &grad_x,
                &grad_x0,
                grads,
                runtime_seq_len,
            )?;
        }

        self.add_inplace(&grad_x, &grad_x0, 1.0)?;

        let grad_x_smear = self.zeros_f32(&[t, d])?;
        let grad_x_prev = self.zeros_f32(&[t, d])?;
        self.kernels.smear_gate_bwd(
            CudaPtr(state.cache.x_post_norm.cu_ptr(stream)?),
            CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
            CudaPtr(grad_x.cu_ptr(stream)?),
            CudaPtr(grad_x_smear.cu_ptr(stream)?),
            CudaPtr(grad_x_prev.cu_ptr(stream)?),
            CudaPtr(grads.smear_gate.cu_ptr(stream)?),
            t as u32,
            runtime_seq_len as u32,
            d as u32,
        )?;

        let grad_x_post_norm = self.zeros_f32(&[t, d])?;
        self.copy_tensor(&grad_x_smear, &grad_x_post_norm)?;
        if t > 1 {
            let dst = grad_x_post_norm.slice_range(0, t - 1)?;
            let src = grad_x_prev.slice_range(1, t)?;
            self.add_inplace(&dst, &src, 1.0)?;
        }

        let grad_x_post_embed = self.zeros_f32(&[t, d])?;
        self.kernels.rms_norm_backward(
            CudaPtr(state.cache.x_post_embed.cu_ptr(stream)?),
            CudaPtr(grad_x_post_norm.cu_ptr(stream)?),
            CudaPtr(grad_x_post_embed.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;

        if self.config.bigram_vocab_size > 0 {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.dot_accumulate(
                CudaPtr(grad_x_post_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(grads.bigram_scale.cu_ptr(stream)?),
                1.0,
                (t * d) as u32,
            )?;

            let grad_bigram_proj_out = self.zeros_f32(&[t, d])?;
            self.kernels.add_scaled_fwd(
                CudaPtr(grad_bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(grad_x_post_embed.cu_ptr(stream)?),
                self.weights.bigram_scale,
                (t * d) as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    grad_bigram_proj_out.cu_ptr(stream)?,
                    buf.bigram_out.cu_ptr(stream)?,
                    grads.bigram_proj.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    1.0,
                )?;
            }
            let grad_bigram_out = self.zeros_f32(&[t, self.config.bigram_dim])?;
            unsafe {
                self.gemm.linear_backward_input_f32(
                    grad_bigram_proj_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    grad_bigram_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.bigram_hash_embed_bwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(grad_bigram_out.cu_ptr(stream)?),
                CudaPtr(grads.bigram_embed.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
        }

        self.kernels.embedding_gather_bwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(grad_x_post_embed.cu_ptr(stream)?),
            CudaPtr(grads.tok_emb.cu_ptr(stream)?),
            d as u32,
            t as u32,
        )?;

        Ok(loss)
    }

    pub fn backward(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<f32> {
        let t = input_ids.shape().iter().product::<usize>();
        let stream = self.gemm.stream().clone();
        let mut state = GpuBackwardState::new(&self.config, t, stream)?;
        self.backward_with_state(input_ids, targets, buf, &mut state, grads)
    }
}
