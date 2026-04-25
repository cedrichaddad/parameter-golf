#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgError, PgResult};
#[cfg(feature = "cuda")]
use pg_model::gpu::{GpuActivations, GpuBackwardState, GpuGradBuffers, GpuModel};
#[cfg(feature = "cuda")]
use pg_model::{ExecutionPlan, GptModel};

#[cfg(feature = "cuda")]
use crate::sliding::build_ttt_chunks;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GpuLoraPhasedTttConfig {
    pub stride: usize,
    pub seq_len: usize,
    pub chunk_tokens: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub phases: usize,
    pub weight_decay: f32,
    pub lr: f32,
}

#[cfg(feature = "cuda")]
impl GpuLoraPhasedTttConfig {
    pub fn from_plan(plan: &ExecutionPlan, token_count: usize) -> Self {
        let seq_len = plan
            .run_spec
            .model
            .eval_seq_len
            .min(token_count.saturating_sub(1))
            .max(1);
        Self {
            stride: plan.eval_plan.stride,
            seq_len,
            chunk_tokens: plan.eval_plan.chunk_tokens,
            lora_rank: plan.eval_plan.lora_rank,
            lora_alpha: plan.eval_plan.lora_alpha,
            phases: plan.eval_plan.phased_ttt_phases.max(1),
            weight_decay: plan.eval_plan.phased_ttt_weight_decay,
            lr: 0.01,
        }
    }
}

#[cfg(feature = "cuda")]
pub fn eval_gpu_lora_phased_ttt(
    cpu_model: &GptModel,
    plan: &ExecutionPlan,
    val_tokens: &[u32],
    base_bytes: &[f32],
    cfg: &GpuLoraPhasedTttConfig,
) -> PgResult<(f64, f64)> {
    if val_tokens.len() < 2 {
        return Ok((0.0, 0.0));
    }
    let ctx = cudarc::driver::CudaContext::new(0)
        .map_err(|e| PgError::InvalidOp(format!("CUDA context init failed: {e:?}")))?;
    let stream = ctx.default_stream();
    let mut model = GpuModel::from_cpu_reference(cpu_model, plan, ctx, stream.clone())?;
    model.enable_q_lora(cfg.lora_rank, cfg.lora_alpha)?;

    let seq_len = cfg.seq_len.min(val_tokens.len() - 1).max(1);
    let mut input_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?;
    let mut target_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?;
    let losses_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::F32)?;
    let mut activations = GpuActivations::new(&cpu_model.config, seq_len, stream.clone())?;
    let mut backward_state = GpuBackwardState::new(&cpu_model.config, seq_len, stream.clone())?;
    let mut grads = GpuGradBuffers::new(&cpu_model.config, stream.clone())?;

    let total_tokens = val_tokens.len() - 1;
    let chunks = build_ttt_chunks(total_tokens, cfg.chunk_tokens, cfg.stride, seq_len);
    let num_chunks = chunks.len().max(1);
    let mut total_loss = 0.0f64;
    let mut total_scored = 0u64;
    let mut total_bytes = 0.0f64;
    let mut current_phase = 0usize;

    for (ci, chunk) in chunks.iter().enumerate() {
        let phase = (ci * cfg.phases / num_chunks).min(cfg.phases - 1);
        if phase != current_phase {
            // Warm-start A across phases and reset B so each phase begins from
            // zero-delta score-first semantics with the accumulated subspace.
            model.reset_q_lora_b()?;
            current_phase = phase;
        }

        let (loss, scored, bytes) = score_chunk_gpu(
            &model,
            val_tokens,
            base_bytes,
            chunk,
            cfg.stride,
            seq_len,
            &mut input_gpu,
            &mut target_gpu,
            &losses_gpu,
            &mut activations,
        )?;
        total_loss += loss;
        total_scored += scored;
        total_bytes += bytes;

        if ci + 1 < chunks.len() {
            let phase_lr_scale = 0.5f32.powi(phase as i32);
            train_chunk_gpu_lora(
                &model,
                val_tokens,
                chunk.chunk_start,
                chunk.chunk_end.min(val_tokens.len() - 1),
                seq_len,
                cfg.lr * phase_lr_scale,
                cfg.weight_decay,
                &mut input_gpu,
                &mut target_gpu,
                &mut activations,
                &mut backward_state,
                &mut grads,
            )?;
        }
    }

    let val_loss = if total_scored > 0 {
        total_loss / total_scored as f64
    } else {
        0.0
    };
    let bits_per_token = val_loss / 2.0f64.ln();
    let tokens_per_byte = if total_bytes > 0.0 {
        total_scored as f64 / total_bytes
    } else {
        1.0
    };
    Ok((val_loss, bits_per_token * tokens_per_byte))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn score_chunk_gpu(
    model: &GpuModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    chunk: &crate::sliding::TttChunk,
    stride: usize,
    seq_len: usize,
    input_gpu: &mut GpuTensor,
    target_gpu: &mut GpuTensor,
    losses_gpu: &GpuTensor,
    activations: &mut GpuActivations,
) -> PgResult<(f64, u64, f64)> {
    let total_tokens = val_tokens.len() - 1;
    let mut loss_sum = 0.0f64;
    let mut token_count = 0u64;
    let mut byte_count = 0.0f64;
    let mut input_host = vec![0u32; seq_len];
    let mut target_host = vec![0u32; seq_len];

    for &ws in &chunk.windows {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end - ws;
        if wlen == 0 {
            continue;
        }
        input_host.fill(0);
        target_host.fill(0);
        input_host[..wlen].copy_from_slice(&val_tokens[ws..end]);
        target_host[..wlen].copy_from_slice(&val_tokens[ws + 1..end + 1]);
        input_gpu.copy_from_host_bytes(bytemuck::cast_slice(&input_host))?;
        target_gpu.copy_from_host_bytes(bytemuck::cast_slice(&target_host))?;

        model.forward_with_seq_len(input_gpu, activations, seq_len)?;
        model.cross_entropy_losses(&activations.logits, target_gpu, losses_gpu, seq_len)?;
        let losses = losses_gpu.to_host_bytes()?;
        let losses = bytemuck::cast_slice::<u8, f32>(&losses);
        let score_start = if ws == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        for (t, nll) in losses.iter().take(wlen).enumerate().skip(score_start) {
            loss_sum += *nll as f64;
            token_count += 1;
            let tok_idx = ws + t;
            byte_count += base_bytes.get(tok_idx).copied().unwrap_or(1.0) as f64;
        }
    }
    Ok((loss_sum, token_count, byte_count))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn train_chunk_gpu_lora(
    model: &GpuModel,
    val_tokens: &[u32],
    chunk_start: usize,
    chunk_end: usize,
    seq_len: usize,
    lr: f32,
    weight_decay: f32,
    input_gpu: &mut GpuTensor,
    target_gpu: &mut GpuTensor,
    activations: &mut GpuActivations,
    backward_state: &mut GpuBackwardState,
    grads: &mut GpuGradBuffers,
) -> PgResult<()> {
    let mut input_host = vec![0u32; seq_len];
    let mut target_host = vec![0u32; seq_len];
    let mut ws = chunk_start;
    while ws + seq_len < chunk_end + 1 && ws + seq_len < val_tokens.len() {
        input_host.copy_from_slice(&val_tokens[ws..ws + seq_len]);
        target_host.copy_from_slice(&val_tokens[ws + 1..ws + seq_len + 1]);
        input_gpu.copy_from_host_bytes(bytemuck::cast_slice(&input_host))?;
        target_gpu.copy_from_host_bytes(bytemuck::cast_slice(&target_host))?;

        grads.zero(&model.kernels)?;
        model.zero_q_lora_grads()?;
        model.backward_with_state_seq_len_no_loss(
            input_gpu,
            target_gpu,
            activations,
            backward_state,
            grads,
            seq_len,
        )?;
        model.step_q_lora_sgd(lr, weight_decay)?;
        ws += seq_len;
    }
    Ok(())
}
