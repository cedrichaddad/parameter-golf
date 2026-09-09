use std::ffi::c_void;
use std::mem;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    ComputePipelineStateRef, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use pg_core::{PgError, PgResult};

use super::{LiteEvalStats, LiteModelFamily, LiteNgramResidual, LiteSpec, LiteTrack};

const METAL_SOURCE: &str = include_str!("../kernels/pg_lite_ngram.metal");
const THREADS: u64 = 256;
const BIGRAM_ROWS: usize = 257;
const BYTE_VALUES: usize = 256;
const MAX_REDUCE_GROUPS: u64 = 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct PgLiteMetalEvalConfig {
    token_count: u32,
    residual_buckets: u32,
    context: u32,
    residual_weight: f32,
}

pub(super) struct MetalLiteTrainEval {
    pub model: LiteNgramResidual,
    pub eval: LiteEvalStats,
}

pub(super) fn runtime_probe() -> Result<String, String> {
    let runtime = MetalLiteRuntime::new()?;
    Ok(format!(
        "system_default_device; kernels={}",
        runtime.pipeline_count()
    ))
}

pub(super) fn train_and_eval_fixed(
    train: &[u8],
    val: &[u8],
    spec: &LiteSpec,
) -> PgResult<MetalLiteTrainEval> {
    if spec.track != LiteTrack::ByteGolf {
        return Err(PgError::InvalidOp(
            "PG-Lite Metal train/eval is only valid for byte_golf; use eval_fixed_model or score_first_eval_model for other tracks"
                .into(),
        ));
    }
    let runtime = metal_runtime()?;
    runtime.train_and_eval_fixed(train, val, spec)
}

pub(super) fn eval_fixed_model(model: &LiteNgramResidual, val: &[u8]) -> PgResult<LiteEvalStats> {
    let runtime = metal_runtime()?;
    runtime.eval_fixed_model(model, val)
}

pub(super) fn score_first_eval_model(
    model: &LiteNgramResidual,
    val: &[u8],
) -> PgResult<LiteEvalStats> {
    let runtime = metal_runtime()?;
    runtime.score_first_eval_model(model, val)
}

fn metal_runtime() -> PgResult<MetalLiteRuntime> {
    MetalLiteRuntime::new().map_err(|err| {
        PgError::InvalidOp(format!(
            "PG-Lite Metal runtime is not executable on this host/build: {err}"
        ))
    })
}

struct MetalLiteRuntime {
    device: Device,
    queue: CommandQueue,
    init_bigram: ComputePipelineState,
    init_residual_u32: ComputePipelineState,
    train: ComputePipelineState,
    downcast_residual: ComputePipelineState,
    bigram_sums: ComputePipelineState,
    residual_sums_u16: ComputePipelineState,
    loss_reduce_u16: ComputePipelineState,
    score_first_u32: ComputePipelineState,
}

impl MetalLiteRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal system default device is unavailable".to_string())?;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = device
            .new_library_with_source(METAL_SOURCE, &options)
            .map_err(|err| format!("Metal source compile failed: {err}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            init_bigram: pipeline(&device, &library, "pg_lite_init_bigram_counts_kernel")?,
            init_residual_u32: pipeline(
                &device,
                &library,
                "pg_lite_init_residual_counts_u32_kernel",
            )?,
            train: pipeline(&device, &library, "pg_lite_ngram_train_kernel")?,
            downcast_residual: pipeline(
                &device,
                &library,
                "pg_lite_downcast_residual_u32_to_u16_kernel",
            )?,
            bigram_sums: pipeline(&device, &library, "pg_lite_bigram_row_sums_kernel")?,
            residual_sums_u16: pipeline(&device, &library, "pg_lite_residual_row_sums_u16_kernel")?,
            loss_reduce_u16: pipeline(
                &device,
                &library,
                "pg_lite_ngram_loss_reduce_u16_presummed_kernel",
            )?,
            score_first_u32: pipeline(&device, &library, "pg_lite_score_first_eval_u32_kernel")?,
            device,
            queue,
        })
    }

    fn pipeline_count(&self) -> usize {
        8
    }

    fn train_and_eval_fixed(
        &self,
        train: &[u8],
        val: &[u8],
        spec: &LiteSpec,
    ) -> PgResult<MetalLiteTrainEval> {
        let residual_buckets = spec.model.residual_buckets.max(1);
        let residual_weight = match spec.model.family {
            LiteModelFamily::ByteNgram | LiteModelFamily::ArtifactOnly => 0.0,
            LiteModelFamily::NgramResidual => spec.model.residual_weight.clamp(0.0, 0.95),
        };
        let train_cfg = config(
            train.len(),
            residual_buckets,
            spec.model.context,
            residual_weight,
        )?;
        let eval_cfg = config(
            val.len(),
            residual_buckets,
            spec.model.context,
            residual_weight,
        )?;
        let bigram_len = BIGRAM_ROWS * BYTE_VALUES;
        let residual_len = residual_buckets
            .checked_mul(BYTE_VALUES)
            .ok_or_else(|| PgError::InvalidOp("PG-Lite residual count table overflow".into()))?;
        let bigram_counts = self.empty_u32_buffer(bigram_len)?;
        let residual_u32 = self.empty_u32_buffer(residual_len)?;
        let residual_u16 = self.empty_u16_buffer(residual_len)?;
        let train_bytes = self.bytes_buffer(train)?;
        let val_bytes = self.bytes_buffer(val)?;
        let bigram_sums = self.empty_u32_buffer(BIGRAM_ROWS)?;
        let residual_sums = self.empty_u32_buffer(residual_buckets)?;
        let reduce_groups = reduce_group_count(val.len());
        let partial_sums = self.empty_f32_buffer(reduce_groups as usize)?;

        let residual_buckets_u32 = checked_u32(residual_buckets, "residual_buckets")?;
        let residual_len_u32 = checked_u32(residual_len, "residual_entries")?;
        self.submit(|encoder| {
            encode_init_bigram(
                encoder,
                &self.init_bigram,
                &bigram_counts,
                bigram_len as u64,
            );
            encode_init_residual_u32(
                encoder,
                &self.init_residual_u32,
                &residual_u32,
                residual_buckets_u32,
                residual_len as u64,
            );
            if !train.is_empty() {
                encode_train(
                    encoder,
                    &self.train,
                    &train_bytes,
                    &bigram_counts,
                    &residual_u32,
                    train_cfg,
                    train.len() as u64,
                );
            }
            encode_downcast_residual(
                encoder,
                &self.downcast_residual,
                &residual_u32,
                &residual_u16,
                residual_len_u32,
                residual_len as u64,
            );
            encode_bigram_sums(encoder, &self.bigram_sums, &bigram_counts, &bigram_sums);
            encode_residual_sums_u16(
                encoder,
                &self.residual_sums_u16,
                &residual_u16,
                &residual_sums,
                residual_buckets_u32,
                residual_buckets as u64,
            );
            if !val.is_empty() {
                encode_loss_reduce_u16(
                    encoder,
                    &self.loss_reduce_u16,
                    &val_bytes,
                    &bigram_counts,
                    &bigram_sums,
                    &residual_u16,
                    &residual_sums,
                    eval_cfg,
                    &partial_sums,
                    reduce_groups,
                );
            }
        })?;

        let bigram_flat = read_buffer::<u32>(&bigram_counts, bigram_len);
        let residual_flat = read_buffer::<u16>(&residual_u16, residual_len);
        let model = model_from_flat_counts(
            bigram_flat,
            residual_flat,
            residual_buckets,
            residual_weight,
            spec.model.context,
            spec.model.family,
        )?;
        let eval = if val.is_empty() {
            LiteEvalStats {
                loss: 0.0,
                tokens: 0,
                score_first_tokens_scored: 0,
                score_first_update_count: 0,
            }
        } else {
            let partial = read_buffer::<f32>(&partial_sums, reduce_groups as usize);
            LiteEvalStats {
                loss: partial.iter().map(|&v| v as f64).sum(),
                tokens: val.len(),
                score_first_tokens_scored: 0,
                score_first_update_count: 0,
            }
        };
        Ok(MetalLiteTrainEval { model, eval })
    }

    fn eval_fixed_model(&self, model: &LiteNgramResidual, val: &[u8]) -> PgResult<LiteEvalStats> {
        if val.is_empty() {
            return Ok(LiteEvalStats {
                loss: 0.0,
                tokens: 0,
                score_first_tokens_scored: 0,
                score_first_update_count: 0,
            });
        }
        let residual_buckets = model.residual_counts.len().max(1);
        let eval_cfg = config(
            val.len(),
            residual_buckets,
            model.context,
            model.residual_weight,
        )?;
        let bigram_flat = flatten_u32_rows(&model.bigram_counts);
        let residual_flat = flatten_u16_rows(&model.residual_counts);
        let bigram_counts = self.u32_buffer(&bigram_flat)?;
        let residual_counts = self.u16_buffer(&residual_flat)?;
        let val_bytes = self.bytes_buffer(val)?;
        let bigram_sums = self.empty_u32_buffer(BIGRAM_ROWS)?;
        let residual_sums = self.empty_u32_buffer(residual_buckets)?;
        let reduce_groups = reduce_group_count(val.len());
        let partial_sums = self.empty_f32_buffer(reduce_groups as usize)?;
        let residual_buckets_u32 = checked_u32(residual_buckets, "residual_buckets")?;
        self.submit(|encoder| {
            encode_bigram_sums(encoder, &self.bigram_sums, &bigram_counts, &bigram_sums);
            encode_residual_sums_u16(
                encoder,
                &self.residual_sums_u16,
                &residual_counts,
                &residual_sums,
                residual_buckets_u32,
                residual_buckets as u64,
            );
            if !val.is_empty() {
                encode_loss_reduce_u16(
                    encoder,
                    &self.loss_reduce_u16,
                    &val_bytes,
                    &bigram_counts,
                    &bigram_sums,
                    &residual_counts,
                    &residual_sums,
                    eval_cfg,
                    &partial_sums,
                    reduce_groups,
                );
            }
        })?;
        let partial = read_buffer::<f32>(&partial_sums, reduce_groups as usize);
        Ok(LiteEvalStats {
            loss: partial.iter().map(|&v| v as f64).sum(),
            tokens: val.len(),
            score_first_tokens_scored: 0,
            score_first_update_count: 0,
        })
    }

    fn score_first_eval_model(
        &self,
        model: &LiteNgramResidual,
        val: &[u8],
    ) -> PgResult<LiteEvalStats> {
        if val.is_empty() {
            return Ok(LiteEvalStats {
                loss: 0.0,
                tokens: 0,
                score_first_tokens_scored: 0,
                score_first_update_count: 0,
            });
        }
        let residual_buckets = model.residual_counts.len().max(1);
        let eval_cfg = config(
            val.len(),
            residual_buckets,
            model.context,
            model.residual_weight,
        )?;
        let bigram_flat = flatten_u32_rows(&model.bigram_counts);
        let residual_flat = flatten_u16_rows_to_u32(&model.residual_counts);
        let bigram_counts = self.u32_buffer(&bigram_flat)?;
        let residual_counts = self.u32_buffer(&residual_flat)?;
        let val_bytes = self.bytes_buffer(val)?;
        let loss_out = self.empty_f32_buffer(1)?;
        self.submit(|encoder| {
            if !val.is_empty() {
                encoder.set_compute_pipeline_state(&self.score_first_u32);
                encoder.set_buffer(0, Some(&val_bytes), 0);
                encoder.set_buffer(1, Some(&bigram_counts), 0);
                encoder.set_buffer(2, Some(&residual_counts), 0);
                set_bytes(encoder, 3, &eval_cfg);
                encoder.set_buffer(4, Some(&loss_out), 0);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            }
        })?;
        let loss = read_buffer::<f32>(&loss_out, 1)
            .first()
            .copied()
            .unwrap_or(0.0) as f64;
        Ok(LiteEvalStats {
            loss,
            tokens: val.len(),
            score_first_tokens_scored: val.len(),
            score_first_update_count: val.len(),
        })
    }

    fn submit<F>(&self, encode: F) -> PgResult<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(PgError::InvalidOp(
                "PG-Lite Metal command buffer completed with error".into(),
            ));
        }
        Ok(())
    }

    fn empty_u32_buffer(&self, len: usize) -> PgResult<Buffer> {
        self.empty_buffer::<u32>(len)
    }

    fn empty_u16_buffer(&self, len: usize) -> PgResult<Buffer> {
        self.empty_buffer::<u16>(len)
    }

    fn empty_f32_buffer(&self, len: usize) -> PgResult<Buffer> {
        self.empty_buffer::<f32>(len.max(1))
    }

    fn empty_buffer<T>(&self, len: usize) -> PgResult<Buffer> {
        let bytes = checked_buffer_bytes::<T>(len)?;
        Ok(self.device.new_buffer(bytes as u64, shared_options()))
    }

    fn bytes_buffer(&self, bytes: &[u8]) -> PgResult<Buffer> {
        if bytes.is_empty() {
            self.empty_buffer::<u8>(1)
        } else {
            Ok(self.device.new_buffer_with_data(
                bytes.as_ptr() as *const c_void,
                bytes.len() as u64,
                shared_options(),
            ))
        }
    }

    fn u32_buffer(&self, values: &[u32]) -> PgResult<Buffer> {
        typed_data_buffer(&self.device, values)
    }

    fn u16_buffer(&self, values: &[u16]) -> PgResult<Buffer> {
        typed_data_buffer(&self.device, values)
    }
}

fn pipeline(
    device: &Device,
    library: &metal::Library,
    symbol: &str,
) -> Result<ComputePipelineState, String> {
    let function = library.get_function(symbol, None)?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| format!("pipeline build failed for {symbol}: {err}"))
}

fn encode_init_bigram(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    bigram_counts: &Buffer,
    count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(bigram_counts), 0);
    dispatch_1d(encoder, count);
}

fn encode_init_residual_u32(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    residual_counts: &Buffer,
    residual_buckets: u32,
    count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(residual_counts), 0);
    set_bytes(encoder, 1, &residual_buckets);
    dispatch_1d(encoder, count);
}

fn encode_train(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    bytes: &Buffer,
    bigram_counts: &Buffer,
    residual_counts: &Buffer,
    cfg: PgLiteMetalEvalConfig,
    token_count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(bytes), 0);
    encoder.set_buffer(1, Some(bigram_counts), 0);
    encoder.set_buffer(2, Some(residual_counts), 0);
    set_bytes(encoder, 3, &cfg);
    dispatch_1d(encoder, token_count);
}

fn encode_downcast_residual(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    residual_u32: &Buffer,
    residual_u16: &Buffer,
    residual_entries: u32,
    count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(residual_u32), 0);
    encoder.set_buffer(1, Some(residual_u16), 0);
    set_bytes(encoder, 2, &residual_entries);
    dispatch_1d(encoder, count);
}

fn encode_bigram_sums(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    bigram_counts: &Buffer,
    bigram_sums: &Buffer,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(bigram_counts), 0);
    encoder.set_buffer(1, Some(bigram_sums), 0);
    dispatch_1d(encoder, BIGRAM_ROWS as u64);
}

fn encode_residual_sums_u16(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    residual_counts: &Buffer,
    residual_sums: &Buffer,
    residual_buckets: u32,
    count: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(residual_counts), 0);
    encoder.set_buffer(1, Some(residual_sums), 0);
    set_bytes(encoder, 2, &residual_buckets);
    dispatch_1d(encoder, count);
}

#[allow(clippy::too_many_arguments)]
fn encode_loss_reduce_u16(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    bytes: &Buffer,
    bigram_counts: &Buffer,
    bigram_sums: &Buffer,
    residual_counts: &Buffer,
    residual_sums: &Buffer,
    cfg: PgLiteMetalEvalConfig,
    partial_sums: &Buffer,
    reduce_groups: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(bytes), 0);
    encoder.set_buffer(1, Some(bigram_counts), 0);
    encoder.set_buffer(2, Some(bigram_sums), 0);
    encoder.set_buffer(3, Some(residual_counts), 0);
    encoder.set_buffer(4, Some(residual_sums), 0);
    set_bytes(encoder, 5, &cfg);
    encoder.set_buffer(6, Some(partial_sums), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(reduce_groups.max(1), 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
}

fn dispatch_1d(encoder: &ComputeCommandEncoderRef, count: u64) {
    if count == 0 {
        return;
    }
    encoder.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(THREADS, 1, 1));
}

fn set_bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}

fn typed_data_buffer<T>(device: &Device, values: &[T]) -> PgResult<Buffer> {
    if values.is_empty() {
        return Ok(device.new_buffer(mem::size_of::<T>() as u64, shared_options()));
    }
    let bytes = checked_buffer_bytes::<T>(values.len())?;
    Ok(device.new_buffer_with_data(
        values.as_ptr() as *const c_void,
        bytes as u64,
        shared_options(),
    ))
}

fn checked_buffer_bytes<T>(len: usize) -> PgResult<usize> {
    len.checked_mul(mem::size_of::<T>())
        .ok_or_else(|| PgError::InvalidOp("PG-Lite Metal buffer size overflow".into()))
}

fn shared_options() -> MTLResourceOptions {
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache
}

fn config(
    token_count: usize,
    residual_buckets: usize,
    context: usize,
    residual_weight: f64,
) -> PgResult<PgLiteMetalEvalConfig> {
    Ok(PgLiteMetalEvalConfig {
        token_count: checked_u32(token_count, "token_count")?,
        residual_buckets: checked_u32(residual_buckets, "residual_buckets")?,
        context: checked_u32(context, "context")?,
        residual_weight: residual_weight.clamp(0.0, 0.95) as f32,
    })
}

fn checked_u32(value: usize, label: &str) -> PgResult<u32> {
    u32::try_from(value).map_err(|_| {
        PgError::InvalidOp(format!(
            "PG-Lite Metal {label}={value} does not fit u32 kernel ABI"
        ))
    })
}

fn reduce_group_count(tokens: usize) -> u64 {
    if tokens == 0 {
        return 1;
    }
    let groups = (tokens as u64).div_ceil(THREADS);
    groups.clamp(1, MAX_REDUCE_GROUPS)
}

fn flatten_u32_rows(rows: &[[u32; BYTE_VALUES]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(rows.len() * BYTE_VALUES);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn flatten_u16_rows(rows: &[[u16; BYTE_VALUES]]) -> Vec<u16> {
    let mut out = Vec::with_capacity(rows.len() * BYTE_VALUES);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn flatten_u16_rows_to_u32(rows: &[[u16; BYTE_VALUES]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(rows.len() * BYTE_VALUES);
    for row in rows {
        out.extend(row.iter().map(|&value| value as u32));
    }
    out
}

fn model_from_flat_counts(
    bigram_flat: Vec<u32>,
    residual_flat: Vec<u16>,
    residual_buckets: usize,
    residual_weight: f64,
    context: usize,
    family: LiteModelFamily,
) -> PgResult<LiteNgramResidual> {
    if bigram_flat.len() != BIGRAM_ROWS * BYTE_VALUES {
        return Err(PgError::DataFormat(
            "PG-Lite Metal bigram readback has invalid length".into(),
        ));
    }
    if residual_flat.len() != residual_buckets * BYTE_VALUES {
        return Err(PgError::DataFormat(
            "PG-Lite Metal residual readback has invalid length".into(),
        ));
    }
    let mut bigram_counts = vec![[0u32; BYTE_VALUES]; BIGRAM_ROWS];
    for (row_idx, row) in bigram_counts.iter_mut().enumerate() {
        let start = row_idx * BYTE_VALUES;
        row.copy_from_slice(&bigram_flat[start..start + BYTE_VALUES]);
    }
    let mut residual_counts = vec![[0u16; BYTE_VALUES]; residual_buckets];
    for (row_idx, row) in residual_counts.iter_mut().enumerate() {
        let start = row_idx * BYTE_VALUES;
        row.copy_from_slice(&residual_flat[start..start + BYTE_VALUES]);
    }
    Ok(LiteNgramResidual {
        bigram_counts,
        residual_counts,
        residual_weight,
        context,
        family,
    })
}

fn read_buffer<T: Copy>(buffer: &Buffer, len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    unsafe {
        let ptr = buffer.contents() as *const T;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}
