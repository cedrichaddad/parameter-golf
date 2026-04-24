use std::time::Instant;

use pg_model::backward::GradBuffers;
use pg_model::{ExecutionPlan, ForwardBuffer, GptModel, RunMode, RunSpec, TrainBackend};
use pg_optim::adamw::{AdamW, AdamWState};
use pg_optim::ema::{Ema, Swa};
#[cfg(feature = "cuda")]
use pg_optim::gpu::{AdamWHyper, GpuAdamWState, GpuMuon, GpuOptimizer};
use pg_optim::muon::Muon;
use pg_optim::scheduler;

use pg_core::PgResult;
use pg_data::bpb::{BpbLuts, compute_bpb};
use pg_data::token_stream::DistributedTokenLoader;

#[derive(Debug, Clone)]
pub struct VariantResult {
    pub run_name: String,
    pub mode: RunMode,
    pub train_backend: TrainBackend,
    pub variant_fingerprint: String,
    pub steps_completed: usize,
    pub train_loss: f32,
    pub proxy_bpb: Option<f64>,
    pub eval_loss: Option<f64>,
    pub final_bpb: Option<f64>,
    pub eval_tokens: Option<usize>,
    pub artifact_bytes: Option<usize>,
    pub artifact_budget_ok: Option<bool>,
    pub bank_update_backend: String,
    pub train_data_source: String,
    pub bpb_byte_source: String,
    pub proxy_metric_source: Option<String>,
    pub ms_per_step: f64,
    pub wallclock_seconds: f64,
    pub rank: usize,
    pub world_size: usize,
    pub distributed_sync: bool,
}

pub struct VariantRunner {
    pub run_spec: RunSpec,
    pub plan: ExecutionPlan,
}

#[cfg(feature = "cuda")]
struct CudaSingleHybridRuntime {
    gpu_model: pg_model::gpu::GpuModel,
    backward_state: pg_model::gpu::GpuBackwardState,
    input_ids: pg_core::GpuTensor,
    targets: pg_core::GpuTensor,
}

#[cfg(feature = "cuda")]
impl CudaSingleHybridRuntime {
    fn new(model: &GptModel, plan: &ExecutionPlan, tokens: usize) -> PgResult<Self> {
        let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda context init failed: {:?}", e))
        })?;
        let stream = ctx.default_stream();
        let gpu_model =
            pg_model::gpu::GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())?;
        Ok(Self {
            backward_state: pg_model::gpu::GpuBackwardState::new_for_plan(
                plan,
                tokens,
                stream.clone(),
            )?,
            gpu_model,
            input_ids: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens],
                pg_core::DType::U32,
            )?,
            targets: pg_core::GpuTensor::zeros_gpu(stream, &[tokens], pg_core::DType::U32)?,
        })
    }
}

#[cfg(feature = "cuda")]
struct CudaSingleFastRuntime {
    gpu_model: pg_model::gpu::GpuModel,
    backward_state: pg_model::gpu::GpuBackwardState,
    input_ids: pg_core::GpuTensor,
    targets: pg_core::GpuTensor,
    gpu_buf: pg_model::gpu::GpuActivations,
    gpu_grads: pg_model::gpu::GpuGradBuffers,
    gpu_optimizer: GpuOptimizer,
    state_tok_emb: GpuAdamWState,
    state_bigram_embed: GpuAdamWState,
    state_bigram_proj: GpuAdamWState,
    state_smear_gate: GpuAdamWState,
    state_skip_weights: GpuAdamWState,
    state_ve_embed: GpuAdamWState,
    state_ve_proj: GpuAdamWState,
    state_ve_layer_scales: GpuAdamWState,
    state_attn_scale: Vec<GpuAdamWState>,
    state_mlp_scale: Vec<GpuAdamWState>,
    state_resid_mix: Vec<GpuAdamWState>,
    state_q_gain: Vec<GpuAdamWState>,
    state_bigram_scale: AdamWState,
    state_ve_scale: AdamWState,
    grad_norm_scratch: pg_core::GpuTensor,
    gpu_muon: GpuMuon,
}

#[cfg(feature = "cuda")]
impl CudaSingleFastRuntime {
    fn new(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
    ) -> PgResult<Self> {
        Self::new_on_device(model, plan, tokens, train_config, 0)
    }

    fn new_on_device(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
        device_ordinal: usize,
    ) -> PgResult<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_ordinal).map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda context init failed: {:?}", e))
        })?;
        let stream = ctx.default_stream();
        let gpu_model =
            pg_model::gpu::GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())?;
        let gpu_buf = pg_model::gpu::GpuActivations::new_for_plan(plan, tokens, stream.clone())?;
        let gpu_grads = pg_model::gpu::GpuGradBuffers::new(&model.config, stream.clone())?;
        let n = model.config.num_layers;
        let d = model.config.model_dim;
        let kv = model.config.kv_dim();
        let mlp = model.config.mlp_dim;
        let bank_shapes = vec![[2 * n, d, d], [2 * n, kv, d], [n, mlp, d], [n, d, mlp]];

        Ok(Self {
            backward_state: pg_model::gpu::GpuBackwardState::new_for_plan(
                plan,
                tokens,
                stream.clone(),
            )?,
            input_ids: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens],
                pg_core::DType::U32,
            )?,
            targets: pg_core::GpuTensor::zeros_gpu(stream.clone(), &[tokens], pg_core::DType::U32)?,
            state_tok_emb: GpuAdamWState::new_like(&gpu_model.weights.tok_emb, stream.clone())?,
            state_bigram_embed: GpuAdamWState::new_like(
                &gpu_model.weights.bigram_embed,
                stream.clone(),
            )?,
            state_bigram_proj: GpuAdamWState::new_like(
                &gpu_model.weights.bigram_proj,
                stream.clone(),
            )?,
            state_smear_gate: GpuAdamWState::new_like(
                &gpu_model.weights.smear_gate,
                stream.clone(),
            )?,
            state_skip_weights: GpuAdamWState::new_like(
                &gpu_model.weights.skip_weights,
                stream.clone(),
            )?,
            state_ve_embed: GpuAdamWState::new_like(&gpu_model.weights.ve_embed, stream.clone())?,
            state_ve_proj: GpuAdamWState::new_like(&gpu_model.weights.ve_proj, stream.clone())?,
            state_ve_layer_scales: GpuAdamWState::new_like(
                &gpu_model.weights.ve_layer_scales,
                stream.clone(),
            )?,
            state_attn_scale: gpu_model
                .weights
                .attn_scales
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_mlp_scale: gpu_model
                .weights
                .mlp_scales
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_resid_mix: gpu_model
                .weights
                .resid_mix
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_q_gain: gpu_model
                .weights
                .q_gains
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_bigram_scale: AdamWState::new(1),
            state_ve_scale: AdamWState::new(1),
            grad_norm_scratch: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[1],
                pg_core::DType::F32,
            )?,
            gpu_muon: GpuMuon::new(
                stream,
                train_config.matrix_lr,
                train_config.muon_momentum,
                train_config.newton_schulz_steps,
                true,
                train_config.muon_wd,
                &bank_shapes,
            )?,
            gpu_model,
            gpu_buf,
            gpu_grads,
            gpu_optimizer: GpuOptimizer::new(),
        })
    }
}

#[cfg(feature = "cuda")]
struct CudaDistributedRuntime {
    replicas: Vec<CudaSingleFastRuntime>,
    comms: Vec<pg_core::nccl::NcclComm>,
    distributed_sync: bool,
}

#[cfg(feature = "cuda")]
impl CudaDistributedRuntime {
    fn new(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
        world_size: usize,
    ) -> PgResult<Self> {
        let device_count = cudarc::driver::CudaContext::device_count().map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda device_count failed: {:?}", e))
        })? as usize;
        if world_size > device_count {
            return Err(pg_core::PgError::InvalidOp(format!(
                "cuda-distributed requested world_size {} but only {} local CUDA devices are visible",
                world_size, device_count
            )));
        }
        let replicas = (0..world_size)
            .map(|ordinal| {
                CudaSingleFastRuntime::new_on_device(model, plan, tokens, train_config, ordinal)
            })
            .collect::<PgResult<Vec<_>>>()?;
        let streams = replicas
            .iter()
            .map(|runtime| runtime.gpu_model.gemm.stream().clone())
            .collect::<Vec<_>>();
        let comms = pg_core::nccl::NcclComm::from_local_devices(streams)?;
        Ok(Self {
            replicas,
            comms,
            distributed_sync: false,
        })
    }
}

impl VariantRunner {
    pub fn new(run_spec: RunSpec) -> PgResult<Self> {
        let plan = ExecutionPlan::from_run_spec(&run_spec)?;
        Ok(Self { run_spec, plan })
    }

    pub fn run(&self, mode: RunMode) -> PgResult<VariantResult> {
        let model_config = self.run_spec.model.to_model_config();
        let train_config = self.run_spec.train.to_train_config();
        validate_backend_request(&self.run_spec, mode)?;
        validate_executable_variant(&self.run_spec, mode)?;
        let mut model = GptModel::new(model_config.clone());
        model.fill_deterministic();
        let mut active_tokens = match mode {
            RunMode::Smoke => self
                .run_spec
                .train
                .seq_len
                .min(16)
                .min(model_config.train_seq_len),
            RunMode::Proxy => self
                .run_spec
                .train
                .seq_len
                .min(64)
                .min(model_config.train_seq_len),
            RunMode::Record => self.run_spec.train.seq_len.min(model_config.train_seq_len),
        };
        let world_size = self.run_spec.train.world_size.max(1);
        let rank = self.run_spec.train.rank;
        if rank >= world_size {
            return Err(pg_core::PgError::InvalidOp(format!(
                "rank {} must be < world_size {}",
                rank, world_size
            )));
        }
        if world_size > 1
            && !matches!(mode, RunMode::Smoke)
            && self.run_spec.train.backend != TrainBackend::CudaDistributed
        {
            return Err(pg_core::PgError::InvalidOp(
                "multi-rank proxy/record training requires backend=cuda-distributed; CPU only supports distributed smoke/data-shard preflight".into(),
            ));
        }
        if world_size > 1 {
            active_tokens = (active_tokens / world_size).max(1);
        }
        let mut buf = ForwardBuffer::new(&model_config, active_tokens);
        let mut grads = GradBuffers::new(&model_config);
        let mut data_loader = if self.run_spec.train.backend == TrainBackend::CudaDistributed {
            None
        } else if let Some(pattern) = self.run_spec.train.train_data_pattern.as_deref() {
            Some(DistributedTokenLoader::new(pattern, rank, world_size)?)
        } else {
            None
        };
        let mut distributed_data_loaders = if self.run_spec.train.backend
            == TrainBackend::CudaDistributed
        {
            if let Some(pattern) = self.run_spec.train.train_data_pattern.as_deref() {
                Some(
                    (0..world_size)
                        .map(|rank_idx| DistributedTokenLoader::new(pattern, rank_idx, world_size))
                        .collect::<PgResult<Vec<_>>>()?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let train_data_source = if data_loader.is_some() {
            "shards"
        } else if self.run_spec.train.backend == TrainBackend::CudaDistributed
            && self.run_spec.train.train_data_pattern.is_some()
        {
            "shards"
        } else {
            "synthetic_sequence"
        };

        let n = model_config.num_layers;
        let d = model_config.model_dim;
        let kv = model_config.kv_dim();
        let mlp = model_config.mlp_dim;
        let bank_shapes: Vec<[usize; 3]> =
            vec![[2 * n, d, d], [2 * n, kv, d], [n, mlp, d], [n, d, mlp]];

        let mut muon = Muon::new(
            train_config.matrix_lr,
            train_config.muon_momentum,
            train_config.newton_schulz_steps,
            true,
            train_config.muon_wd,
            &bank_shapes,
        );
        let mut adamw_embed = AdamW::new(
            train_config.embed_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );
        let mut adamw_scalar = AdamW::new(
            train_config.scalar_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );

        let mut state_tok_emb = AdamWState::new(model.tok_emb.len());
        let mut state_bigram_embed = AdamWState::new(model.bigram_embed.len());
        let mut state_bigram_proj = AdamWState::new(model.bigram_proj.len());
        let mut state_smear_gate = AdamWState::new(model.smear_gate.len());
        let mut state_skip_weights = AdamWState::new(model.skip_weights.len());
        let mut state_ve_embed = AdamWState::new(model.ve_embed.len());
        let mut state_ve_proj = AdamWState::new(model.ve_proj.len());
        let mut state_ve_scale = AdamWState::new(1);
        let mut state_ve_layer_scales = AdamWState::new(model.ve_layer_scales.len());
        let mut state_bigram_scale = AdamWState::new(1);
        let mut state_attn_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_mlp_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_resid_mix: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(2 * d)).collect();
        let mut state_q_gain: Vec<AdamWState> = (0..n)
            .map(|_| AdamWState::new(model_config.num_heads))
            .collect();

        let total_params = count_params(&model);
        let mut ema = Ema::new(train_config.ema_decay, total_params);
        let mut swa = Swa::new(total_params);
        let mut flat_buf = vec![0.0f32; total_params];

        let requested_steps = match mode {
            RunMode::Smoke => 4usize,
            RunMode::Proxy => 32usize,
            RunMode::Record => train_config.total_iterations,
        };
        let max_steps = requested_steps.min(train_config.total_iterations);
        let fast_bank_updates =
            matches!(mode, RunMode::Smoke) || self.run_spec.train.fast_bank_updates;
        let bank_update_backend = match self.run_spec.train.backend {
            TrainBackend::Cpu => {
                if fast_bank_updates {
                    "fast_sgd"
                } else {
                    "muon_ns5"
                }
            }
            TrainBackend::CudaSingle | TrainBackend::CudaSingleParity => {
                if self.run_spec.train.backend == TrainBackend::CudaSingleParity {
                    if fast_bank_updates {
                        "cpu_fast_sgd_mirror"
                    } else {
                        "cpu_muon_ns5_mirror"
                    }
                } else {
                    "gpu_muon_ns5"
                }
            }
            TrainBackend::CudaDistributed => "nccl_allreduce_gpu_muon_ns5",
        };
        #[cfg(feature = "cuda")]
        let mut cuda_single_parity_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaSingleParity {
                Some(CudaSingleHybridRuntime::new(
                    &model, &self.plan, buf.tokens,
                )?)
            } else {
                None
            };
        #[cfg(feature = "cuda")]
        let mut cuda_single_fast_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaSingle {
                Some(CudaSingleFastRuntime::new(
                    &model,
                    &self.plan,
                    buf.tokens,
                    &train_config,
                )?)
            } else {
                None
            };
        #[cfg(feature = "cuda")]
        let mut cuda_distributed_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaDistributed {
                Some(CudaDistributedRuntime::new(
                    &model,
                    &self.plan,
                    buf.tokens,
                    &train_config,
                    world_size,
                )?)
            } else {
                None
            };

        let start = Instant::now();
        let mut final_loss = 0.0f32;
        let mut steps_completed = 0usize;
        let mut last_input_ids = Vec::new();
        let mut last_targets = Vec::new();
        for step in 0..max_steps {
            let elapsed = start.elapsed().as_secs_f32();
            if elapsed > train_config.max_wallclock_seconds {
                break;
            }
            let lr_scale = scheduler::lr_scale(
                step,
                train_config.warmup_steps,
                train_config.total_iterations,
                train_config.warmdown_iters,
            );
            muon.lr = train_config.matrix_lr * lr_scale;
            muon.momentum = train_config.muon_momentum_at(step);
            adamw_embed.lr = train_config.embed_lr * lr_scale;
            adamw_scalar.lr = train_config.scalar_lr * lr_scale;

            let distributed_batches: Option<Vec<(Vec<u32>, Vec<u32>)>> = if self
                .run_spec
                .train
                .backend
                == TrainBackend::CudaDistributed
            {
                if let Some(loaders) = distributed_data_loaders.as_mut() {
                    let global_tokens = buf.tokens * world_size;
                    Some(
                        loaders
                            .iter_mut()
                            .map(|loader| {
                                let (x, y) = loader.next_batch(global_tokens, buf.tokens)?;
                                Ok::<_, pg_core::PgError>((
                                    x.into_iter().map(|v| v as u32).collect(),
                                    y.into_iter().map(|v| v as u32).collect(),
                                ))
                            })
                            .collect::<PgResult<Vec<_>>>()?,
                    )
                } else {
                    Some(
                        (0..world_size)
                            .map(|rank_idx| {
                                let offset = rank_idx * buf.tokens + step * buf.tokens * world_size;
                                (
                                    (0..buf.tokens)
                                        .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                                        .collect::<Vec<_>>(),
                                    (1..=buf.tokens)
                                        .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect(),
                    )
                }
            } else {
                None
            };
            let (input_ids, targets): (Vec<u32>, Vec<u32>) =
                if let Some(batches) = distributed_batches.as_ref() {
                    (batches[0].0.clone(), batches[0].1.clone())
                } else if let Some(loader) = data_loader.as_mut() {
                    let global_tokens = buf.tokens * world_size;
                    let (x, y) = loader.next_batch(global_tokens, buf.tokens)?;
                    (
                        x.into_iter().map(|v| v as u32).collect(),
                        y.into_iter().map(|v| v as u32).collect(),
                    )
                } else {
                    let offset = rank * buf.tokens + step * buf.tokens * world_size;
                    (
                        (0..buf.tokens)
                            .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                            .collect(),
                        (1..=buf.tokens)
                            .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                            .collect(),
                    )
                };
            if let Some(batches) = distributed_batches.as_ref() {
                last_input_ids = batches
                    .iter()
                    .flat_map(|(x, _)| x.iter().copied())
                    .collect();
                last_targets = batches
                    .iter()
                    .flat_map(|(_, y)| y.iter().copied())
                    .collect();
            } else {
                last_input_ids = input_ids.clone();
                last_targets = targets.clone();
            }

            grads.zero();
            final_loss = match self.run_spec.train.backend {
                TrainBackend::Cpu => model.backward(&input_ids, &targets, &mut buf, &mut grads),
                #[cfg(feature = "cuda")]
                TrainBackend::CudaSingleParity => {
                    let runtime = cuda_single_parity_runtime
                        .as_mut()
                        .expect("cuda single parity runtime must be initialized");
                    cuda_single_hybrid_step(
                        runtime, &model, &self.plan, &input_ids, &targets, &mut grads,
                    )?
                }
                #[cfg(feature = "cuda")]
                TrainBackend::CudaSingle => {
                    let runtime = cuda_single_fast_runtime
                        .as_mut()
                        .expect("cuda single fast runtime must be initialized");
                    cuda_single_fast_step(
                        runtime,
                        &mut model,
                        &self.plan,
                        &input_ids,
                        &targets,
                        &train_config,
                        step,
                        lr_scale,
                    )?
                }
                #[cfg(feature = "cuda")]
                TrainBackend::CudaDistributed => {
                    let runtime = cuda_distributed_runtime
                        .as_mut()
                        .expect("cuda distributed runtime must be initialized");
                    cuda_distributed_step(
                        runtime,
                        distributed_batches
                            .as_ref()
                            .expect("distributed backend must materialize per-rank batches"),
                        &train_config,
                        step,
                        lr_scale,
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                _ => unreachable!("backend should have been rejected before run"),
            };
            if !matches!(
                self.run_spec.train.backend,
                TrainBackend::CudaSingle | TrainBackend::CudaDistributed
            ) {
                grads.clip_grad_norm(train_config.grad_clip_norm);

                if fast_bank_updates {
                    // Smoke mode is a correctness/liveness gate. Full CPU NS5 over 26M+
                    // parameters is too slow for that path, so use the same gradients
                    // with a cheap bank update and reserve Muon for proxy/record runs.
                    smoke_bank_step(
                        &mut model.qo_bank,
                        &grads.qo_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.kv_bank,
                        &grads.kv_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.mlp_up_bank,
                        &grads.mlp_up_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.mlp_down_bank,
                        &grads.mlp_down_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                } else {
                    muon.step_bank(0, &mut model.qo_bank, &grads.qo_bank, &bank_shapes[0]);
                    muon.step_bank(1, &mut model.kv_bank, &grads.kv_bank, &bank_shapes[1]);
                    muon.step_bank(
                        2,
                        &mut model.mlp_up_bank,
                        &grads.mlp_up_bank,
                        &bank_shapes[2],
                    );
                    muon.step_bank(
                        3,
                        &mut model.mlp_down_bank,
                        &grads.mlp_down_bank,
                        &bank_shapes[3],
                    );
                }

                adamw_embed.step(&mut model.tok_emb, &grads.tok_emb, &mut state_tok_emb);
                adamw_embed.step(
                    &mut model.bigram_embed,
                    &grads.bigram_embed,
                    &mut state_bigram_embed,
                );
                adamw_embed.step(
                    &mut model.bigram_proj,
                    &grads.bigram_proj,
                    &mut state_bigram_proj,
                );
                adamw_embed.step(&mut model.ve_embed, &grads.ve_embed, &mut state_ve_embed);

                adamw_scalar.step(
                    &mut model.smear_gate,
                    &grads.smear_gate,
                    &mut state_smear_gate,
                );
                adamw_scalar.step(
                    &mut model.skip_weights,
                    &grads.skip_weights,
                    &mut state_skip_weights,
                );
                adamw_scalar.step(&mut model.ve_proj, &grads.ve_proj, &mut state_ve_proj);
                {
                    let mut ve_scale_slice = [model.ve_scale];
                    let grad_ve_scale_slice = [grads.ve_scale];
                    adamw_scalar.step(
                        &mut ve_scale_slice,
                        &grad_ve_scale_slice,
                        &mut state_ve_scale,
                    );
                    model.ve_scale = ve_scale_slice[0];
                }
                adamw_scalar.step(
                    &mut model.ve_layer_scales,
                    &grads.ve_layer_scales,
                    &mut state_ve_layer_scales,
                );
                {
                    let mut bigram_scale_slice = [model.bigram_scale];
                    let grad_bigram_scale_slice = [grads.bigram_scale];
                    adamw_scalar.step(
                        &mut bigram_scale_slice,
                        &grad_bigram_scale_slice,
                        &mut state_bigram_scale,
                    );
                    model.bigram_scale = bigram_scale_slice[0];
                }
                for i in 0..n {
                    adamw_scalar.step(
                        &mut model.blocks[i].attn_scale,
                        &grads.block_attn_scale[i],
                        &mut state_attn_scale[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].mlp_scale,
                        &grads.block_mlp_scale[i],
                        &mut state_mlp_scale[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].resid_mix,
                        &grads.block_resid_mix[i],
                        &mut state_resid_mix[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].q_gain,
                        &grads.block_q_gain[i],
                        &mut state_q_gain[i],
                    );
                }

                flatten_params_into(&model, &mut flat_buf);
                ema.update(&flat_buf);
                if train_config.should_swa(step) {
                    swa.accumulate(&flat_buf);
                }
            }
            steps_completed = step + 1;
        }

        let wallclock_seconds = start.elapsed().as_secs_f64();
        let ms_per_step = if steps_completed > 0 {
            (wallclock_seconds * 1000.0) / steps_completed as f64
        } else {
            0.0
        };

        #[cfg(feature = "cuda")]
        if let Some(runtime) = cuda_single_fast_runtime.as_ref() {
            runtime
                .gpu_model
                .sync_to_cpu_reference(&mut model, &self.plan)?;
        }
        #[cfg(feature = "cuda")]
        if let Some(runtime) = cuda_distributed_runtime.as_ref() {
            runtime.replicas[0]
                .gpu_model
                .sync_to_cpu_reference(&mut model, &self.plan)?;
        }
        #[cfg(feature = "cuda")]
        let distributed_sync = cuda_distributed_runtime
            .as_ref()
            .map(|runtime| runtime.distributed_sync)
            .unwrap_or(false);
        #[cfg(not(feature = "cuda"))]
        let distributed_sync = false;

        let artifact_bytes = match mode {
            RunMode::Smoke => None,
            _ => {
                let artifact_path = std::path::Path::new(&self.run_spec.train.artifact_path);
                pg_quant::export::export_model_with_spec(
                    &model,
                    &self.run_spec.quant,
                    &self.plan.variant_fingerprint,
                    artifact_path,
                )
                .ok()
            }
        };
        let artifact_budget_ok = artifact_bytes.map(|bytes| self.plan.artifact_budget_ok(bytes));
        let (bpb_luts, bpb_byte_source) =
            load_bpb_luts(&self.run_spec, model_config.vocab_size, mode)?;
        let proxy_bpb = if matches!(mode, RunMode::Proxy) {
            let prev: Vec<u16> = last_input_ids.iter().map(|&v| v as u16).collect();
            let tgt: Vec<u16> = last_targets.iter().map(|&v| v as u16).collect();
            let byte_count = bpb_luts.count_bytes(&prev, &tgt);
            Some(compute_bpb(
                final_loss as f64,
                last_targets.len() as f64,
                byte_count,
            ))
        } else {
            None
        };
        let proxy_metric_source = proxy_bpb.map(|_| "last_batch_train_loss".to_string());
        let (eval_loss, final_bpb, eval_tokens) =
            if let Some(pattern) = self.run_spec.train.validation_data_pattern.as_deref() {
                let mut eval_model = GptModel::new(model_config.clone());
                eval_model.fill_deterministic();
                if !matches!(mode, RunMode::Smoke) && artifact_bytes.is_some() {
                    pg_quant::export::load_artifact(
                        std::path::Path::new(&self.run_spec.train.artifact_path),
                        &mut eval_model,
                    )?;
                } else {
                    eval_model = model;
                }
                let max_eval_tokens = self.run_spec.eval.max_tokens.map(|limit| limit.max(2));
                let tokens = pg_data::token_stream::load_validation_tokens_limited(
                    pattern,
                    max_eval_tokens,
                )?
                .into_iter()
                .map(|v| v as u32)
                .collect::<Vec<_>>();
                let token_bytes = bpb_luts.pair_byte_counts_u32(&tokens);
                let seq_len = self
                    .run_spec
                    .model
                    .eval_seq_len
                    .min(tokens.len().saturating_sub(1))
                    .max(1);
                let (loss, bpb) = if self.run_spec.eval.qttt {
                    let mut cfg = pg_eval::qttt::QttTConfig::paper_default(seq_len);
                    cfg.stride = self.run_spec.eval.stride;
                    cfg.seq_len = seq_len;
                    cfg.chunk_tokens = self.run_spec.eval.chunk_tokens;
                    pg_eval::qttt::eval_qttt(&mut eval_model, &tokens, &token_bytes, &cfg)
                } else {
                    pg_eval::sliding::eval_sliding(
                        &eval_model,
                        &tokens,
                        &token_bytes,
                        self.run_spec.eval.stride,
                        seq_len,
                    )
                };
                (Some(loss), Some(bpb), Some(tokens.len()))
            } else {
                (None, None, None)
            };

        Ok(VariantResult {
            run_name: self.run_spec.name.clone(),
            mode,
            train_backend: self.run_spec.train.backend,
            variant_fingerprint: self.plan.variant_fingerprint.clone(),
            steps_completed,
            train_loss: final_loss,
            proxy_bpb,
            eval_loss,
            final_bpb,
            eval_tokens,
            artifact_bytes,
            artifact_budget_ok,
            bank_update_backend: bank_update_backend.to_string(),
            train_data_source: train_data_source.to_string(),
            bpb_byte_source: bpb_byte_source.to_string(),
            proxy_metric_source,
            ms_per_step,
            wallclock_seconds,
            rank,
            world_size,
            distributed_sync,
        })
    }
}

#[cfg(feature = "cuda")]
fn cuda_single_hybrid_step(
    runtime: &mut CudaSingleHybridRuntime,
    model: &GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    grads: &mut GradBuffers,
) -> PgResult<f32> {
    use pg_model::gpu::{GpuActivations, GpuGradBuffers};

    runtime.gpu_model.sync_from_cpu_reference(model, plan)?;
    let stream = runtime.gpu_model.gemm.stream().clone();
    runtime
        .input_ids
        .copy_from_host_bytes(bytemuck::cast_slice(input_ids))?;
    runtime
        .targets
        .copy_from_host_bytes(bytemuck::cast_slice(targets))?;
    let mut gpu_buf = GpuActivations::new_for_plan(plan, input_ids.len(), stream.clone())?;
    let mut gpu_grads = GpuGradBuffers::new(&model.config, stream.clone())?;
    let loss = runtime.gpu_model.backward_with_state(
        &runtime.input_ids,
        &runtime.targets,
        &mut gpu_buf,
        &mut runtime.backward_state,
        &mut gpu_grads,
    )?;
    stream
        .synchronize()
        .map_err(|e| pg_core::PgError::InvalidOp(format!("stream sync failed: {:?}", e)))?;

    grads
        .tok_emb
        .copy_from_slice(&download_gpu_f32(&gpu_grads.tok_emb)?);
    grads
        .bigram_embed
        .copy_from_slice(&download_gpu_f32(&gpu_grads.bigram_embed)?);
    grads
        .bigram_proj
        .copy_from_slice(&download_gpu_f32(&gpu_grads.bigram_proj)?);
    grads.bigram_scale = download_gpu_f32(&gpu_grads.bigram_scale)?[0];
    grads
        .smear_gate
        .copy_from_slice(&download_gpu_f32(&gpu_grads.smear_gate)?);
    grads
        .skip_weights
        .copy_from_slice(&download_gpu_f32(&gpu_grads.skip_weights)?);
    grads
        .qo_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.qo_bank)?);
    grads
        .kv_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.kv_bank)?);
    grads
        .mlp_up_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.mlp_up_bank)?);
    grads
        .mlp_down_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.mlp_down_bank)?);
    for (dst, src) in grads
        .block_attn_scale
        .iter_mut()
        .zip(gpu_grads.block_attn_scale.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_mlp_scale
        .iter_mut()
        .zip(gpu_grads.block_mlp_scale.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_resid_mix
        .iter_mut()
        .zip(gpu_grads.block_resid_mix.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_q_gain
        .iter_mut()
        .zip(gpu_grads.block_q_gain.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    grads
        .ve_embed
        .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_embed)?);
    grads
        .ve_proj
        .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_proj)?);
    grads.ve_scale = download_gpu_f32(&gpu_grads.ve_scale)?[0];
    grads
        .ve_layer_scales
        .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_layer_scales)?);

    Ok(loss)
}

#[cfg(feature = "cuda")]
fn zero_gpu_grads(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grads: &pg_model::gpu::GpuGradBuffers,
) -> PgResult<()> {
    let zero = |tensor: &pg_core::GpuTensor| {
        kernels.scale_inplace(
            pg_kernels::gpu_kernels::CudaPtr(tensor.cu_ptr(kernels.stream())?),
            0.0,
            tensor.numel() as u32,
        )
    };

    zero(&grads.tok_emb)?;
    zero(&grads.bigram_embed)?;
    zero(&grads.bigram_proj)?;
    zero(&grads.bigram_scale)?;
    zero(&grads.smear_gate)?;
    zero(&grads.skip_weights)?;
    zero(&grads.qo_bank)?;
    zero(&grads.kv_bank)?;
    zero(&grads.mlp_up_bank)?;
    zero(&grads.mlp_down_bank)?;
    for tensor in &grads.block_attn_scale {
        zero(tensor)?;
    }
    for tensor in &grads.block_mlp_scale {
        zero(tensor)?;
    }
    for tensor in &grads.block_resid_mix {
        zero(tensor)?;
    }
    for tensor in &grads.block_q_gain {
        zero(tensor)?;
    }
    zero(&grads.ve_embed)?;
    zero(&grads.ve_proj)?;
    zero(&grads.ve_scale)?;
    zero(&grads.ve_layer_scales)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_single_fast_step(
    runtime: &mut CudaSingleFastRuntime,
    _model: &mut GptModel,
    _plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<f32> {
    let loss = cuda_fast_compute_grads(runtime, input_ids, targets)?;
    cuda_fast_apply_updates(runtime, train_config, step, lr_scale)?;
    Ok(loss)
}

#[cfg(feature = "cuda")]
fn cuda_fast_compute_grads(
    runtime: &mut CudaSingleFastRuntime,
    input_ids: &[u32],
    targets: &[u32],
) -> PgResult<f32> {
    zero_gpu_grads(&runtime.gpu_model.kernels, &runtime.gpu_grads)?;

    runtime
        .input_ids
        .copy_from_host_bytes(bytemuck::cast_slice(input_ids))?;
    runtime
        .targets
        .copy_from_host_bytes(bytemuck::cast_slice(targets))?;

    let loss = runtime.gpu_model.backward_with_state(
        &runtime.input_ids,
        &runtime.targets,
        &mut runtime.gpu_buf,
        &mut runtime.backward_state,
        &mut runtime.gpu_grads,
    )?;
    Ok(loss)
}

#[cfg(feature = "cuda")]
fn collect_gpu_grad_refs(grads: &pg_model::gpu::GpuGradBuffers) -> Vec<&pg_core::GpuTensor> {
    let mut grad_refs: Vec<&pg_core::GpuTensor> = vec![
        &grads.tok_emb,
        &grads.bigram_embed,
        &grads.bigram_proj,
        &grads.bigram_scale,
        &grads.smear_gate,
        &grads.skip_weights,
        &grads.qo_bank,
        &grads.kv_bank,
        &grads.mlp_up_bank,
        &grads.mlp_down_bank,
        &grads.ve_embed,
        &grads.ve_proj,
        &grads.ve_scale,
        &grads.ve_layer_scales,
    ];
    grad_refs.extend(grads.block_attn_scale.iter());
    grad_refs.extend(grads.block_mlp_scale.iter());
    grad_refs.extend(grads.block_resid_mix.iter());
    grad_refs.extend(grads.block_q_gain.iter());
    grad_refs
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_updates(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<()> {
    let (gpu_optimizer, grad_norm_scratch, gpu_model, gpu_grads) = (
        &mut runtime.gpu_optimizer,
        &mut runtime.grad_norm_scratch,
        &runtime.gpu_model,
        &runtime.gpu_grads,
    );
    let grad_refs = collect_gpu_grad_refs(gpu_grads);
    gpu_optimizer.clip_grad_norm(
        &gpu_model.kernels,
        &grad_refs,
        train_config.grad_clip_norm,
        grad_norm_scratch,
    )?;

    let embed_hyper = AdamWHyper {
        lr: train_config.embed_lr * lr_scale,
        beta1: train_config.adam_beta1,
        beta2: train_config.adam_beta2,
        eps: train_config.adam_eps,
        weight_decay: train_config.adam_wd,
    };
    let scalar_hyper = AdamWHyper {
        lr: train_config.scalar_lr * lr_scale,
        beta1: train_config.adam_beta1,
        beta2: train_config.adam_beta2,
        eps: train_config.adam_eps,
        weight_decay: train_config.adam_wd,
    };
    let matrix_lr = train_config.matrix_lr * lr_scale;
    runtime.gpu_muon.lr = matrix_lr;
    runtime.gpu_muon.momentum = train_config.muon_momentum_at(step);
    runtime.gpu_muon.weight_decay = train_config.muon_wd;
    runtime.gpu_muon.step_bank(
        &runtime.gpu_model.kernels,
        0,
        &runtime.gpu_model.weights.qo_bank,
        &runtime.gpu_grads.qo_bank,
    )?;
    runtime.gpu_muon.step_bank(
        &runtime.gpu_model.kernels,
        1,
        &runtime.gpu_model.weights.kv_bank,
        &runtime.gpu_grads.kv_bank,
    )?;
    runtime.gpu_muon.step_bank(
        &runtime.gpu_model.kernels,
        2,
        &runtime.gpu_model.weights.mlp_up_bank,
        &runtime.gpu_grads.mlp_up_bank,
    )?;
    runtime.gpu_muon.step_bank(
        &runtime.gpu_model.kernels,
        3,
        &runtime.gpu_model.weights.mlp_down_bank,
        &runtime.gpu_grads.mlp_down_bank,
    )?;

    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.tok_emb,
        &runtime.gpu_grads.tok_emb,
        &mut runtime.state_tok_emb,
        embed_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.bigram_embed,
        &runtime.gpu_grads.bigram_embed,
        &mut runtime.state_bigram_embed,
        embed_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.bigram_proj,
        &runtime.gpu_grads.bigram_proj,
        &mut runtime.state_bigram_proj,
        embed_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.ve_embed,
        &runtime.gpu_grads.ve_embed,
        &mut runtime.state_ve_embed,
        embed_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.smear_gate,
        &runtime.gpu_grads.smear_gate,
        &mut runtime.state_smear_gate,
        scalar_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.skip_weights,
        &runtime.gpu_grads.skip_weights,
        &mut runtime.state_skip_weights,
        scalar_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.ve_proj,
        &runtime.gpu_grads.ve_proj,
        &mut runtime.state_ve_proj,
        scalar_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.ve_layer_scales,
        &runtime.gpu_grads.ve_layer_scales,
        &mut runtime.state_ve_layer_scales,
        scalar_hyper,
    )?;
    for i in 0..runtime.state_attn_scale.len() {
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.attn_scales[i],
            &runtime.gpu_grads.block_attn_scale[i],
            &mut runtime.state_attn_scale[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.mlp_scales[i],
            &runtime.gpu_grads.block_mlp_scale[i],
            &mut runtime.state_mlp_scale[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.resid_mix[i],
            &runtime.gpu_grads.block_resid_mix[i],
            &mut runtime.state_resid_mix[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.q_gains[i],
            &runtime.gpu_grads.block_q_gain[i],
            &mut runtime.state_q_gain[i],
            scalar_hyper,
        )?;
    }

    let stream = runtime.gpu_model.gemm.stream().clone();
    stream
        .synchronize()
        .map_err(|e| pg_core::PgError::InvalidOp(format!("stream sync failed: {:?}", e)))?;

    let scalar_adam = AdamW::new(
        train_config.scalar_lr * lr_scale,
        train_config.adam_beta1,
        train_config.adam_beta2,
        train_config.adam_eps,
        train_config.adam_wd,
    );
    {
        let mut param = [runtime.gpu_model.weights.bigram_scale];
        let grad = [download_gpu_f32(&runtime.gpu_grads.bigram_scale)?[0]];
        scalar_adam.step(&mut param, &grad, &mut runtime.state_bigram_scale);
        runtime.gpu_model.weights.bigram_scale = param[0];
    }
    {
        let mut param = [runtime.gpu_model.weights.ve_scale];
        let grad = [download_gpu_f32(&runtime.gpu_grads.ve_scale)?[0]];
        scalar_adam.step(&mut param, &grad, &mut runtime.state_ve_scale);
        runtime.gpu_model.weights.ve_scale = param[0];
    }
    runtime.gpu_model.weights.ve_layer_scales_host =
        download_gpu_f32(&runtime.gpu_model.weights.ve_layer_scales)?;

    Ok(())
}

#[cfg(feature = "cuda")]
fn scale_gpu_tensor(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    tensor: &pg_core::GpuTensor,
    scale: f32,
) -> PgResult<()> {
    kernels.scale_inplace(
        pg_kernels::gpu_kernels::CudaPtr(tensor.cu_ptr(kernels.stream())?),
        scale,
        tensor.numel() as u32,
    )
}

#[cfg(feature = "cuda")]
fn scale_all_gpu_grads(runtime: &mut CudaSingleFastRuntime, scale: f32) -> PgResult<()> {
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.tok_emb,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.bigram_embed,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.bigram_proj,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.bigram_scale,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.smear_gate,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.skip_weights,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.qo_bank,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.kv_bank,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.mlp_up_bank,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.mlp_down_bank,
        scale,
    )?;
    for tensor in &runtime.gpu_grads.block_attn_scale {
        scale_gpu_tensor(&runtime.gpu_model.kernels, tensor, scale)?;
    }
    for tensor in &runtime.gpu_grads.block_mlp_scale {
        scale_gpu_tensor(&runtime.gpu_model.kernels, tensor, scale)?;
    }
    for tensor in &runtime.gpu_grads.block_resid_mix {
        scale_gpu_tensor(&runtime.gpu_model.kernels, tensor, scale)?;
    }
    for tensor in &runtime.gpu_grads.block_q_gain {
        scale_gpu_tensor(&runtime.gpu_model.kernels, tensor, scale)?;
    }
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.ve_embed,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.ve_proj,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.ve_scale,
        scale,
    )?;
    scale_gpu_tensor(
        &runtime.gpu_model.kernels,
        &runtime.gpu_grads.ve_layer_scales,
        scale,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_all_reduce_average(runtime: &mut CudaDistributedRuntime) -> PgResult<()> {
    macro_rules! all_reduce_field {
        ($field:ident) => {{
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for (replica, comm) in runtime.replicas.iter_mut().zip(runtime.comms.iter()) {
                comm.all_reduce_sum_tensor_f32_in_place(&mut replica.gpu_grads.$field)?;
            }
            cudarc::nccl::group_end()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
        }};
    }

    all_reduce_field!(tok_emb);
    all_reduce_field!(bigram_embed);
    all_reduce_field!(bigram_proj);
    all_reduce_field!(bigram_scale);
    all_reduce_field!(smear_gate);
    all_reduce_field!(skip_weights);
    all_reduce_field!(qo_bank);
    all_reduce_field!(kv_bank);
    all_reduce_field!(mlp_up_bank);
    all_reduce_field!(mlp_down_bank);
    all_reduce_field!(ve_embed);
    all_reduce_field!(ve_proj);
    all_reduce_field!(ve_scale);
    all_reduce_field!(ve_layer_scales);

    let n_blocks = runtime
        .replicas
        .first()
        .map(|replica| replica.gpu_grads.block_attn_scale.len())
        .unwrap_or(0);
    for i in 0..n_blocks {
        cudarc::nccl::group_start()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
        for (replica, comm) in runtime.replicas.iter_mut().zip(runtime.comms.iter()) {
            comm.all_reduce_sum_tensor_f32_in_place(&mut replica.gpu_grads.block_attn_scale[i])?;
            comm.all_reduce_sum_tensor_f32_in_place(&mut replica.gpu_grads.block_mlp_scale[i])?;
            comm.all_reduce_sum_tensor_f32_in_place(&mut replica.gpu_grads.block_resid_mix[i])?;
            comm.all_reduce_sum_tensor_f32_in_place(&mut replica.gpu_grads.block_q_gain[i])?;
        }
        cudarc::nccl::group_end()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
    }

    let inv_world = 1.0f32 / runtime.replicas.len() as f32;
    for replica in runtime.replicas.iter_mut() {
        scale_all_gpu_grads(replica, inv_world)?;
    }
    runtime.distributed_sync = true;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_step(
    runtime: &mut CudaDistributedRuntime,
    batches: &[(Vec<u32>, Vec<u32>)],
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<f32> {
    if runtime.replicas.len() != batches.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "cuda_distributed_step got {} batches for {} replicas",
            batches.len(),
            runtime.replicas.len()
        )));
    }

    let mut total_loss = 0.0f32;
    for (replica, (input_ids, targets)) in runtime.replicas.iter_mut().zip(batches.iter()) {
        total_loss += cuda_fast_compute_grads(replica, input_ids, targets)?;
    }
    cuda_distributed_all_reduce_average(runtime)?;
    for replica in runtime.replicas.iter_mut() {
        cuda_fast_apply_updates(replica, train_config, step, lr_scale)?;
    }
    Ok(total_loss / runtime.replicas.len() as f32)
}

#[cfg(feature = "cuda")]
fn download_gpu_f32(tensor: &pg_core::GpuTensor) -> PgResult<Vec<f32>> {
    let bytes = tensor.to_host_bytes()?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn validate_executable_variant(run_spec: &RunSpec, mode: RunMode) -> PgResult<()> {
    let _ = (run_spec, mode);
    Ok(())
}

fn validate_backend_request(run_spec: &RunSpec, mode: RunMode) -> PgResult<()> {
    if matches!(mode, RunMode::Record) && run_spec.train.backend != TrainBackend::CudaDistributed {
        return Err(pg_core::PgError::InvalidOp(
            "record mode requires --backend cuda-distributed; cpu and cuda-single backends are not submission-valid for a real 600s attempt".into(),
        ));
    }
    if matches!(mode, RunMode::Record) && run_spec.train.fast_bank_updates {
        return Err(pg_core::PgError::InvalidOp(
            "record mode forbids --fast-bank-updates because it bypasses Muon".into(),
        ));
    }
    if run_spec.train.world_size > 1
        && matches!(
            run_spec.train.backend,
            TrainBackend::CudaSingle | TrainBackend::CudaSingleParity
        )
    {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-single backends require world_size=1; use cuda-distributed for multi-rank execution".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaSingleParity && !matches!(mode, RunMode::Smoke) {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-single-parity is a smoke/debug backend only and cannot produce proxy or record metrics".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaDistributed && run_spec.train.world_size < 2 {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-distributed requires --world-size >= 2".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaDistributed && run_spec.train.rank != 0 {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-distributed currently runs as one local multi-GPU process, so rank must be 0"
                .into(),
        ));
    }
    #[cfg(not(feature = "cuda"))]
    if matches!(
        run_spec.train.backend,
        TrainBackend::CudaSingle | TrainBackend::CudaSingleParity | TrainBackend::CudaDistributed
    ) {
        return Err(pg_core::PgError::InvalidOp(
            "selected CUDA backend requires building with --features cuda".into(),
        ));
    }
    match run_spec.train.backend {
        TrainBackend::Cpu => Ok(()),
        TrainBackend::CudaSingle => Ok(()),
        TrainBackend::CudaSingleParity => Ok(()),
        TrainBackend::CudaDistributed => Ok(()),
    }
}

fn load_bpb_luts(
    run_spec: &RunSpec,
    vocab_size: usize,
    mode: RunMode,
) -> PgResult<(BpbLuts, &'static str)> {
    if let Some(path) = run_spec.eval.tokenizer_vocab_path.as_deref() {
        let luts = BpbLuts::from_vocab_file(std::path::Path::new(path))?;
        if matches!(mode, RunMode::Record) && luts.base_bytes.len() != vocab_size {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record mode tokenizer vocab has {} pieces but model vocab_size is {}; use the submission tokenizer for final BPB",
                luts.base_bytes.len(),
                vocab_size
            )));
        }
        Ok((luts, "tokenizer_vocab"))
    } else if matches!(mode, RunMode::Record) && run_spec.train.validation_data_pattern.is_some() {
        Err(pg_core::PgError::InvalidOp(
            "record mode evaluation requires --tokenizer-vocab; placeholder byte counts are not submission-valid".into(),
        ))
    } else {
        Ok((BpbLuts::placeholder(vocab_size), "placeholder"))
    }
}

fn count_params(model: &GptModel) -> usize {
    let mut total = 0;
    total += model.tok_emb.len();
    total += model.bigram_embed.len();
    total += model.bigram_proj.len();
    total += 1;
    total += model.smear_gate.len();
    total += model.skip_weights.len();
    total += model.qo_bank.len();
    total += model.kv_bank.len();
    total += model.mlp_up_bank.len();
    total += model.mlp_down_bank.len();
    for bp in &model.blocks {
        total += bp.attn_scale.len();
        total += bp.mlp_scale.len();
        total += bp.resid_mix.len();
        total += bp.q_gain.len();
    }
    total += model.ve_embed.len();
    total += model.ve_proj.len();
    total += 1;
    total += model.ve_layer_scales.len();
    total
}

fn smoke_bank_step(param: &mut [f32], grad: &[f32], lr: f32, weight_decay: f32) {
    let decay = 1.0 - lr * weight_decay;
    for (p, &g) in param.iter_mut().zip(grad.iter()) {
        *p = decay * *p - lr * g;
    }
}

fn flatten_params_into(model: &GptModel, flat: &mut [f32]) {
    let mut pos = 0;
    fn copy(src: &[f32], dst: &mut [f32], pos: &mut usize) {
        dst[*pos..*pos + src.len()].copy_from_slice(src);
        *pos += src.len();
    }

    copy(&model.tok_emb, flat, &mut pos);
    copy(&model.bigram_embed, flat, &mut pos);
    copy(&model.bigram_proj, flat, &mut pos);
    flat[pos] = model.bigram_scale;
    pos += 1;
    copy(&model.smear_gate, flat, &mut pos);
    copy(&model.skip_weights, flat, &mut pos);
    copy(&model.qo_bank, flat, &mut pos);
    copy(&model.kv_bank, flat, &mut pos);
    copy(&model.mlp_up_bank, flat, &mut pos);
    copy(&model.mlp_down_bank, flat, &mut pos);
    for bp in &model.blocks {
        copy(&bp.attn_scale, flat, &mut pos);
        copy(&bp.mlp_scale, flat, &mut pos);
        copy(&bp.resid_mix, flat, &mut pos);
        copy(&bp.q_gain, flat, &mut pos);
    }
    copy(&model.ve_embed, flat, &mut pos);
    copy(&model.ve_proj, flat, &mut pos);
    flat[pos] = model.ve_scale;
    pos += 1;
    copy(&model.ve_layer_scales, flat, &mut pos);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_requires_distributed_backend() {
        for backend in [
            TrainBackend::Cpu,
            TrainBackend::CudaSingle,
            TrainBackend::CudaSingleParity,
        ] {
            let mut spec = RunSpec::default();
            spec.train.backend = backend;
            let err = validate_backend_request(&spec, RunMode::Record).unwrap_err();
            assert!(
                err.to_string().contains("cuda-distributed"),
                "unexpected error for {:?}: {}",
                backend,
                err
            );
        }
    }

    #[test]
    fn cuda_single_requires_world_size_one() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        spec.train.world_size = 2;
        let err = validate_backend_request(&spec, RunMode::Smoke).unwrap_err();
        assert!(err.to_string().contains("world_size=1"));
    }

    #[test]
    fn cuda_single_parity_stays_smoke_only() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingleParity;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("smoke/debug"));
    }

    #[test]
    fn cuda_single_smoke_support_matches_build() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        let result = validate_backend_request(&spec, RunMode::Smoke);
        if cfg!(feature = "cuda") {
            assert!(result.is_ok(), "expected cuda-single smoke to be allowed");
        } else {
            let err = result.unwrap_err();
            assert!(
                err.to_string()
                    .contains("requires building with --features cuda")
            );
        }
    }

    #[test]
    fn cuda_distributed_requires_world_size_two_or_more() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 1;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("world-size >= 2"));
    }

    #[test]
    fn cuda_distributed_requires_rank_zero() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 2;
        spec.train.rank = 1;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("rank must be 0"));
    }

    #[test]
    fn executable_variants_include_recurrence_and_parallel_residual() {
        let mut spec = RunSpec::default();
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 2;
        spec.model.recurrence.repeat_layers = 2;
        spec.model.parallel_residual.enabled = true;
        spec.model.parallel_residual.split_attention_mlp = true;
        assert!(validate_executable_variant(&spec, RunMode::Proxy).is_ok());
    }
}
