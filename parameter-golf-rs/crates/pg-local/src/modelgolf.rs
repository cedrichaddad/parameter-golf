use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pg_core::{PgError, PgResult};
use pg_model::{
    DistributedOptimizerBackend, ExecutionPlan, ForwardBuffer, GptModel, ModelComputePrecision,
    OutputCeBackend, QuantScheme, QuantSpec, RunSpec,
};
use serde::{Deserialize, Serialize};

use crate::write_report_if_requested;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelGolfSection {
    Full,
    Pack,
    PackExperiment,
    PackSourceReport,
    LqerSourceReport,
    Cache,
    CacheExperiment,
    CacheSourceReport,
    Delta,
    DeltaSourceReport,
    Train,
    TrainSourceReport,
    ResourceCostSourceReport,
    OptimizerCommSourceReport,
    Kernel,
    KernelExperiment,
    KernelSourceReport,
    Wind,
    WindExperiment,
    WindSourceReport,
    Scale,
    ReleaseEvidence,
    ReleaseCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfOptions {
    pub section: ModelGolfSection,
    pub spec: PathBuf,
    pub output: Option<PathBuf>,
    pub hardware: String,
    pub runtime: String,
    pub memory_budget_bytes: Option<usize>,
    pub latency_target_ms: Option<f64>,
    pub quality_budget_ppl_pct: Option<f64>,
    pub context: Option<usize>,
    pub batch: Option<usize>,
    pub artifact_budget_bytes: Option<usize>,
    pub delta_budget_bytes: Option<usize>,
    pub proof_artifact: Option<PathBuf>,
    pub quality_calibration: Option<PathBuf>,
    pub release_evidence: Option<PathBuf>,
    pub release_artifact: Option<PathBuf>,
    pub validation_dataset_id: Option<String>,
    pub validation_command: Option<String>,
    pub heldout_bpb: Option<f64>,
    pub baseline_bpb: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub backend_id: Option<String>,
    pub kernel_id: Option<String>,
    pub long_context_dataset_id: Option<String>,
    pub fused_runtime: Option<bool>,
    pub parity_pass: Option<bool>,
    pub long_context_bpb_delta_pct: Option<f64>,
    pub speedup_x: Option<f64>,
    pub calibration_dataset_id: Option<String>,
    pub production_svd_validated: Option<bool>,
    pub calibrated_tensor_sensitivity: Option<bool>,
    pub equal_byte_bpb_delta: Option<f64>,
    pub domain_dataset_id: Option<String>,
    pub trained_delta_bytes: Option<usize>,
    pub legality_pass: Option<bool>,
    pub score_first_trace_or_review: Option<bool>,
    pub equal_byte_domain_bpb_delta: Option<f64>,
    pub training_run_id: Option<String>,
    pub gpu_backend_integrated: Option<bool>,
    pub proxy_calibrated: Option<bool>,
    pub post_export_bpb_delta_vs_posthoc: Option<f64>,
    pub measurement_run_id: Option<String>,
    pub power_meter_id: Option<String>,
    pub wall_time_seconds: Option<f64>,
    pub average_power_watts: Option<f64>,
    pub energy_joules: Option<f64>,
    pub telemetry_validated: Option<bool>,
    pub distributed_backend_id: Option<String>,
    pub reduce_scatter_parity_pass: Option<bool>,
    pub all_gather_parity_pass: Option<bool>,
    pub optimizer_update_parity_pass: Option<bool>,
    pub nccl_trace_validated: Option<bool>,
    pub overlap_validated: Option<bool>,
    pub measured_comm_time_ms: Option<f64>,
    pub measured_step_time_ms: Option<f64>,
    pub communication_speedup_x: Option<f64>,
    pub generated_kernel_ids: Option<String>,
    pub generated_kernels: Option<bool>,
    pub memory_reduction_x: Option<f64>,
    pub trace_corpus_id: Option<String>,
    pub calibration_report_id: Option<String>,
    pub fresh_profiler_traces: Option<bool>,
    pub external_timing_validated: Option<bool>,
    pub holdout_spearman: Option<f64>,
    pub mean_abs_pct_error: Option<f64>,
    pub evidence_source: Option<PathBuf>,
    pub source_report_dir: Option<PathBuf>,
    pub evidence_id: Option<String>,
    pub generated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfReport {
    pub kind: &'static str,
    pub project_name: &'static str,
    pub thesis: &'static str,
    pub spec_path: String,
    pub spec_name: String,
    pub spec_fingerprint: String,
    pub resource_contract: ResourceContractReport,
    pub cost_model: ConstraintCostModelReport,
    pub artifact_ir: ModelArtifactIrReport,
    pub pack: PackPlannerReport,
    pub pack_experiment: Option<PackMeasuredExperimentReport>,
    pub cache: CachePlannerReport,
    pub cache_experiment: Option<CacheGolfKvExperimentReport>,
    pub delta: DeltaPlannerReport,
    pub train: TrainPlannerReport,
    pub optimizer_comm: OptimizerCommReport,
    pub kernel_forge: KernelForgeReport,
    pub kernel_experiment: Option<KernelForgeCeExperimentReport>,
    pub wind_tunnel: ModelGolfWindReport,
    pub wind_experiment: Option<ModelGolfWindExperimentReport>,
    pub scale_golf: ScaleGolfReport,
    pub platform_status: Vec<PlatformModuleStatus>,
    pub release_readiness: ModelGolfReleaseReadinessReport,
    pub caveats: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContractReport {
    pub hardware: String,
    pub runtime: String,
    pub artifact_budget_bytes: usize,
    pub memory_budget_bytes: Option<usize>,
    pub latency_target_ms: Option<f64>,
    pub quality_budget_ppl_pct: Option<f64>,
    pub training_time_budget_seconds: f64,
    pub nominal_power_watts_proxy: f64,
    pub train_energy_budget_joules_proxy: f64,
    pub context_tokens: usize,
    pub batch_sequences: usize,
    pub train_batch_tokens: usize,
    pub eval_stride: usize,
    pub adaptation_legality: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintCostModelReport {
    pub kind: &'static str,
    pub training_time_budget_seconds: f64,
    pub nominal_power_watts_proxy: f64,
    pub train_energy_budget_joules_proxy: f64,
    pub train_tokens_per_budget_second: f64,
    pub artifact_bytes_per_budget_second: f64,
    pub parameter_elems_per_energy_joule_proxy: f64,
    pub latency_target_ms: Option<f64>,
    pub latency_budget_tokens_per_second: Option<f64>,
    pub energy_proxy_source: String,
    pub evidence_boundary: &'static str,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifactIrReport {
    pub model_family: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub mlp_dim: usize,
    pub total_parameter_elems_estimate: usize,
    pub tensor_groups: Vec<ModelGolfTensorGroup>,
    pub typed_metadata: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfTensorGroup {
    pub name: String,
    pub role: TensorRole,
    pub rows: usize,
    pub cols: usize,
    pub elems: usize,
    pub current_bits: u8,
    pub current_weight_bytes: usize,
    pub scale_bytes: usize,
    pub lqer_bytes: usize,
    pub sensitivity: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    AttentionQ,
    AttentionO,
    KeyCacheProjection,
    ValueCacheProjection,
    MlpUp,
    MlpDown,
    TokenEmbedding,
    AttentionGate,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackPlannerReport {
    pub kind: &'static str,
    pub algorithm: &'static str,
    pub objective: &'static str,
    pub target_artifact_bytes: usize,
    pub bucket_bytes: usize,
    pub feasible: bool,
    pub selected_bytes: usize,
    pub selected_quality_loss: f64,
    pub estimated_bytes_remaining: isize,
    pub selected_options: Vec<PrecisionOptionReport>,
    pub lqer_candidates: Vec<LqerCandidateReport>,
    pub candidate_count: usize,
    pub quality_calibration: PackQualityCalibrationReport,
    pub quality_comparison: PackQualityComparisonReport,
    pub lqer_proofs: Vec<PackLqerProofReport>,
    pub artifact_proof: Option<PackArtifactProofReport>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackQualityComparisonReport {
    pub kind: &'static str,
    pub comparison_budget_bytes: usize,
    pub selected_plan_name: &'static str,
    pub rows: Vec<PackQualityComparisonRowReport>,
    pub best_proxy_quality_plan: Option<String>,
    pub selected_beats_uniform_best_proxy: Option<bool>,
    pub selected_beats_mixed_no_lqer_proxy: Option<bool>,
    pub selected_beats_lqer_control_proxy: Option<bool>,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackQualityComparisonRowReport {
    pub plan_name: String,
    pub artifact_bytes: usize,
    pub byte_delta_vs_selected: isize,
    pub fits_comparison_budget: bool,
    pub estimated_quality_loss: f64,
    pub local_weighted_residual_mse_proxy: f64,
    pub local_activation_ce_bound_proxy: f64,
    pub lqer_group_count: usize,
    pub lqer_groups: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackQualityCalibrationReport {
    pub kind: &'static str,
    pub source_path: Option<String>,
    pub applied: bool,
    pub points: usize,
    pub default_scale: f64,
    pub mean_abs_error_before: Option<f64>,
    pub mean_abs_error_after: Option<f64>,
    pub max_abs_error_after: Option<f64>,
    pub factors: Vec<PackQualityCalibrationFactorReport>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackQualityCalibrationFactorReport {
    pub scope: String,
    pub bits: u8,
    pub lqer: bool,
    pub samples: usize,
    pub scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackLqerProofReport {
    pub kind: &'static str,
    pub group: String,
    pub role: TensorRole,
    pub bits: u8,
    pub requested_rank: usize,
    pub actual_rank: usize,
    pub source_rows: usize,
    pub source_cols: usize,
    pub proof_rows: usize,
    pub proof_cols: usize,
    pub residual_frobenius_before: f64,
    pub residual_frobenius_after: f64,
    pub lower_rank_residual_frobenius: Option<f64>,
    pub residual_reduction_pct: f64,
    pub selected_rank_no_worse_than_lower_rank: bool,
    pub activation_l2_norm: f64,
    pub ce_linf_bound_before: f64,
    pub ce_linf_bound_after: f64,
    pub ce_bound_reduction_pct: f64,
    pub finite: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackArtifactProofReport {
    pub kind: &'static str,
    pub artifact_path: String,
    pub artifact_bytes: usize,
    pub target_artifact_bytes: usize,
    pub artifact_budget_ok: bool,
    pub strict_reload_ok: bool,
    pub variant_fingerprint: String,
    pub smoke_tokens: usize,
    pub pre_export_loss: f64,
    pub post_reload_loss: f64,
    pub loss_delta_abs: f64,
    pub finite_loss: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeasuredExperimentReport {
    pub kind: &'static str,
    pub protocol: &'static str,
    pub source_spec_path: String,
    pub source_spec_name: String,
    pub eval_model_shape: String,
    pub eval_tokens: usize,
    pub target_plan_name: &'static str,
    pub rows: Vec<PackMeasuredExperimentRowReport>,
    pub best_measured_bpb_proxy_plan: Option<String>,
    pub target_beats_uniform_q4_bpb_proxy: Option<bool>,
    pub target_beats_mixed_no_lqer_bpb_proxy: Option<bool>,
    pub target_beats_random_lqer_control_bpb_proxy: Option<bool>,
    pub status: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeasuredExperimentRowReport {
    pub plan_name: String,
    pub artifact_bytes: usize,
    pub byte_delta_vs_target: isize,
    pub fits_target_artifact_budget: bool,
    pub quant_bits: String,
    pub lqer_enabled: bool,
    pub lqer_rank: usize,
    pub lqer_top_k: usize,
    pub lqer_selection_policy: String,
    pub lqer_groups: Vec<String>,
    pub pre_export_loss: f64,
    pub post_reload_loss: f64,
    pub measured_bpb_proxy: f64,
    pub loss_delta_abs: f64,
    pub local_decode_tokens_per_second: f64,
    pub finite: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOptionReport {
    pub group: String,
    pub role: TensorRole,
    pub bits: u8,
    pub bytes: usize,
    pub weight_bytes: usize,
    pub scale_bytes: usize,
    pub lqer_bytes: usize,
    pub estimated_quality_loss: f64,
    pub runtime_kernel: String,
    pub residual_correction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LqerCandidateReport {
    pub group: String,
    pub role: TensorRole,
    pub rank: usize,
    pub effective_rank: usize,
    pub bytes_added: usize,
    pub predicted_loss_reduction: f64,
    pub quality_per_byte: f64,
    pub selected_in_pack_plan: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePlannerReport {
    pub kind: &'static str,
    pub algorithm: &'static str,
    pub context_tokens: usize,
    pub batch_sequences: usize,
    pub fp16_cache_bytes: usize,
    pub target_cache_bytes: usize,
    pub feasible: bool,
    pub selected: CachePolicyReport,
    pub candidates: Vec<CachePolicyReport>,
    pub residual_sketch_policy: CacheResidualSketchPolicyReport,
    pub eviction_policy: CacheEvictionPolicyReport,
    pub long_context_eval: CacheLongContextEvalReport,
    pub selected_policy_proof: CachePolicyProofReport,
    pub bound: &'static str,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicyReport {
    pub k_bits: u8,
    pub v_bits: u8,
    pub k_quantization: &'static str,
    pub v_quantization: &'static str,
    pub block_size_tokens: usize,
    pub layout: &'static str,
    pub estimated_cache_bytes: usize,
    pub estimated_attention_error_bound: f64,
    pub predicted_attention_speedup: f64,
    pub budget_ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheResidualSketchPolicyReport {
    pub kind: &'static str,
    pub enabled: bool,
    pub trigger_attention_error_bound: f64,
    pub protected_recent_tokens: usize,
    pub sketch_rank: usize,
    pub sketch_bits: u8,
    pub estimated_residual_bytes: usize,
    pub estimated_bound_reduction_pct: f64,
    pub selected_reason: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEvictionPolicyReport {
    pub kind: &'static str,
    pub policy: &'static str,
    pub protected_recent_tokens: usize,
    pub sink_tokens: usize,
    pub evictable_tokens: usize,
    pub estimated_evicted_bytes_at_budget: usize,
    pub eviction_needed_for_budget: bool,
    pub score_components: Vec<String>,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLongContextEvalReport {
    pub kind: &'static str,
    pub rows: Vec<CacheLongContextEvalRowReport>,
    pub selected_context_fits_budget: bool,
    pub max_context_tokens_under_budget: usize,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLongContextEvalRowReport {
    pub context_tokens: usize,
    pub fp16_cache_bytes: usize,
    pub selected_policy_cache_bytes: usize,
    pub residual_sketch_bytes: usize,
    pub total_cache_bytes: usize,
    pub memory_reduction_vs_fp16_pct: f64,
    pub fits_target_cache_budget: bool,
    pub estimated_attention_error_bound: f64,
    pub predicted_attention_speedup: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicyProofReport {
    pub kind: &'static str,
    pub proof_tokens: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub layout: &'static str,
    pub k_bits: u8,
    pub v_bits: u8,
    pub block_size_tokens: usize,
    pub packed_cache_bytes_single_layer: usize,
    pub fp16_cache_bytes_single_layer: usize,
    pub compression_ratio_single_layer: f64,
    pub max_key_l2_error: f64,
    pub max_value_l2_error: f64,
    pub max_value_l2_norm: f64,
    pub max_query_l2_norm: f64,
    pub max_attention_error_bound: f64,
    pub max_observed_attention_l2_error: f64,
    pub max_dequant_attention_parity_l2_error: f64,
    pub bound_covers_observed_error: bool,
    pub dequant_attention_parity_ok: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGolfKvExperimentReport {
    pub kind: &'static str,
    pub protocol: &'static str,
    pub source_spec_name: String,
    pub target_cache_budget_bytes: usize,
    pub target_cache_budget_pct_of_fp16: f64,
    pub proof_tokens: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size_tokens: usize,
    pub layout: &'static str,
    pub rows: Vec<CacheGolfKvExperimentRowReport>,
    pub long_context_memory: CacheGolfKvExperimentLongContextReport,
    pub best_observed_error_plan: Option<String>,
    pub best_bound_tightness_plan: Option<String>,
    pub best_budgeted_plan: Option<String>,
    pub all_bounds_cover_observed_error: bool,
    pub status: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGolfKvExperimentRowReport {
    pub plan_name: String,
    pub k_bits: u8,
    pub v_bits: u8,
    pub packed_cache_bytes_single_layer: usize,
    pub fp16_cache_bytes_single_layer: usize,
    pub compression_ratio_single_layer: f64,
    pub max_key_l2_error: f64,
    pub max_value_l2_error: f64,
    pub max_value_l2_norm: f64,
    pub max_query_l2_norm: f64,
    pub max_attention_error_bound: f64,
    pub max_observed_attention_l2_error: f64,
    pub bound_to_observed_ratio: f64,
    pub max_dequant_attention_parity_l2_error: f64,
    pub bound_covers_observed_error: bool,
    pub dequant_attention_parity_ok: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGolfKvExperimentLongContextReport {
    pub kind: &'static str,
    pub rows: Vec<CacheGolfKvExperimentContextRowReport>,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGolfKvExperimentContextRowReport {
    pub context_tokens: usize,
    pub batch_sequences: usize,
    pub fp16_cache_bytes: usize,
    pub target_cache_budget_bytes: usize,
    pub selected_grid_cache_bytes: usize,
    pub memory_reduction_vs_fp16_pct: f64,
    pub fits_target_cache_budget: bool,
    pub selected_grid_plan_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPlannerReport {
    pub kind: &'static str,
    pub algorithm: &'static str,
    pub delta_budget_bytes: usize,
    pub selected_bytes: usize,
    pub selected_predicted_gain: f64,
    pub selected_deltas: Vec<DeltaOptionReport>,
    pub candidates: Vec<DeltaOptionReport>,
    pub selected_low_rank_proofs: Vec<DeltaLowRankProofReport>,
    pub score_first_legality_audit: DeltaScoreFirstLegalityAuditReport,
    pub domain_evaluation: DeltaDomainEvaluationReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaOptionReport {
    pub name: String,
    pub delta_type: &'static str,
    pub rank: Option<usize>,
    pub bytes: usize,
    pub predicted_domain_gain: f64,
    pub gain_per_byte: f64,
    pub legality: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaLowRankProofReport {
    pub kind: &'static str,
    pub delta_name: String,
    pub delta_type: &'static str,
    pub requested_rank: usize,
    pub actual_rank: usize,
    pub source_rows: usize,
    pub source_cols: usize,
    pub proof_rows: usize,
    pub proof_cols: usize,
    pub factor_a_elems: usize,
    pub factor_b_elems: usize,
    pub top_singular_values: Vec<f64>,
    pub zero_delta_weighted_error: f64,
    pub selected_rank_weighted_error: f64,
    pub lower_rank_weighted_error: Option<f64>,
    pub error_reduction_vs_zero_pct: f64,
    pub selected_rank_no_worse_than_lower_rank: bool,
    pub finite: bool,
    pub curvature_positive: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaScoreFirstLegalityAuditReport {
    pub kind: &'static str,
    pub score_first_required: bool,
    pub spec_declares_score_first_legal: bool,
    pub selected_delta_count: usize,
    pub artifact_paid_delta_count: usize,
    pub score_first_or_train_only_delta_count: usize,
    pub illegal_delta_count: usize,
    pub pass: bool,
    pub notes: Vec<String>,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaDomainEvaluationReport {
    pub kind: &'static str,
    pub comparison_budget_bytes: usize,
    pub selected_plan_name: &'static str,
    pub rows: Vec<DeltaDomainEvaluationRowReport>,
    pub best_proxy_plan: Option<String>,
    pub selected_beats_no_delta_proxy: Option<bool>,
    pub selected_beats_best_single_proxy: Option<bool>,
    pub selected_beats_static_control_proxy: Option<bool>,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaDomainEvaluationRowReport {
    pub plan_name: String,
    pub artifact_bytes: usize,
    pub byte_delta_vs_selected: isize,
    pub fits_comparison_budget: bool,
    pub delta_count: usize,
    pub predicted_domain_gain: f64,
    pub estimated_domain_loss_proxy: f64,
    pub estimated_bpb_delta_proxy: f64,
    pub low_rank_weighted_error_proxy: Option<f64>,
    pub score_first_legal: bool,
    pub legality_notes: Vec<String>,
    pub delta_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainPlannerReport {
    pub kind: &'static str,
    pub objective: &'static str,
    pub quantization_distance_proxy: f64,
    pub gradient_norm_target: f64,
    pub smoothness_proxy: f64,
    pub export_gap_upper_bound: f64,
    pub late_qat_threshold: f32,
    pub compression_entropy_proxy_weight: f64,
    pub residual_byte_cost_weight: f64,
    pub stages: Vec<TrainStageReport>,
    pub local_proof: TrainGolfLocalProofReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStageReport {
    pub name: &'static str,
    pub trigger: String,
    pub purpose: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainGolfLocalProofReport {
    pub kind: &'static str,
    pub weights: usize,
    pub bits: u8,
    pub block_size: usize,
    pub lambda: f64,
    pub step_size: f64,
    pub distance_sq_before: f64,
    pub distance_sq_after_fixed_projection_step: f64,
    pub distance_reduction_pct: f64,
    pub regularization_loss: f64,
    pub composed_objective_loss: f64,
    pub quadratic_full_loss: f64,
    pub quadratic_projected_loss: f64,
    pub observed_export_gap: f64,
    pub export_gap_bound: f64,
    pub export_gap_bound_covers_observed: bool,
    pub gradient_matches_stop_gradient_objective: bool,
    pub finite: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerCommReport {
    pub kind: &'static str,
    pub compiler_boundary: &'static str,
    pub train_backend: String,
    pub distributed_optimizer_backend: String,
    pub world_size: usize,
    pub optimizer_sharded: bool,
    pub nccl_overlap_mode: String,
    pub bank_payload_bytes_f32: usize,
    pub replicated_all_reduce_wire_bytes_per_rank: usize,
    pub sharded_reduce_scatter_wire_bytes_per_rank: usize,
    pub sharded_param_all_gather_wire_bytes_per_rank: usize,
    pub bf16_shadow_all_gather_wire_bytes_per_rank: usize,
    pub sharded_total_wire_bytes_per_rank: usize,
    pub owned_optimizer_state_reduction_x: f64,
    pub parameter_all_gather_required: bool,
    pub bf16_shadow_all_gather_requested: bool,
    pub local_graph_requested: bool,
    pub pre_norm_graph_requested: bool,
    pub fused_global_clip_requested: bool,
    pub parallel_local_requested: bool,
    pub shard_separable_update_contract: bool,
    pub local_proof: OptimizerCommLocalProofReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerCommLocalProofReport {
    pub kind: &'static str,
    pub proof_world_size: usize,
    pub proof_parameter_elems: usize,
    pub learning_rate: f64,
    pub shard_ranges: Vec<OptimizerCommShardRangeReport>,
    pub replicated_checksum: f64,
    pub sharded_checksum: f64,
    pub max_abs_diff: f64,
    pub exact_equivalence: bool,
    pub finite: bool,
    pub assumption: &'static str,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerCommShardRangeReport {
    pub rank: usize,
    pub start: usize,
    pub end: usize,
    pub elements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelForgeReport {
    pub kind: &'static str,
    pub compiler_boundary: &'static str,
    pub exact_tiled_ce: ExactTiledCeReport,
    pub fusion_primitives: Vec<FusionPrimitiveReport>,
    pub equivalence_tests: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactTiledCeReport {
    pub backend_requested: String,
    pub exact_without_persistent_logits: bool,
    pub rows_m: usize,
    pub vocab_v: usize,
    pub hidden_d: usize,
    pub tile_t: usize,
    pub full_logits_bytes_f32: usize,
    pub tiled_logits_scratch_bytes_f32: usize,
    pub row_stats_scratch_bytes_f32: usize,
    pub tiled_scratch_bytes_estimate: usize,
    pub scratch_reduction_x: f64,
    pub gradient_formula: &'static str,
    pub local_proof: ExactTiledCeProofReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactTiledCeProofReport {
    pub kind: &'static str,
    pub proof_rows_m: usize,
    pub proof_vocab_v: usize,
    pub proof_hidden_d: usize,
    pub tile_t: usize,
    pub softcap_pos: f64,
    pub softcap_neg: f64,
    pub loss_scale: f64,
    pub full_logits_bytes_f32: usize,
    pub tiled_logits_scratch_bytes_f32: usize,
    pub row_stats_scratch_bytes_f32: usize,
    pub tiled_scratch_bytes_estimate: usize,
    pub scratch_reduction_x: f64,
    pub max_loss_abs_diff: f64,
    pub max_d_hidden_abs_diff: f64,
    pub max_d_weight_abs_diff: f64,
    pub parity_ok: bool,
    pub finite: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelForgeCeExperimentReport {
    pub kind: &'static str,
    pub protocol: &'static str,
    pub source_spec_name: String,
    pub rows_m: usize,
    pub vocab_v: usize,
    pub hidden_d: usize,
    pub tile_t: usize,
    pub rows: Vec<KernelForgeCeExperimentRowReport>,
    pub full_vs_tiled: KernelForgeCeExperimentComparisonReport,
    pub status: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelForgeCeExperimentRowReport {
    pub plan_name: String,
    pub materializes_persistent_logits: bool,
    pub logits_scratch_bytes_f32: usize,
    pub backward_grad_logits_bytes_f32: usize,
    pub total_ce_scratch_bytes_f32: usize,
    pub memory_reduction_vs_full_x: f64,
    pub forward_ms_cpu: f64,
    pub backward_ms_cpu: f64,
    pub total_ms_cpu: f64,
    pub loss_mean: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelForgeCeExperimentComparisonReport {
    pub max_loss_abs_diff: f64,
    pub max_d_hidden_abs_diff: f64,
    pub max_d_weight_abs_diff: f64,
    pub parity_ok: bool,
    pub tiled_uses_less_scratch: bool,
    pub tiled_cpu_speedup_vs_full: f64,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPrimitiveReport {
    pub name: &'static str,
    pub inputs: Vec<&'static str>,
    pub outputs: Vec<&'static str>,
    pub status: String,
    pub proof_obligation: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfWindReport {
    pub kind: &'static str,
    pub estimate_only: bool,
    pub train_step_ms_estimate: f64,
    pub expected_steps_in_600s: usize,
    pub expected_train_wall_seconds: f64,
    pub artifact_bytes_estimate: usize,
    pub top_bottlenecks: Vec<String>,
    pub pareto_candidates: Vec<WindParetoCandidate>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindParetoCandidate {
    pub name: String,
    pub expected_delta_ms_per_step: f64,
    pub expected_delta_bytes: isize,
    pub evidence_required: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfWindExperimentReport {
    pub kind: &'static str,
    pub protocol: &'static str,
    pub source_spec_name: String,
    pub candidate_count: usize,
    pub cheap_ranked_count: usize,
    pub full_evaluated_count: usize,
    pub full_eval_top_k: usize,
    pub holdout_spearman: Option<f64>,
    pub mean_abs_rank_error: Option<f64>,
    pub calibration_pass: bool,
    pub top_3_overlap_count: usize,
    pub best_cheap_candidate: Option<String>,
    pub best_full_candidate: Option<String>,
    pub rows: Vec<ModelGolfWindExperimentRowReport>,
    pub status: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfWindExperimentRowReport {
    pub candidate_name: String,
    pub cheap_rank: usize,
    pub evaluated_cheap_rank: Option<usize>,
    pub full_rank: Option<usize>,
    pub quant_bits: String,
    pub lqer_enabled: bool,
    pub lqer_rank: usize,
    pub lqer_top_k: usize,
    pub cache_k_bits: u8,
    pub cache_v_bits: u8,
    pub delta_budget_bytes: usize,
    pub estimated_artifact_bytes: usize,
    pub estimated_artifact_budget_fit: bool,
    pub estimated_cache_bytes: usize,
    pub estimated_memory_budget_fit: Option<bool>,
    pub estimated_delta_bytes: usize,
    pub cheap_score: f64,
    pub cheap_quality_loss: f64,
    pub cheap_cache_error_proxy: f64,
    pub cheap_delta_gain: f64,
    pub full_evaluated: bool,
    pub full_eval_artifact_bytes: Option<usize>,
    pub full_eval_bpb_proxy: Option<f64>,
    pub full_eval_score: Option<f64>,
    pub full_eval_loss_delta_abs: Option<f64>,
    pub full_eval_decode_tokens_per_second: Option<f64>,
    pub full_eval_lqer_groups: Vec<String>,
    pub rank_error: Option<isize>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleGolfReport {
    pub kind: &'static str,
    pub thesis: &'static str,
    pub current_contract_track: String,
    pub shared_compiler_surfaces: Vec<&'static str>,
    pub tracks: Vec<ScaleGolfTrackReport>,
    pub invariant: ScaleGolfInvariantReport,
    pub local_ready: bool,
    pub release_ready: bool,
    pub next_action: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleGolfTrackReport {
    pub track: &'static str,
    pub target: &'static str,
    pub primary_constraint: &'static str,
    pub hardware_scope: String,
    pub artifact_budget_bytes: Option<usize>,
    pub memory_budget_bytes: Option<usize>,
    pub context_tokens: usize,
    pub training_time_budget_seconds: Option<f64>,
    pub train_world_size: usize,
    pub uses_surfaces: Vec<&'static str>,
    pub required_modules: Vec<&'static str>,
    pub local_ready: bool,
    pub release_ready: bool,
    pub quality_per_resource_proxy: f64,
    pub recommended_next_action: String,
    pub remote_evidence_needed: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleGolfInvariantReport {
    pub all_tracks_use_resource_contract: bool,
    pub all_tracks_use_model_ir: bool,
    pub all_tracks_use_artifact_compiler: bool,
    pub all_tracks_use_runtime_planner: bool,
    pub all_tracks_use_evaluator: bool,
    pub all_tracks_use_cost_model: bool,
    pub track_count: usize,
    pub expected_track_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformModuleStatus {
    pub module: &'static str,
    pub implemented_surface: &'static str,
    pub remaining_remote_evidence: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfReleaseReadinessReport {
    pub kind: &'static str,
    pub release_ready: bool,
    pub local_planner_ready: bool,
    pub release_evidence_source: Option<String>,
    pub blocker_count: usize,
    pub checklist: Vec<ModelGolfReleaseRequirementReport>,
    pub next_action: String,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfReleaseSourceReport {
    pub kind: &'static str,
    pub pillar: &'static str,
    pub spec_name: String,
    pub spec_fingerprint: String,
    pub hardware: String,
    pub runtime: String,
    pub generated_at: String,
    pub claims: serde_json::Value,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGolfReleaseRequirementReport {
    pub pillar: &'static str,
    pub requirement: &'static str,
    pub status: &'static str,
    pub blocking: bool,
    pub current_evidence: String,
    pub evidence_needed: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfReleaseEvidenceFile {
    kind: Option<String>,
    spec_name: Option<String>,
    spec_fingerprint: Option<String>,
    hardware: Option<String>,
    runtime: Option<String>,
    evidence_id: Option<String>,
    generated_at: Option<String>,
    source_reports: Option<Vec<ModelGolfReleaseSourceReportEvidence>>,
    pack: Option<ModelGolfPackReleaseEvidence>,
    lqer: Option<ModelGolfLqerReleaseEvidence>,
    cache: Option<ModelGolfCacheReleaseEvidence>,
    delta: Option<ModelGolfDeltaReleaseEvidence>,
    train: Option<ModelGolfTrainReleaseEvidence>,
    resource_cost: Option<ModelGolfResourceCostReleaseEvidence>,
    optimizer_comm: Option<ModelGolfOptimizerCommReleaseEvidence>,
    kernel_forge: Option<ModelGolfKernelReleaseEvidence>,
    wind_tunnel: Option<ModelGolfWindReleaseEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelGolfReleaseSourceReportEvidence {
    pillar: String,
    kind: String,
    path: String,
    sha256: String,
}

#[derive(Debug, Clone)]
struct ModelGolfSourceReportValidation {
    valid: bool,
    details: String,
    claims_by_pillar: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfPackReleaseEvidence {
    artifact_path: Option<String>,
    artifact_sha256: Option<String>,
    validation_dataset_id: Option<String>,
    validation_command: Option<String>,
    artifact_bytes: Option<usize>,
    strict_reload_pass: Option<bool>,
    heldout_bpb: Option<f64>,
    baseline_bpb: Option<f64>,
    relative_bpb_increase_pct: Option<f64>,
    decode_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfLqerReleaseEvidence {
    calibration_dataset_id: Option<String>,
    selected_lqer_group_names: Option<Vec<String>>,
    production_svd_validated: Option<bool>,
    calibrated_tensor_sensitivity: Option<bool>,
    equal_byte_bpb_delta: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfCacheReleaseEvidence {
    backend_id: Option<String>,
    kernel_id: Option<String>,
    long_context_dataset_id: Option<String>,
    k_bits: Option<u8>,
    v_bits: Option<u8>,
    block_size_tokens: Option<usize>,
    layout: Option<String>,
    context_tokens: Option<usize>,
    batch_sequences: Option<usize>,
    fused_runtime: Option<bool>,
    parity_pass: Option<bool>,
    memory_budget_fit: Option<bool>,
    long_context_bpb_delta_pct: Option<f64>,
    speedup_x: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfDeltaReleaseEvidence {
    domain_dataset_id: Option<String>,
    selected_delta_names: Option<Vec<String>>,
    trained_delta_bytes: Option<usize>,
    legality_pass: Option<bool>,
    score_first_trace_or_review: Option<bool>,
    equal_byte_domain_bpb_delta: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfTrainReleaseEvidence {
    training_run_id: Option<String>,
    backend_id: Option<String>,
    gpu_backend_integrated: Option<bool>,
    proxy_calibrated: Option<bool>,
    post_export_bpb_delta_vs_posthoc: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfResourceCostReleaseEvidence {
    measurement_run_id: Option<String>,
    power_meter_id: Option<String>,
    wall_time_seconds: Option<f64>,
    average_power_watts: Option<f64>,
    energy_joules: Option<f64>,
    telemetry_validated: Option<bool>,
    wall_time_budget_seconds: Option<f64>,
    energy_budget_joules: Option<f64>,
    wall_time_budget_fit: Option<bool>,
    energy_budget_fit: Option<bool>,
    energy_consistency_error_pct: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfOptimizerCommReleaseEvidence {
    distributed_backend_id: Option<String>,
    world_size: Option<usize>,
    optimizer_sharded: Option<bool>,
    nccl_overlap_mode: Option<String>,
    sharded_total_wire_bytes_per_rank: Option<usize>,
    owned_optimizer_state_reduction_x: Option<f64>,
    reduce_scatter_parity_pass: Option<bool>,
    all_gather_parity_pass: Option<bool>,
    optimizer_update_parity_pass: Option<bool>,
    nccl_trace_validated: Option<bool>,
    overlap_validated: Option<bool>,
    measured_comm_time_ms: Option<f64>,
    measured_step_time_ms: Option<f64>,
    communication_speedup_x: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfKernelReleaseEvidence {
    backend_id: Option<String>,
    generated_kernel_ids: Option<Vec<String>>,
    tile_t: Option<usize>,
    generated_kernels: Option<bool>,
    parity_pass: Option<bool>,
    speedup_x: Option<f64>,
    memory_reduction_x: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGolfWindReleaseEvidence {
    trace_corpus_id: Option<String>,
    calibration_report_id: Option<String>,
    fresh_profiler_traces: Option<bool>,
    external_timing_validated: Option<bool>,
    holdout_spearman: Option<f64>,
    mean_abs_pct_error: Option<f64>,
}

#[derive(Debug, Clone)]
struct DpState {
    loss: f64,
    bytes: usize,
    option_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DeltaDpState {
    gain: f64,
    bytes: usize,
    option_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PackQualityCalibrationModel {
    report: PackQualityCalibrationReport,
    group_scales: BTreeMap<(String, u8, bool), f64>,
    role_scales: BTreeMap<(TensorRole, u8, bool), f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PackQualityCalibrationFile {
    points: Vec<PackQualityCalibrationPoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct PackQualityCalibrationPoint {
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    role: Option<TensorRole>,
    bits: u8,
    #[serde(default)]
    lqer: bool,
    #[serde(default)]
    estimated_quality_loss: Option<f64>,
    #[serde(
        default,
        alias = "measured_loss_delta",
        alias = "observed_quality_loss",
        alias = "quality_loss"
    )]
    measured_quality_loss: Option<f64>,
}

#[derive(Debug, Clone)]
struct CalibrationPointEval {
    group: Option<String>,
    role: TensorRole,
    bits: u8,
    lqer: bool,
    estimated: f64,
    measured: f64,
}

#[derive(Debug, Clone, Default)]
struct CalibrationAccumulator {
    samples: usize,
    scale_sum: f64,
}

fn required_release_string(value: &Option<String>, flag: &str) -> PgResult<String> {
    let Some(value) = value.as_ref() else {
        return Err(PgError::InvalidOp(format!("{flag} is required")));
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(PgError::InvalidOp(format!("{flag} must be non-empty")))
    } else {
        Ok(trimmed.to_string())
    }
}

fn required_release_positive_f64(value: Option<f64>, flag: &str) -> PgResult<f64> {
    let Some(value) = value else {
        return Err(PgError::InvalidOp(format!("{flag} is required")));
    };
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PgError::InvalidOp(format!(
            "{flag} must be finite and positive"
        )))
    }
}

fn required_release_finite_f64(value: Option<f64>, flag: &str) -> PgResult<f64> {
    let Some(value) = value else {
        return Err(PgError::InvalidOp(format!("{flag} is required")));
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PgError::InvalidOp(format!("{flag} must be finite")))
    }
}

fn required_release_f64_at_least(value: Option<f64>, flag: &str, minimum: f64) -> PgResult<f64> {
    let value = required_release_finite_f64(value, flag)?;
    if value >= minimum {
        Ok(value)
    } else {
        Err(PgError::InvalidOp(format!(
            "{flag} must be at least {minimum}"
        )))
    }
}

fn required_release_f64_at_most(value: Option<f64>, flag: &str, maximum: f64) -> PgResult<f64> {
    let value = required_release_finite_f64(value, flag)?;
    if value <= maximum {
        Ok(value)
    } else {
        Err(PgError::InvalidOp(format!(
            "{flag} must be at most {maximum}"
        )))
    }
}

fn finite_relative_error_pct(actual: f64, expected: f64) -> PgResult<f64> {
    if !actual.is_finite() || !expected.is_finite() || actual <= 0.0 {
        return Err(PgError::InvalidOp(
            "relative error inputs must be finite and actual must be positive".to_string(),
        ));
    }
    Ok(100.0 * (actual - expected).abs() / actual.abs().max(1e-9))
}

fn required_release_true(value: Option<bool>, flag: &str) -> PgResult<bool> {
    match value {
        Some(true) => Ok(true),
        Some(false) => Err(PgError::InvalidOp(format!("{flag} must be true"))),
        None => Err(PgError::InvalidOp(format!("{flag} is required"))),
    }
}

fn required_release_csv_strings(value: &Option<String>, flag: &str) -> PgResult<Vec<String>> {
    let raw = required_release_string(value, flag)?;
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if values.is_empty() {
        Err(PgError::InvalidOp(format!(
            "{flag} must contain at least one non-empty id"
        )))
    } else {
        Ok(values)
    }
}

fn modelgolf_generated_at_unix() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix:{seconds}")
}

fn raw_evidence_claim_value_matches(
    actual: Option<&serde_json::Value>,
    expected: &serde_json::Value,
) -> bool {
    let Some(actual) = actual else {
        return false;
    };
    match (actual.as_f64(), expected.as_f64()) {
        (Some(actual), Some(expected)) => {
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-9
        }
        _ => actual == expected,
    }
}

fn validate_raw_evidence_packet(
    raw: &str,
    pillar: &str,
    expected_claims: &serde_json::Value,
) -> PgResult<usize> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|err| {
        PgError::InvalidOp(format!(
            "--evidence-source must be a JSON modelgolf_raw_evidence packet: {err}"
        ))
    })?;
    if value.get("kind").and_then(serde_json::Value::as_str) != Some("modelgolf_raw_evidence") {
        return Err(PgError::InvalidOp(
            "--evidence-source kind must be modelgolf_raw_evidence".to_string(),
        ));
    }
    if value.get("pillar").and_then(serde_json::Value::as_str) != Some(pillar) {
        return Err(PgError::InvalidOp(format!(
            "--evidence-source pillar must be {pillar}"
        )));
    }
    let claims = value
        .get("claims")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            PgError::InvalidOp("--evidence-source claims object is required".to_string())
        })?;
    let expected = expected_claims.as_object().ok_or_else(|| {
        PgError::InvalidOp("internal raw evidence expected claims must be an object".to_string())
    })?;
    for (key, expected_value) in expected {
        if !raw_evidence_claim_value_matches(claims.get(key), expected_value) {
            return Err(PgError::InvalidOp(format!(
                "--evidence-source claims.{key} does not match the source-report claim"
            )));
        }
    }
    Ok(expected.len())
}

fn modelgolf_raw_claim_keys_for_pillar(pillar: &str) -> &'static [&'static str] {
    match pillar {
        "pack" => &[
            "artifact_sha256",
            "validation_dataset_id",
            "validation_command",
            "artifact_bytes",
            "strict_reload_pass",
            "heldout_bpb",
            "baseline_bpb",
            "relative_bpb_increase_pct",
            "decode_tokens_per_second",
        ],
        "lqer" => &[
            "calibration_dataset_id",
            "selected_lqer_group_names",
            "production_svd_validated",
            "calibrated_tensor_sensitivity",
            "equal_byte_bpb_delta",
        ],
        "cache" => &[
            "backend_id",
            "kernel_id",
            "long_context_dataset_id",
            "k_bits",
            "v_bits",
            "block_size_tokens",
            "layout",
            "context_tokens",
            "batch_sequences",
            "fused_runtime",
            "parity_pass",
            "memory_budget_fit",
            "long_context_bpb_delta_pct",
            "speedup_x",
        ],
        "delta" => &[
            "domain_dataset_id",
            "selected_delta_names",
            "trained_delta_bytes",
            "legality_pass",
            "score_first_trace_or_review",
            "equal_byte_domain_bpb_delta",
        ],
        "train" => &[
            "training_run_id",
            "backend_id",
            "gpu_backend_integrated",
            "proxy_calibrated",
            "post_export_bpb_delta_vs_posthoc",
        ],
        "resource_cost" => &[
            "measurement_run_id",
            "power_meter_id",
            "wall_time_seconds",
            "average_power_watts",
            "energy_joules",
            "telemetry_validated",
            "wall_time_budget_seconds",
            "energy_budget_joules",
            "wall_time_budget_fit",
            "energy_budget_fit",
            "energy_consistency_error_pct",
        ],
        "optimizer_comm" => &[
            "distributed_backend_id",
            "world_size",
            "optimizer_sharded",
            "nccl_overlap_mode",
            "sharded_total_wire_bytes_per_rank",
            "owned_optimizer_state_reduction_x",
            "reduce_scatter_parity_pass",
            "all_gather_parity_pass",
            "optimizer_update_parity_pass",
            "nccl_trace_validated",
            "overlap_validated",
            "measured_comm_time_ms",
            "measured_step_time_ms",
            "communication_speedup_x",
        ],
        "kernel_forge" => &[
            "backend_id",
            "generated_kernel_ids",
            "tile_t",
            "generated_kernels",
            "parity_pass",
            "speedup_x",
            "memory_reduction_x",
        ],
        "wind_tunnel" => &[
            "trace_corpus_id",
            "calibration_report_id",
            "fresh_profiler_traces",
            "external_timing_validated",
            "holdout_spearman",
            "mean_abs_pct_error",
        ],
        _ => &[],
    }
}

fn required_raw_evidence_claim(
    options: &ModelGolfOptions,
    pillar: &str,
    expected_claims: &serde_json::Value,
) -> PgResult<serde_json::Value> {
    let evidence_source = options.evidence_source.as_deref().ok_or_else(|| {
        PgError::InvalidOp("--evidence-source is required for release source reports".to_string())
    })?;
    let metadata = std::fs::metadata(evidence_source).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to read --evidence-source {}: {err}",
            evidence_source.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(PgError::InvalidOp(format!(
            "--evidence-source {} must be a regular file",
            evidence_source.display()
        )));
    }
    let bytes = metadata.len();
    if bytes == 0 {
        return Err(PgError::InvalidOp(format!(
            "--evidence-source {} is empty",
            evidence_source.display()
        )));
    }
    let canonical_path =
        std::fs::canonicalize(evidence_source).unwrap_or_else(|_| evidence_source.to_path_buf());
    let raw = std::fs::read_to_string(&canonical_path).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to read --evidence-source {} as UTF-8 JSON: {err}",
            canonical_path.display()
        ))
    })?;
    let validated_claim_count = validate_raw_evidence_packet(&raw, pillar, expected_claims)?;
    Ok(serde_json::json!({
        "path": canonical_path.display().to_string(),
        "sha256": format!("sha256:{}", modelgolf_sha256_file(&canonical_path)?),
        "bytes": bytes,
        "semantic_kind": "modelgolf_raw_evidence",
        "validated_claim_count": validated_claim_count,
    }))
}

fn run_modelgolf_pack_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    artifact_budget_bytes: usize,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let artifact_path = options.release_artifact.as_deref().ok_or_else(|| {
        PgError::InvalidOp("--release-artifact is required for pack-source-report".to_string())
    })?;
    let validation_dataset_id =
        required_release_string(&options.validation_dataset_id, "--validation-dataset-id")?;
    let validation_command =
        required_release_string(&options.validation_command, "--validation-command")?;
    let heldout_bpb = required_release_positive_f64(options.heldout_bpb, "--heldout-bpb")?;
    let baseline_bpb = required_release_positive_f64(options.baseline_bpb, "--baseline-bpb")?;
    let decode_tokens_per_second = required_release_positive_f64(
        options.decode_tokens_per_second,
        "--decode-tokens-per-second",
    )?;
    let artifact_bytes = std::fs::metadata(artifact_path)?.len() as usize;
    if artifact_bytes == 0 {
        return Err(PgError::InvalidOp(format!(
            "--release-artifact {} is empty",
            artifact_path.display()
        )));
    }
    if artifact_bytes > artifact_budget_bytes {
        return Err(PgError::InvalidOp(format!(
            "--release-artifact {} is {artifact_bytes} bytes, above artifact budget {artifact_budget_bytes}",
            artifact_path.display()
        )));
    }
    let artifact_sha256 = format!("sha256:{}", modelgolf_sha256_file(artifact_path)?);
    let mut loaded = GptModel::new(spec.model.to_model_config());
    pg_quant::export::load_artifact_with_spec(artifact_path, &mut loaded, &spec.quant, true)?;
    let relative_bpb_increase_pct = 100.0 * (heldout_bpb - baseline_bpb) / baseline_bpb;
    if !relative_bpb_increase_pct.is_finite() {
        return Err(PgError::InvalidOp(
            "relative BPB increase is not finite".to_string(),
        ));
    }
    if !release_quality_within_budget(
        Some(heldout_bpb),
        Some(baseline_bpb),
        Some(relative_bpb_increase_pct),
        resource_contract.quality_budget_ppl_pct,
    ) {
        return Err(PgError::InvalidOp(format!(
            "heldout BPB {heldout_bpb} exceeds quality budget {:?} against baseline {baseline_bpb}",
            resource_contract.quality_budget_ppl_pct
        )));
    }
    let expected_raw_claims = serde_json::json!({
        "artifact_sha256": artifact_sha256,
        "validation_dataset_id": validation_dataset_id,
        "validation_command": validation_command,
        "artifact_bytes": artifact_bytes,
        "strict_reload_pass": true,
        "heldout_bpb": heldout_bpb,
        "baseline_bpb": baseline_bpb,
        "relative_bpb_increase_pct": relative_bpb_increase_pct,
        "decode_tokens_per_second": decode_tokens_per_second,
    });
    let raw_evidence = required_raw_evidence_claim(options, "pack", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "artifact_path": artifact_path.display().to_string(),
        "artifact_sha256": artifact_sha256,
        "validation_dataset_id": validation_dataset_id,
        "validation_command": validation_command,
        "artifact_bytes": artifact_bytes,
        "artifact_budget_bytes": artifact_budget_bytes,
        "strict_reload_pass": true,
        "heldout_bpb": heldout_bpb,
        "baseline_bpb": baseline_bpb,
        "relative_bpb_increase_pct": relative_bpb_increase_pct,
        "quality_budget_ppl_pct": resource_contract.quality_budget_ppl_pct,
        "decode_tokens_per_second": decode_tokens_per_second,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "pack",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "Pack release source report binds caller-supplied held-out BPB/decode metrics to a SHA-256 verified strict-reload artifact; the command verifies bytes, strict reload, finite metrics, and quality budget but does not create the held-out dataset itself",
    })
}

fn run_modelgolf_cache_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    cache: &CachePlannerReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let backend_id = required_release_string(&options.backend_id, "--backend-id")?;
    let kernel_id = required_release_string(&options.kernel_id, "--kernel-id")?;
    let long_context_dataset_id = required_release_string(
        &options.long_context_dataset_id,
        "--long-context-dataset-id",
    )?;
    let fused_runtime = required_release_true(options.fused_runtime, "--fused-runtime")?;
    let parity_pass = required_release_true(options.parity_pass, "--parity-pass")?;
    let long_context_bpb_delta_pct = required_release_finite_f64(
        options.long_context_bpb_delta_pct,
        "--long-context-bpb-delta-pct",
    )?;
    if !release_quality_delta_pct_within_budget(
        Some(long_context_bpb_delta_pct),
        resource_contract.quality_budget_ppl_pct,
    ) {
        return Err(PgError::InvalidOp(format!(
            "--long-context-bpb-delta-pct {long_context_bpb_delta_pct} exceeds quality budget {:?}",
            resource_contract.quality_budget_ppl_pct
        )));
    }
    let speedup_x = required_release_f64_at_least(options.speedup_x, "--speedup-x", 1.0)?;
    if !cache.feasible || !cache.selected.budget_ok {
        return Err(PgError::InvalidOp(format!(
            "selected CacheGolf policy does not fit the active memory contract: selected_bytes={}, target_cache_bytes={}",
            cache.selected.estimated_cache_bytes, cache.target_cache_bytes
        )));
    }
    if !cache.selected_policy_proof.bound_covers_observed_error
        || !cache.selected_policy_proof.dequant_attention_parity_ok
    {
        return Err(PgError::InvalidOp(
            "selected CacheGolf local proof did not pass parity/bound checks".to_string(),
        ));
    }
    let expected_raw_claims = serde_json::json!({
        "backend_id": backend_id,
        "kernel_id": kernel_id,
        "long_context_dataset_id": long_context_dataset_id,
        "k_bits": cache.selected.k_bits,
        "v_bits": cache.selected.v_bits,
        "block_size_tokens": cache.selected.block_size_tokens,
        "layout": cache.selected.layout,
        "context_tokens": cache.context_tokens,
        "batch_sequences": cache.batch_sequences,
        "fused_runtime": fused_runtime,
        "parity_pass": parity_pass,
        "memory_budget_fit": true,
        "long_context_bpb_delta_pct": long_context_bpb_delta_pct,
        "speedup_x": speedup_x,
    });
    let raw_evidence = required_raw_evidence_claim(options, "cache", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "backend_id": backend_id,
        "kernel_id": kernel_id,
        "long_context_dataset_id": long_context_dataset_id,
        "k_bits": cache.selected.k_bits,
        "v_bits": cache.selected.v_bits,
        "block_size_tokens": cache.selected.block_size_tokens,
        "layout": cache.selected.layout,
        "context_tokens": cache.context_tokens,
        "batch_sequences": cache.batch_sequences,
        "selected_policy_cache_bytes": cache.selected.estimated_cache_bytes,
        "target_cache_bytes": cache.target_cache_bytes,
        "fused_runtime": fused_runtime,
        "parity_pass": parity_pass,
        "memory_budget_fit": true,
        "long_context_bpb_delta_pct": long_context_bpb_delta_pct,
        "speedup_x": speedup_x,
        "local_bound_covers_observed_error": cache.selected_policy_proof.bound_covers_observed_error,
        "local_dequant_attention_parity_ok": cache.selected_policy_proof.dequant_attention_parity_ok,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "cache",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "CacheGolf release source report binds caller-supplied fused-runtime long-context quality/timing claims to the currently selected KV policy; the command verifies selected policy bytes, local parity/bound proof, finite metrics, quality budget, and speedup threshold but does not run the external fused backend itself",
    })
}

fn run_modelgolf_lqer_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    pack: &PackPlannerReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let calibration_dataset_id =
        required_release_string(&options.calibration_dataset_id, "--calibration-dataset-id")?;
    let production_svd_validated = required_release_true(
        options.production_svd_validated,
        "--production-svd-validated",
    )?;
    let calibrated_tensor_sensitivity = required_release_true(
        options.calibrated_tensor_sensitivity,
        "--calibrated-tensor-sensitivity",
    )?;
    let equal_byte_bpb_delta =
        required_release_f64_at_most(options.equal_byte_bpb_delta, "--equal-byte-bpb-delta", 0.0)?;
    if pack.lqer_candidates.is_empty() {
        return Err(PgError::InvalidOp(
            "current Pack plan has no LQER candidates to bind".to_string(),
        ));
    }
    if pack.lqer_proofs.is_empty() {
        return Err(PgError::InvalidOp(
            "current Pack plan selected no LQER proofs to bind".to_string(),
        ));
    }
    if pack.lqer_proofs.iter().any(|proof| !proof.finite) {
        return Err(PgError::InvalidOp(
            "current Pack plan has a non-finite selected LQER proof".to_string(),
        ));
    }
    let selected_lqer_group_names = pack
        .lqer_proofs
        .iter()
        .map(|proof| proof.group.clone())
        .collect::<Vec<_>>();
    let expected_raw_claims = serde_json::json!({
        "calibration_dataset_id": calibration_dataset_id,
        "selected_lqer_group_names": selected_lqer_group_names,
        "production_svd_validated": production_svd_validated,
        "calibrated_tensor_sensitivity": calibrated_tensor_sensitivity,
        "equal_byte_bpb_delta": equal_byte_bpb_delta,
    });
    let raw_evidence = required_raw_evidence_claim(options, "lqer", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "calibration_dataset_id": calibration_dataset_id,
        "selected_lqer_group_names": selected_lqer_group_names,
        "production_svd_validated": production_svd_validated,
        "calibrated_tensor_sensitivity": calibrated_tensor_sensitivity,
        "equal_byte_bpb_delta": equal_byte_bpb_delta,
        "candidate_count": pack.lqer_candidates.len(),
        "selected_lqer_proof_count": pack.lqer_proofs.len(),
        "quality_calibration_applied": pack.quality_calibration.applied,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "lqer",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "LQER release source report binds caller-supplied production SVD, calibration, and equal-byte quality claims to the current Pack/LQER plan; the command verifies local selected LQER proofs and release thresholds but does not run the external calibration itself",
    })
}

fn run_modelgolf_delta_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    delta: &DeltaPlannerReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let domain_dataset_id =
        required_release_string(&options.domain_dataset_id, "--domain-dataset-id")?;
    let Some(trained_delta_bytes) = options.trained_delta_bytes else {
        return Err(PgError::InvalidOp(
            "--trained-delta-bytes is required".to_string(),
        ));
    };
    if trained_delta_bytes == 0 || trained_delta_bytes > delta.selected_bytes {
        return Err(PgError::InvalidOp(format!(
            "--trained-delta-bytes must be in 1..={} for the current selected delta plan, got {trained_delta_bytes}",
            delta.selected_bytes
        )));
    }
    let legality_pass = required_release_true(options.legality_pass, "--legality-pass")?;
    let score_first_trace_or_review = required_release_true(
        options.score_first_trace_or_review,
        "--score-first-trace-or-review",
    )?;
    let equal_byte_domain_bpb_delta = required_release_f64_at_most(
        options.equal_byte_domain_bpb_delta,
        "--equal-byte-domain-bpb-delta",
        0.0,
    )?;
    if !delta.score_first_legality_audit.pass {
        return Err(PgError::InvalidOp(
            "current DeltaGolf selected plan failed the local score-first legality audit"
                .to_string(),
        ));
    }
    if delta.selected_deltas.is_empty() {
        return Err(PgError::InvalidOp(
            "current DeltaGolf plan selected no trainable/artifact deltas".to_string(),
        ));
    }
    let selected_delta_names = delta
        .selected_deltas
        .iter()
        .map(|delta| delta.name.clone())
        .collect::<Vec<_>>();
    let expected_raw_claims = serde_json::json!({
        "domain_dataset_id": domain_dataset_id,
        "selected_delta_names": selected_delta_names,
        "trained_delta_bytes": trained_delta_bytes,
        "legality_pass": legality_pass,
        "score_first_trace_or_review": score_first_trace_or_review,
        "equal_byte_domain_bpb_delta": equal_byte_domain_bpb_delta,
    });
    let raw_evidence = required_raw_evidence_claim(options, "delta", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "domain_dataset_id": domain_dataset_id,
        "selected_delta_names": selected_delta_names,
        "trained_delta_bytes": trained_delta_bytes,
        "selected_delta_budget_bytes": delta.selected_bytes,
        "legality_pass": legality_pass,
        "score_first_trace_or_review": score_first_trace_or_review,
        "equal_byte_domain_bpb_delta": equal_byte_domain_bpb_delta,
        "local_score_first_legality_audit": delta.score_first_legality_audit.pass,
        "selected_low_rank_proof_count": delta.selected_low_rank_proofs.len(),
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "delta",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "DeltaGolf release source report binds caller-supplied trained delta, legality, and equal-byte domain quality claims to the currently selected delta names/byte budget; the command verifies local legality/proof gates and release thresholds but does not train the domain delta itself",
    })
}

fn run_modelgolf_train_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    train: &TrainPlannerReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let training_run_id = required_release_string(&options.training_run_id, "--training-run-id")?;
    let backend_id = required_release_string(&options.backend_id, "--backend-id")?;
    let gpu_backend_integrated =
        required_release_true(options.gpu_backend_integrated, "--gpu-backend-integrated")?;
    let proxy_calibrated = required_release_true(options.proxy_calibrated, "--proxy-calibrated")?;
    let post_export_bpb_delta_vs_posthoc = required_release_f64_at_most(
        options.post_export_bpb_delta_vs_posthoc,
        "--post-export-bpb-delta-vs-posthoc",
        0.0,
    )?;
    if !train.local_proof.finite || !train.local_proof.export_gap_bound_covers_observed {
        return Err(PgError::InvalidOp(
            "current TrainGolf local regularizer/export-gap proof did not pass".to_string(),
        ));
    }
    let expected_raw_claims = serde_json::json!({
        "training_run_id": training_run_id,
        "backend_id": backend_id,
        "gpu_backend_integrated": gpu_backend_integrated,
        "proxy_calibrated": proxy_calibrated,
        "post_export_bpb_delta_vs_posthoc": post_export_bpb_delta_vs_posthoc,
    });
    let raw_evidence = required_raw_evidence_claim(options, "train", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "training_run_id": training_run_id,
        "backend_id": backend_id,
        "gpu_backend_integrated": gpu_backend_integrated,
        "proxy_calibrated": proxy_calibrated,
        "post_export_bpb_delta_vs_posthoc": post_export_bpb_delta_vs_posthoc,
        "local_regularizer_finite": train.local_proof.finite,
        "local_export_gap_bound_covers_observed": train.local_proof.export_gap_bound_covers_observed,
        "distance_reduction_pct": train.local_proof.distance_reduction_pct,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "train",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "TrainGolf release source report binds caller-supplied backend training, proxy calibration, and post-export quality claims to the current artifact-aware objective; the command verifies local regularizer/export-gap proof and release thresholds but does not run GPU training itself",
    })
}

fn run_modelgolf_resource_cost_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let measurement_run_id =
        required_release_string(&options.measurement_run_id, "--measurement-run-id")?;
    let power_meter_id = required_release_string(&options.power_meter_id, "--power-meter-id")?;
    let wall_time_seconds =
        required_release_positive_f64(options.wall_time_seconds, "--wall-time-seconds")?;
    let average_power_watts =
        required_release_positive_f64(options.average_power_watts, "--average-power-watts")?;
    let energy_joules = required_release_positive_f64(options.energy_joules, "--energy-joules")?;
    let telemetry_validated =
        required_release_true(options.telemetry_validated, "--telemetry-validated")?;
    let wall_time_budget_seconds = resource_contract.training_time_budget_seconds;
    let energy_budget_joules = resource_contract.train_energy_budget_joules_proxy;
    let wall_time_budget_fit = wall_time_seconds <= wall_time_budget_seconds;
    let energy_budget_fit = energy_joules <= energy_budget_joules;
    if !wall_time_budget_fit {
        return Err(PgError::InvalidOp(format!(
            "--wall-time-seconds {wall_time_seconds} exceeds training-time budget {wall_time_budget_seconds}"
        )));
    }
    if !energy_budget_fit {
        return Err(PgError::InvalidOp(format!(
            "--energy-joules {energy_joules} exceeds energy budget {energy_budget_joules}"
        )));
    }
    let expected_energy_joules = wall_time_seconds * average_power_watts;
    let energy_consistency_error_pct =
        finite_relative_error_pct(energy_joules, expected_energy_joules)?;
    if energy_consistency_error_pct > 5.0 {
        return Err(PgError::InvalidOp(format!(
            "--energy-joules must agree with --wall-time-seconds * --average-power-watts within 5%, got {energy_consistency_error_pct:.6}%"
        )));
    }
    let expected_raw_claims = serde_json::json!({
        "measurement_run_id": measurement_run_id,
        "power_meter_id": power_meter_id,
        "wall_time_seconds": wall_time_seconds,
        "average_power_watts": average_power_watts,
        "energy_joules": energy_joules,
        "telemetry_validated": telemetry_validated,
        "wall_time_budget_seconds": wall_time_budget_seconds,
        "energy_budget_joules": energy_budget_joules,
        "wall_time_budget_fit": wall_time_budget_fit,
        "energy_budget_fit": energy_budget_fit,
        "energy_consistency_error_pct": energy_consistency_error_pct,
    });
    let raw_evidence = required_raw_evidence_claim(options, "resource_cost", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "measurement_run_id": measurement_run_id,
        "power_meter_id": power_meter_id,
        "wall_time_seconds": wall_time_seconds,
        "average_power_watts": average_power_watts,
        "energy_joules": energy_joules,
        "telemetry_validated": telemetry_validated,
        "wall_time_budget_seconds": wall_time_budget_seconds,
        "energy_budget_joules": energy_budget_joules,
        "wall_time_budget_fit": wall_time_budget_fit,
        "energy_budget_fit": energy_budget_fit,
        "energy_consistency_error_pct": energy_consistency_error_pct,
        "resource_contract_energy_proxy_source": "hardware-label nominal power heuristic",
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "resource_cost",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "Resource/cost release source report binds caller-supplied measured wall-time, average-power, and energy telemetry to the current resource contract; the command verifies telemetry consistency and budget fit but does not measure power itself",
    })
}

fn run_modelgolf_optimizer_comm_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    optimizer_comm: &OptimizerCommReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let distributed_backend_id =
        required_release_string(&options.distributed_backend_id, "--distributed-backend-id")?;
    let reduce_scatter_parity_pass = required_release_true(
        options.reduce_scatter_parity_pass,
        "--reduce-scatter-parity-pass",
    )?;
    let all_gather_parity_pass =
        required_release_true(options.all_gather_parity_pass, "--all-gather-parity-pass")?;
    let optimizer_update_parity_pass = required_release_true(
        options.optimizer_update_parity_pass,
        "--optimizer-update-parity-pass",
    )?;
    let nccl_trace_validated =
        required_release_true(options.nccl_trace_validated, "--nccl-trace-validated")?;
    let overlap_validated =
        required_release_true(options.overlap_validated, "--overlap-validated")?;
    let measured_comm_time_ms =
        required_release_positive_f64(options.measured_comm_time_ms, "--measured-comm-time-ms")?;
    let measured_step_time_ms =
        required_release_positive_f64(options.measured_step_time_ms, "--measured-step-time-ms")?;
    let communication_speedup_x = required_release_f64_at_least(
        options.communication_speedup_x,
        "--communication-speedup-x",
        1.0,
    )?;
    if measured_comm_time_ms > measured_step_time_ms {
        return Err(PgError::InvalidOp(format!(
            "--measured-comm-time-ms {measured_comm_time_ms} cannot exceed --measured-step-time-ms {measured_step_time_ms}"
        )));
    }
    if !optimizer_comm.local_proof.finite
        || !optimizer_comm.local_proof.exact_equivalence
        || !optimizer_comm.shard_separable_update_contract
    {
        return Err(PgError::InvalidOp(
            "current Optimizer/Comm local equivalence proof did not pass".to_string(),
        ));
    }
    let expected_raw_claims = serde_json::json!({
        "distributed_backend_id": distributed_backend_id,
        "world_size": optimizer_comm.world_size,
        "optimizer_sharded": optimizer_comm.optimizer_sharded,
        "nccl_overlap_mode": optimizer_comm.nccl_overlap_mode,
        "sharded_total_wire_bytes_per_rank": optimizer_comm.sharded_total_wire_bytes_per_rank,
        "owned_optimizer_state_reduction_x": optimizer_comm.owned_optimizer_state_reduction_x,
        "reduce_scatter_parity_pass": reduce_scatter_parity_pass,
        "all_gather_parity_pass": all_gather_parity_pass,
        "optimizer_update_parity_pass": optimizer_update_parity_pass,
        "nccl_trace_validated": nccl_trace_validated,
        "overlap_validated": overlap_validated,
        "measured_comm_time_ms": measured_comm_time_ms,
        "measured_step_time_ms": measured_step_time_ms,
        "communication_speedup_x": communication_speedup_x,
    });
    let raw_evidence =
        required_raw_evidence_claim(options, "optimizer_comm", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "distributed_backend_id": distributed_backend_id,
        "world_size": optimizer_comm.world_size,
        "optimizer_sharded": optimizer_comm.optimizer_sharded,
        "nccl_overlap_mode": optimizer_comm.nccl_overlap_mode,
        "sharded_total_wire_bytes_per_rank": optimizer_comm.sharded_total_wire_bytes_per_rank,
        "owned_optimizer_state_reduction_x": optimizer_comm.owned_optimizer_state_reduction_x,
        "reduce_scatter_parity_pass": reduce_scatter_parity_pass,
        "all_gather_parity_pass": all_gather_parity_pass,
        "optimizer_update_parity_pass": optimizer_update_parity_pass,
        "nccl_trace_validated": nccl_trace_validated,
        "overlap_validated": overlap_validated,
        "measured_comm_time_ms": measured_comm_time_ms,
        "measured_step_time_ms": measured_step_time_ms,
        "communication_speedup_x": communication_speedup_x,
        "local_equivalence_proof": optimizer_comm.local_proof.exact_equivalence,
        "local_max_abs_diff": optimizer_comm.local_proof.max_abs_diff,
        "shard_separable_update_contract": optimizer_comm.shard_separable_update_contract,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "optimizer_comm",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "Optimizer/Comm release source report binds caller-supplied distributed parity, NCCL trace, overlap, and timing claims to the current sharded optimizer communication plan; the command verifies local equivalence proof and threshold checks but does not run the distributed backend itself",
    })
}

fn run_modelgolf_kernel_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    kernel_forge: &KernelForgeReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let backend_id = required_release_string(&options.backend_id, "--backend-id")?;
    let generated_kernel_ids =
        required_release_csv_strings(&options.generated_kernel_ids, "--generated-kernel-ids")?;
    let generated_kernels =
        required_release_true(options.generated_kernels, "--generated-kernels")?;
    let parity_pass = required_release_true(options.parity_pass, "--parity-pass")?;
    let speedup_x = required_release_f64_at_least(options.speedup_x, "--speedup-x", 1.0)?;
    let memory_reduction_x =
        required_release_f64_at_least(options.memory_reduction_x, "--memory-reduction-x", 1.0)?;
    if !kernel_forge.exact_tiled_ce.local_proof.finite
        || !kernel_forge.exact_tiled_ce.local_proof.parity_ok
    {
        return Err(PgError::InvalidOp(
            "current KernelForge exact tiled CE local proof did not pass".to_string(),
        ));
    }
    let expected_raw_claims = serde_json::json!({
        "backend_id": backend_id,
        "generated_kernel_ids": generated_kernel_ids,
        "tile_t": kernel_forge.exact_tiled_ce.tile_t,
        "generated_kernels": generated_kernels,
        "parity_pass": parity_pass,
        "speedup_x": speedup_x,
        "memory_reduction_x": memory_reduction_x,
    });
    let raw_evidence = required_raw_evidence_claim(options, "kernel_forge", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "backend_id": backend_id,
        "generated_kernel_ids": generated_kernel_ids,
        "tile_t": kernel_forge.exact_tiled_ce.tile_t,
        "generated_kernels": generated_kernels,
        "parity_pass": parity_pass,
        "speedup_x": speedup_x,
        "memory_reduction_x": memory_reduction_x,
        "local_exact_tiled_ce_parity": kernel_forge.exact_tiled_ce.local_proof.parity_ok,
        "local_scratch_reduction_x": kernel_forge.exact_tiled_ce.scratch_reduction_x,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "kernel_forge",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "KernelForge release source report binds caller-supplied generated-kernel parity and performance claims to the current exact tiled CE tile size; the command verifies local exact-CE proof and release thresholds but does not generate backend kernels itself",
    })
}

fn run_modelgolf_wind_source_report(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    wind_tunnel: &ModelGolfWindReport,
    options: &ModelGolfOptions,
) -> PgResult<ModelGolfReleaseSourceReport> {
    let trace_corpus_id = required_release_string(&options.trace_corpus_id, "--trace-corpus-id")?;
    let calibration_report_id =
        required_release_string(&options.calibration_report_id, "--calibration-report-id")?;
    let fresh_profiler_traces =
        required_release_true(options.fresh_profiler_traces, "--fresh-profiler-traces")?;
    let external_timing_validated = required_release_true(
        options.external_timing_validated,
        "--external-timing-validated",
    )?;
    let holdout_spearman =
        required_release_f64_at_least(options.holdout_spearman, "--holdout-spearman", 0.50)?;
    let mean_abs_pct_error =
        required_release_f64_at_most(options.mean_abs_pct_error, "--mean-abs-pct-error", 25.0)?;
    let expected_raw_claims = serde_json::json!({
        "trace_corpus_id": trace_corpus_id,
        "calibration_report_id": calibration_report_id,
        "fresh_profiler_traces": fresh_profiler_traces,
        "external_timing_validated": external_timing_validated,
        "holdout_spearman": holdout_spearman,
        "mean_abs_pct_error": mean_abs_pct_error,
    });
    let raw_evidence = required_raw_evidence_claim(options, "wind_tunnel", &expected_raw_claims)?;
    let claims = serde_json::json!({
        "source_report_schema_version": 1,
        "raw_evidence": raw_evidence,
        "trace_corpus_id": trace_corpus_id,
        "calibration_report_id": calibration_report_id,
        "fresh_profiler_traces": fresh_profiler_traces,
        "external_timing_validated": external_timing_validated,
        "holdout_spearman": holdout_spearman,
        "mean_abs_pct_error": mean_abs_pct_error,
        "local_estimate_only": wind_tunnel.estimate_only,
        "pareto_candidate_count": wind_tunnel.pareto_candidates.len(),
        "local_recommendation": wind_tunnel.recommendation,
    });
    Ok(ModelGolfReleaseSourceReport {
        kind: "modelgolf_release_source_report",
        pillar: "wind_tunnel",
        spec_name: spec.name.clone(),
        spec_fingerprint: spec_fingerprint.to_string(),
        hardware: resource_contract.hardware.clone(),
        runtime: resource_contract.runtime.clone(),
        generated_at: modelgolf_generated_at_unix(),
        claims,
        evidence_boundary: "Wind Tunnel release source report binds caller-supplied fresh trace, calibration, and external validation metrics to the current local candidate plan; the command verifies release thresholds but does not collect profiler traces itself",
    })
}

fn modelgolf_required_release_pillars() -> [&'static str; 9] {
    [
        "resource_cost",
        "pack",
        "lqer",
        "cache",
        "delta",
        "train",
        "optimizer_comm",
        "kernel_forge",
        "wind_tunnel",
    ]
}

fn required_source_claim(
    claims_by_pillar: &BTreeMap<String, serde_json::Value>,
    pillar: &str,
    key: &str,
) -> PgResult<serde_json::Value> {
    let Some(claims) = claims_by_pillar.get(pillar) else {
        return Err(PgError::InvalidOp(format!(
            "source report pillar={pillar} missing"
        )));
    };
    let Some(value) = claims.get(key) else {
        return Err(PgError::InvalidOp(format!(
            "source report pillar={pillar} missing claim {key}"
        )));
    };
    Ok(value.clone())
}

fn run_modelgolf_release_evidence(
    spec: &RunSpec,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    options: &ModelGolfOptions,
) -> PgResult<serde_json::Value> {
    let source_report_dir = options.source_report_dir.as_deref().ok_or_else(|| {
        PgError::InvalidOp("--source-report-dir is required for release-evidence".to_string())
    })?;
    let evidence_id = required_release_string(&options.evidence_id, "--evidence-id")?;
    let generated_at = options
        .generated_at
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(modelgolf_generated_at_unix);
    let entries = std::fs::read_dir(source_report_dir).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to read --source-report-dir {}: {err}",
            source_report_dir.display()
        ))
    })?;
    let mut claims_by_pillar = BTreeMap::<String, serde_json::Value>::new();
    let mut paths_by_pillar = BTreeMap::<String, PathBuf>::new();
    let mut sha_by_pillar = BTreeMap::<String, String>::new();
    let release_evidence_base = release_evidence_base_dir(options.output.as_deref());

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).map_err(|err| {
            PgError::InvalidOp(format!(
                "failed to read source-report candidate {}: {err}",
                path.display()
            ))
        })?;
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        if value.get("kind").and_then(serde_json::Value::as_str)
            != Some("modelgolf_release_source_report")
        {
            continue;
        }
        let pillar = value
            .get("pillar")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PgError::InvalidOp(format!("source-report {} missing pillar", path.display()))
            })?;
        if claims_by_pillar.contains_key(pillar) {
            return Err(PgError::InvalidOp(format!(
                "duplicate source report pillar={pillar} in {}",
                source_report_dir.display()
            )));
        }
        let report_spec_name = value.get("spec_name").and_then(serde_json::Value::as_str);
        let report_spec_fingerprint = value
            .get("spec_fingerprint")
            .and_then(serde_json::Value::as_str);
        let report_hardware = value.get("hardware").and_then(serde_json::Value::as_str);
        let report_runtime = value.get("runtime").and_then(serde_json::Value::as_str);
        if report_spec_name != Some(spec.name.as_str())
            || report_spec_fingerprint != Some(spec_fingerprint)
            || report_hardware != Some(resource_contract.hardware.as_str())
            || report_runtime != Some(resource_contract.runtime.as_str())
        {
            return Err(PgError::InvalidOp(format!(
                "source-report {} pillar={pillar} does not match current spec fingerprint or hardware/runtime labels",
                path.display()
            )));
        }
        let claims = value
            .get("claims")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                PgError::InvalidOp(format!(
                    "source-report {} pillar={pillar} missing claims object",
                    path.display()
                ))
            })?;
        let canonical_path = std::fs::canonicalize(&path).unwrap_or(path);
        let sha256 = format!(
            "sha256:{}",
            modelgolf_sha256_file(&canonical_path).map_err(|err| {
                PgError::InvalidOp(format!(
                    "failed to hash source-report {}: {err}",
                    canonical_path.display()
                ))
            })?
        );
        claims_by_pillar.insert(
            pillar.to_string(),
            serde_json::Value::Object(claims.clone()),
        );
        paths_by_pillar.insert(pillar.to_string(), canonical_path);
        sha_by_pillar.insert(pillar.to_string(), sha256);
    }

    for pillar in modelgolf_required_release_pillars() {
        if !claims_by_pillar.contains_key(pillar) {
            return Err(PgError::InvalidOp(format!(
                "--source-report-dir {} missing required pillar={pillar}",
                source_report_dir.display()
            )));
        }
    }

    let source_reports = modelgolf_required_release_pillars()
        .iter()
        .map(|pillar| {
            let path = paths_by_pillar
                .get(*pillar)
                .expect("required pillar path is present");
            Ok(ModelGolfReleaseSourceReportEvidence {
                pillar: (*pillar).to_string(),
                kind: "modelgolf_release_source_report".to_string(),
                path: release_portable_evidence_path(path, &release_evidence_base),
                sha256: sha_by_pillar
                    .get(*pillar)
                    .expect("required pillar sha is present")
                    .clone(),
            })
        })
        .collect::<PgResult<Vec<_>>>()?;
    let validation_probe = ModelGolfReleaseEvidenceFile {
        kind: Some("modelgolf_release_evidence".to_string()),
        spec_name: Some(spec.name.clone()),
        spec_fingerprint: Some(spec_fingerprint.to_string()),
        hardware: Some(resource_contract.hardware.clone()),
        runtime: Some(resource_contract.runtime.clone()),
        evidence_id: Some(evidence_id.clone()),
        generated_at: Some(generated_at.clone()),
        source_reports: Some(source_reports.clone()),
        pack: None,
        lqer: None,
        cache: None,
        delta: None,
        train: None,
        resource_cost: None,
        optimizer_comm: None,
        kernel_forge: None,
        wind_tunnel: None,
    };
    let source_report_validation = validate_modelgolf_source_reports(
        Some(&validation_probe),
        spec.name.as_str(),
        spec_fingerprint,
        resource_contract,
        options.output.as_deref(),
    );
    if !source_report_validation.valid {
        return Err(PgError::InvalidOp(format!(
            "--source-report-dir {} contains invalid release source reports: {}",
            source_report_dir.display(),
            source_report_validation.details
        )));
    }
    let claims_by_pillar = source_report_validation.claims_by_pillar;

    Ok(serde_json::json!({
        "kind": "modelgolf_release_evidence",
        "spec_name": spec.name.clone(),
        "spec_fingerprint": spec_fingerprint,
        "hardware": resource_contract.hardware.clone(),
        "runtime": resource_contract.runtime.clone(),
        "evidence_id": evidence_id,
        "generated_at": generated_at,
        "source_reports": source_reports,
        "resource_cost": {
            "measurement_run_id": required_source_claim(&claims_by_pillar, "resource_cost", "measurement_run_id")?,
            "power_meter_id": required_source_claim(&claims_by_pillar, "resource_cost", "power_meter_id")?,
            "wall_time_seconds": required_source_claim(&claims_by_pillar, "resource_cost", "wall_time_seconds")?,
            "average_power_watts": required_source_claim(&claims_by_pillar, "resource_cost", "average_power_watts")?,
            "energy_joules": required_source_claim(&claims_by_pillar, "resource_cost", "energy_joules")?,
            "telemetry_validated": required_source_claim(&claims_by_pillar, "resource_cost", "telemetry_validated")?,
            "wall_time_budget_seconds": required_source_claim(&claims_by_pillar, "resource_cost", "wall_time_budget_seconds")?,
            "energy_budget_joules": required_source_claim(&claims_by_pillar, "resource_cost", "energy_budget_joules")?,
            "wall_time_budget_fit": required_source_claim(&claims_by_pillar, "resource_cost", "wall_time_budget_fit")?,
            "energy_budget_fit": required_source_claim(&claims_by_pillar, "resource_cost", "energy_budget_fit")?,
            "energy_consistency_error_pct": required_source_claim(&claims_by_pillar, "resource_cost", "energy_consistency_error_pct")?,
        },
        "pack": {
            "artifact_path": required_source_claim(&claims_by_pillar, "pack", "artifact_path")?,
            "artifact_sha256": required_source_claim(&claims_by_pillar, "pack", "artifact_sha256")?,
            "validation_dataset_id": required_source_claim(&claims_by_pillar, "pack", "validation_dataset_id")?,
            "validation_command": required_source_claim(&claims_by_pillar, "pack", "validation_command")?,
            "artifact_bytes": required_source_claim(&claims_by_pillar, "pack", "artifact_bytes")?,
            "strict_reload_pass": required_source_claim(&claims_by_pillar, "pack", "strict_reload_pass")?,
            "heldout_bpb": required_source_claim(&claims_by_pillar, "pack", "heldout_bpb")?,
            "baseline_bpb": required_source_claim(&claims_by_pillar, "pack", "baseline_bpb")?,
            "relative_bpb_increase_pct": required_source_claim(&claims_by_pillar, "pack", "relative_bpb_increase_pct")?,
            "decode_tokens_per_second": required_source_claim(&claims_by_pillar, "pack", "decode_tokens_per_second")?,
        },
        "lqer": {
            "calibration_dataset_id": required_source_claim(&claims_by_pillar, "lqer", "calibration_dataset_id")?,
            "selected_lqer_group_names": required_source_claim(&claims_by_pillar, "lqer", "selected_lqer_group_names")?,
            "production_svd_validated": required_source_claim(&claims_by_pillar, "lqer", "production_svd_validated")?,
            "calibrated_tensor_sensitivity": required_source_claim(&claims_by_pillar, "lqer", "calibrated_tensor_sensitivity")?,
            "equal_byte_bpb_delta": required_source_claim(&claims_by_pillar, "lqer", "equal_byte_bpb_delta")?,
        },
        "cache": {
            "backend_id": required_source_claim(&claims_by_pillar, "cache", "backend_id")?,
            "kernel_id": required_source_claim(&claims_by_pillar, "cache", "kernel_id")?,
            "long_context_dataset_id": required_source_claim(&claims_by_pillar, "cache", "long_context_dataset_id")?,
            "k_bits": required_source_claim(&claims_by_pillar, "cache", "k_bits")?,
            "v_bits": required_source_claim(&claims_by_pillar, "cache", "v_bits")?,
            "block_size_tokens": required_source_claim(&claims_by_pillar, "cache", "block_size_tokens")?,
            "layout": required_source_claim(&claims_by_pillar, "cache", "layout")?,
            "context_tokens": required_source_claim(&claims_by_pillar, "cache", "context_tokens")?,
            "batch_sequences": required_source_claim(&claims_by_pillar, "cache", "batch_sequences")?,
            "fused_runtime": required_source_claim(&claims_by_pillar, "cache", "fused_runtime")?,
            "parity_pass": required_source_claim(&claims_by_pillar, "cache", "parity_pass")?,
            "memory_budget_fit": required_source_claim(&claims_by_pillar, "cache", "memory_budget_fit")?,
            "long_context_bpb_delta_pct": required_source_claim(&claims_by_pillar, "cache", "long_context_bpb_delta_pct")?,
            "speedup_x": required_source_claim(&claims_by_pillar, "cache", "speedup_x")?,
        },
        "delta": {
            "domain_dataset_id": required_source_claim(&claims_by_pillar, "delta", "domain_dataset_id")?,
            "selected_delta_names": required_source_claim(&claims_by_pillar, "delta", "selected_delta_names")?,
            "trained_delta_bytes": required_source_claim(&claims_by_pillar, "delta", "trained_delta_bytes")?,
            "legality_pass": required_source_claim(&claims_by_pillar, "delta", "legality_pass")?,
            "score_first_trace_or_review": required_source_claim(&claims_by_pillar, "delta", "score_first_trace_or_review")?,
            "equal_byte_domain_bpb_delta": required_source_claim(&claims_by_pillar, "delta", "equal_byte_domain_bpb_delta")?,
        },
        "train": {
            "training_run_id": required_source_claim(&claims_by_pillar, "train", "training_run_id")?,
            "backend_id": required_source_claim(&claims_by_pillar, "train", "backend_id")?,
            "gpu_backend_integrated": required_source_claim(&claims_by_pillar, "train", "gpu_backend_integrated")?,
            "proxy_calibrated": required_source_claim(&claims_by_pillar, "train", "proxy_calibrated")?,
            "post_export_bpb_delta_vs_posthoc": required_source_claim(&claims_by_pillar, "train", "post_export_bpb_delta_vs_posthoc")?,
        },
        "optimizer_comm": {
            "distributed_backend_id": required_source_claim(&claims_by_pillar, "optimizer_comm", "distributed_backend_id")?,
            "world_size": required_source_claim(&claims_by_pillar, "optimizer_comm", "world_size")?,
            "optimizer_sharded": required_source_claim(&claims_by_pillar, "optimizer_comm", "optimizer_sharded")?,
            "nccl_overlap_mode": required_source_claim(&claims_by_pillar, "optimizer_comm", "nccl_overlap_mode")?,
            "sharded_total_wire_bytes_per_rank": required_source_claim(&claims_by_pillar, "optimizer_comm", "sharded_total_wire_bytes_per_rank")?,
            "owned_optimizer_state_reduction_x": required_source_claim(&claims_by_pillar, "optimizer_comm", "owned_optimizer_state_reduction_x")?,
            "reduce_scatter_parity_pass": required_source_claim(&claims_by_pillar, "optimizer_comm", "reduce_scatter_parity_pass")?,
            "all_gather_parity_pass": required_source_claim(&claims_by_pillar, "optimizer_comm", "all_gather_parity_pass")?,
            "optimizer_update_parity_pass": required_source_claim(&claims_by_pillar, "optimizer_comm", "optimizer_update_parity_pass")?,
            "nccl_trace_validated": required_source_claim(&claims_by_pillar, "optimizer_comm", "nccl_trace_validated")?,
            "overlap_validated": required_source_claim(&claims_by_pillar, "optimizer_comm", "overlap_validated")?,
            "measured_comm_time_ms": required_source_claim(&claims_by_pillar, "optimizer_comm", "measured_comm_time_ms")?,
            "measured_step_time_ms": required_source_claim(&claims_by_pillar, "optimizer_comm", "measured_step_time_ms")?,
            "communication_speedup_x": required_source_claim(&claims_by_pillar, "optimizer_comm", "communication_speedup_x")?,
        },
        "kernel_forge": {
            "backend_id": required_source_claim(&claims_by_pillar, "kernel_forge", "backend_id")?,
            "generated_kernel_ids": required_source_claim(&claims_by_pillar, "kernel_forge", "generated_kernel_ids")?,
            "tile_t": required_source_claim(&claims_by_pillar, "kernel_forge", "tile_t")?,
            "generated_kernels": required_source_claim(&claims_by_pillar, "kernel_forge", "generated_kernels")?,
            "parity_pass": required_source_claim(&claims_by_pillar, "kernel_forge", "parity_pass")?,
            "speedup_x": required_source_claim(&claims_by_pillar, "kernel_forge", "speedup_x")?,
            "memory_reduction_x": required_source_claim(&claims_by_pillar, "kernel_forge", "memory_reduction_x")?,
        },
        "wind_tunnel": {
            "trace_corpus_id": required_source_claim(&claims_by_pillar, "wind_tunnel", "trace_corpus_id")?,
            "calibration_report_id": required_source_claim(&claims_by_pillar, "wind_tunnel", "calibration_report_id")?,
            "fresh_profiler_traces": required_source_claim(&claims_by_pillar, "wind_tunnel", "fresh_profiler_traces")?,
            "external_timing_validated": required_source_claim(&claims_by_pillar, "wind_tunnel", "external_timing_validated")?,
            "holdout_spearman": required_source_claim(&claims_by_pillar, "wind_tunnel", "holdout_spearman")?,
            "mean_abs_pct_error": required_source_claim(&claims_by_pillar, "wind_tunnel", "mean_abs_pct_error")?,
        },
    }))
}

fn nominal_power_watts_proxy(hardware: &str, world_size: usize) -> (f64, String) {
    let hardware_lower = hardware.to_ascii_lowercase();
    let accelerators = world_size.max(1) as f64;
    if hardware_lower.contains("h100") {
        (
            700.0 * accelerators,
            "h100_700w_per_accelerator_proxy".to_string(),
        )
    } else if hardware_lower.contains("a100") {
        (
            400.0 * accelerators,
            "a100_400w_per_accelerator_proxy".to_string(),
        )
    } else if hardware_lower.contains("4090") || hardware_lower.contains("rtx") {
        (
            450.0 * accelerators,
            "desktop_gpu_450w_per_accelerator_proxy".to_string(),
        )
    } else if hardware_lower.contains("m2")
        || hardware_lower.contains("m3")
        || hardware_lower.contains("apple")
        || hardware_lower.contains("mac")
    {
        (35.0, "apple_silicon_35w_system_proxy".to_string())
    } else if hardware_lower.contains("laptop") {
        (35.0, "laptop_35w_system_proxy".to_string())
    } else {
        (
            65.0 * accelerators,
            "generic_local_65w_per_worker_proxy".to_string(),
        )
    }
}

fn plan_constraint_cost_model(
    spec: &RunSpec,
    resource_contract: &ResourceContractReport,
    total_parameter_elems_estimate: usize,
) -> ConstraintCostModelReport {
    let training_time_budget_seconds = resource_contract.training_time_budget_seconds.max(1e-9);
    let train_tokens_total = spec
        .train
        .batch_tokens
        .saturating_mul(spec.train.total_iterations.max(1));
    let train_tokens_per_budget_second = train_tokens_total as f64 / training_time_budget_seconds;
    let artifact_bytes_per_budget_second =
        resource_contract.artifact_budget_bytes as f64 / training_time_budget_seconds;
    let parameter_elems_per_energy_joule_proxy = total_parameter_elems_estimate as f64
        / resource_contract.train_energy_budget_joules_proxy.max(1e-9);
    let latency_budget_tokens_per_second = resource_contract.latency_target_ms.and_then(|ms| {
        (ms.is_finite() && ms > 0.0)
            .then_some(resource_contract.batch_sequences as f64 * 1000.0 / ms)
    });

    ConstraintCostModelReport {
        kind: "modelgolf_constraint_cost_model",
        training_time_budget_seconds: resource_contract.training_time_budget_seconds,
        nominal_power_watts_proxy: resource_contract.nominal_power_watts_proxy,
        train_energy_budget_joules_proxy: resource_contract.train_energy_budget_joules_proxy,
        train_tokens_per_budget_second,
        artifact_bytes_per_budget_second,
        parameter_elems_per_energy_joule_proxy,
        latency_target_ms: resource_contract.latency_target_ms,
        latency_budget_tokens_per_second,
        energy_proxy_source: "hardware-label nominal power heuristic".to_string(),
        evidence_boundary: "planning proxy derived from spec wall-clock budget and nominal hardware power labels; not measured wall energy, thermal throttling, datacenter PUE, or release evidence",
        notes: vec![
            "Use this report to compare constraints and candidate plans before paid runs; measured release claims must come from profiler/power telemetry.".to_string(),
            "Training-time budget comes from RunSpec.train.max_wallclock_seconds unless future CLI/API overrides provide a stricter contract.".to_string(),
        ],
    }
}

pub fn run_modelgolf_plan(options: ModelGolfOptions) -> PgResult<ModelGolfReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let model_config = spec.model.to_model_config();
    let manifest = pg_quant::compile_quant_layout_manifest(&spec.quant, Some(&model_config))?;
    let tensor_groups = modelgolf_tensor_groups(&manifest.groups);
    let artifact_budget_bytes = options
        .artifact_budget_bytes
        .unwrap_or(spec.quant.target_artifact_bytes);
    let context_tokens = options.context.unwrap_or(spec.model.eval_seq_len);
    let batch_sequences = options.batch.unwrap_or_else(|| {
        spec.train
            .batch_tokens
            .checked_div(spec.train.seq_len.max(1))
            .unwrap_or(1)
            .max(1)
    });
    let training_time_budget_seconds = f64::from(spec.train.max_wallclock_seconds.max(1e-6));
    let (nominal_power_watts_proxy, energy_proxy_source) =
        nominal_power_watts_proxy(&options.hardware, spec.train.world_size);
    let train_energy_budget_joules_proxy = training_time_budget_seconds * nominal_power_watts_proxy;
    let resource_contract = ResourceContractReport {
        hardware: options.hardware.clone(),
        runtime: options.runtime.clone(),
        artifact_budget_bytes,
        memory_budget_bytes: options.memory_budget_bytes,
        latency_target_ms: options.latency_target_ms,
        quality_budget_ppl_pct: options.quality_budget_ppl_pct,
        training_time_budget_seconds,
        nominal_power_watts_proxy,
        train_energy_budget_joules_proxy,
        context_tokens,
        batch_sequences,
        train_batch_tokens: spec.train.batch_tokens,
        eval_stride: spec.eval.stride,
        adaptation_legality: if spec.eval.legal_score_first {
            "score_first_adaptation_only".to_string()
        } else {
            "static_eval_only_or_unchecked".to_string()
        },
        evidence_boundary: "resource contract is spec/user supplied plus local hardware-label energy proxy; release energy claims require measured telemetry",
    };
    let artifact_ir = ModelArtifactIrReport {
        model_family: format!("{:?}", spec.model.family),
        vocab_size: model_config.vocab_size,
        num_layers: model_config.num_layers,
        model_dim: model_config.model_dim,
        num_heads: model_config.num_heads,
        num_kv_heads: model_config.num_kv_heads,
        mlp_dim: model_config.mlp_dim,
        total_parameter_elems_estimate: model_config.param_count(),
        tensor_groups: tensor_groups.clone(),
        typed_metadata: vec![
            format!("compute_precision={:?}", spec.model.compute_precision),
            format!("output_ce_backend={:?}", spec.model.output_ce_backend),
            format!(
                "distributed_optimizer_backend={:?}",
                spec.train.distributed_optimizer_backend
            ),
            format!("compression={:?}", spec.quant.compression),
            format!("eval_adaptation_backend={:?}", spec.eval.adaptation_backend),
        ],
    };
    let cost_model = plan_constraint_cost_model(
        &spec,
        &resource_contract,
        artifact_ir.total_parameter_elems_estimate,
    );
    let mut cost_model = cost_model;
    cost_model.energy_proxy_source = energy_proxy_source;

    let quality_calibration = options
        .quality_calibration
        .as_deref()
        .map(|path| load_pack_quality_calibration(path, &tensor_groups, &spec))
        .transpose()?;
    let mut pack = plan_pack(
        &tensor_groups,
        &spec,
        artifact_budget_bytes,
        quality_calibration.as_ref(),
    );
    if let Some(path) = options.proof_artifact.as_deref() {
        pack.artifact_proof = Some(prove_pack_artifact(
            &spec,
            &plan,
            path,
            artifact_budget_bytes,
        )?);
    }
    let pack_experiment = if options.section == ModelGolfSection::PackExperiment {
        Some(run_pack_measured_experiment(
            &spec,
            options.spec.as_path(),
            artifact_budget_bytes,
            &plan.variant_fingerprint,
        )?)
    } else {
        None
    };
    let cache = plan_cache(
        &spec,
        context_tokens,
        batch_sequences,
        options.memory_budget_bytes,
    )?;
    let cache_experiment = if options.section == ModelGolfSection::CacheExperiment {
        Some(run_cachegolf_kv_experiment(
            &spec,
            context_tokens,
            batch_sequences,
            cache.target_cache_bytes,
        )?)
    } else {
        None
    };
    let delta_budget_bytes = options.delta_budget_bytes.unwrap_or_else(|| {
        artifact_budget_bytes
            .checked_div(16)
            .unwrap_or(64 * 1024)
            .clamp(64 * 1024, 1_000_000)
    });
    let delta = plan_delta(&spec, delta_budget_bytes)?;
    let train = plan_train(&spec, &pack);
    let optimizer_comm = plan_optimizer_comm(&spec);
    let kernel_forge = plan_kernel_forge(&spec);
    let kernel_experiment = if options.section == ModelGolfSection::KernelExperiment {
        Some(run_kernel_forge_ce_experiment(&spec))
    } else {
        None
    };
    let wind_tunnel = plan_wind(&spec, &plan, &pack, &cache, &delta);
    let wind_experiment = if options.section == ModelGolfSection::WindExperiment {
        Some(run_modelgolf_wind_experiment(
            &spec,
            &plan.variant_fingerprint,
            artifact_budget_bytes,
            context_tokens,
            batch_sequences,
            options.memory_budget_bytes,
        )?)
    } else {
        None
    };
    let scale_golf = plan_scale_golf(ScaleGolfPlanInputs {
        spec: &spec,
        resource_contract: &resource_contract,
        cost_model: &cost_model,
        artifact_ir: &artifact_ir,
        pack: &pack,
        cache: &cache,
        delta: &delta,
        train: &train,
        optimizer_comm: &optimizer_comm,
        kernel_forge: &kernel_forge,
        wind_tunnel: &wind_tunnel,
    });
    let platform_status = platform_statuses();
    let release_evidence = options
        .release_evidence
        .as_deref()
        .map(load_modelgolf_release_evidence)
        .transpose()?;
    let spec_name_for_release = spec.name.clone();
    let spec_fingerprint_for_release = plan.variant_fingerprint.clone();
    let release_readiness = modelgolf_release_readiness(
        &spec_name_for_release,
        &spec_fingerprint_for_release,
        &resource_contract,
        &pack,
        &cache,
        &delta,
        &train,
        &optimizer_comm,
        &kernel_forge,
        &wind_tunnel,
        options.release_evidence.as_deref(),
        release_evidence.as_ref(),
    );
    let pack_source_report = if options.section == ModelGolfSection::PackSourceReport {
        Some(run_modelgolf_pack_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            artifact_budget_bytes,
            &options,
        )?)
    } else {
        None
    };
    let lqer_source_report = if options.section == ModelGolfSection::LqerSourceReport {
        Some(run_modelgolf_lqer_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &pack,
            &options,
        )?)
    } else {
        None
    };
    let cache_source_report = if options.section == ModelGolfSection::CacheSourceReport {
        Some(run_modelgolf_cache_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &cache,
            &options,
        )?)
    } else {
        None
    };
    let delta_source_report = if options.section == ModelGolfSection::DeltaSourceReport {
        Some(run_modelgolf_delta_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &delta,
            &options,
        )?)
    } else {
        None
    };
    let train_source_report = if options.section == ModelGolfSection::TrainSourceReport {
        Some(run_modelgolf_train_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &train,
            &options,
        )?)
    } else {
        None
    };
    let resource_cost_source_report =
        if options.section == ModelGolfSection::ResourceCostSourceReport {
            Some(run_modelgolf_resource_cost_source_report(
                &spec,
                &spec_fingerprint_for_release,
                &resource_contract,
                &options,
            )?)
        } else {
            None
        };
    let optimizer_comm_source_report =
        if options.section == ModelGolfSection::OptimizerCommSourceReport {
            Some(run_modelgolf_optimizer_comm_source_report(
                &spec,
                &spec_fingerprint_for_release,
                &resource_contract,
                &optimizer_comm,
                &options,
            )?)
        } else {
            None
        };
    let kernel_source_report = if options.section == ModelGolfSection::KernelSourceReport {
        Some(run_modelgolf_kernel_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &kernel_forge,
            &options,
        )?)
    } else {
        None
    };
    let wind_source_report = if options.section == ModelGolfSection::WindSourceReport {
        Some(run_modelgolf_wind_source_report(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &wind_tunnel,
            &options,
        )?)
    } else {
        None
    };
    let release_evidence_report = if options.section == ModelGolfSection::ReleaseEvidence {
        Some(run_modelgolf_release_evidence(
            &spec,
            &spec_fingerprint_for_release,
            &resource_contract,
            &options,
        )?)
    } else {
        None
    };
    let status = if pack.feasible && cache.feasible {
        "planner_ready"
    } else {
        "planner_infeasible_for_contract"
    }
    .to_string();
    let report = ModelGolfReport {
        kind: "modelgolf_constraint_native_platform_plan",
        project_name: "ModelGolf",
        thesis: "compile language models against bytes, memory, latency, training time, energy, adaptation legality, and hardware topology",
        spec_path: options.spec.display().to_string(),
        spec_name: spec.name,
        spec_fingerprint: plan.variant_fingerprint,
        resource_contract,
        cost_model,
        artifact_ir,
        pack,
        pack_experiment,
        cache,
        cache_experiment,
        delta,
        train,
        optimizer_comm,
        kernel_forge,
        kernel_experiment,
        wind_tunnel,
        wind_experiment,
        scale_golf,
        platform_status,
        release_readiness,
        caveats: vec![
            "This is a deterministic local planner and proof/report surface, not final H100 timing or validation BPB evidence.".to_string(),
            "LQER, cache, delta, and export-gap scores are calibrated surrogates until backed by model-specific measurements.".to_string(),
            "Optimizer/communication equivalence is proven for shard-separable updates locally; NCCL overlap and distributed Muon performance still require target-backend traces.".to_string(),
            "KernelForge includes exact tiled CE CPU parity references; new GPU kernels still require backend lowering, parity, and performance runs.".to_string(),
        ],
        status,
    };

    match options.section {
        ModelGolfSection::Full => write_report_if_requested(&report, options.output.as_deref())?,
        ModelGolfSection::Pack => {
            write_report_if_requested(&report.pack, options.output.as_deref())?
        }
        ModelGolfSection::PackExperiment => {
            write_report_if_requested(&report.pack_experiment, options.output.as_deref())?
        }
        ModelGolfSection::PackSourceReport => {
            write_report_if_requested(&pack_source_report, options.output.as_deref())?
        }
        ModelGolfSection::LqerSourceReport => {
            write_report_if_requested(&lqer_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Cache => {
            write_report_if_requested(&report.cache, options.output.as_deref())?
        }
        ModelGolfSection::CacheExperiment => {
            write_report_if_requested(&report.cache_experiment, options.output.as_deref())?
        }
        ModelGolfSection::CacheSourceReport => {
            write_report_if_requested(&cache_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Delta => {
            write_report_if_requested(&report.delta, options.output.as_deref())?
        }
        ModelGolfSection::DeltaSourceReport => {
            write_report_if_requested(&delta_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Train => {
            write_report_if_requested(&report.train, options.output.as_deref())?
        }
        ModelGolfSection::TrainSourceReport => {
            write_report_if_requested(&train_source_report, options.output.as_deref())?
        }
        ModelGolfSection::ResourceCostSourceReport => {
            write_report_if_requested(&resource_cost_source_report, options.output.as_deref())?
        }
        ModelGolfSection::OptimizerCommSourceReport => {
            write_report_if_requested(&optimizer_comm_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Kernel => {
            write_report_if_requested(&report.kernel_forge, options.output.as_deref())?
        }
        ModelGolfSection::KernelExperiment => {
            write_report_if_requested(&report.kernel_experiment, options.output.as_deref())?
        }
        ModelGolfSection::KernelSourceReport => {
            write_report_if_requested(&kernel_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Wind => {
            write_report_if_requested(&report.wind_tunnel, options.output.as_deref())?
        }
        ModelGolfSection::WindExperiment => {
            write_report_if_requested(&report.wind_experiment, options.output.as_deref())?
        }
        ModelGolfSection::WindSourceReport => {
            write_report_if_requested(&wind_source_report, options.output.as_deref())?
        }
        ModelGolfSection::Scale => {
            write_report_if_requested(&report.scale_golf, options.output.as_deref())?
        }
        ModelGolfSection::ReleaseEvidence => {
            write_report_if_requested(&release_evidence_report, options.output.as_deref())?
        }
        ModelGolfSection::ReleaseCheck => {
            write_report_if_requested(&report.release_readiness, options.output.as_deref())?
        }
    }
    Ok(report)
}

fn modelgolf_tensor_groups(
    groups: &[pg_quant::CompiledQuantGroupManifest],
) -> Vec<ModelGolfTensorGroup> {
    groups
        .iter()
        .map(|group| {
            let rows = group.rows.unwrap_or(0);
            let cols = group.cols.unwrap_or(0);
            let elems = rows.saturating_mul(cols);
            let role = tensor_role_for_name(group.name);
            ModelGolfTensorGroup {
                name: group.name.to_string(),
                role,
                rows,
                cols,
                elems,
                current_bits: group.bits,
                current_weight_bytes: group.packed_weight_bytes.unwrap_or(0),
                scale_bytes: group.scale_bytes.unwrap_or(0),
                lqer_bytes: group.lqer_bytes.unwrap_or(0),
                sensitivity: role_sensitivity(role),
            }
        })
        .collect()
}

fn plan_pack(
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
    target_artifact_bytes: usize,
    quality_calibration: Option<&PackQualityCalibrationModel>,
) -> PackPlannerReport {
    let bucket_bytes = 1;
    let option_sets = groups
        .iter()
        .map(|group| precision_options_for_group(group, spec, quality_calibration))
        .collect::<Vec<_>>();
    let candidate_count = option_sets.iter().map(Vec::len).sum();
    let (feasible, selected_options, selected_bytes, selected_quality_loss) =
        select_pack_options(&option_sets, target_artifact_bytes);
    let quality_comparison = compare_pack_quality_plans(
        groups,
        spec,
        &option_sets,
        &selected_options,
        selected_bytes,
    );
    let lqer_candidates = lqer_candidates(groups, spec, &selected_options, quality_calibration);
    let lqer_proofs = prove_selected_lqer_groups(groups, spec, &selected_options);
    let mut notes = vec![
        "Multiple-choice knapsack is exact over integer byte estimates; dominated states are pruned without changing the optimum.".to_string(),
        "Option bytes are compression-adjusted artifact estimates; weight_bytes, scale_bytes, and lqer_bytes preserve raw component estimates.".to_string(),
        "Quality loss is an additive surrogate weighted by tensor role; calibrate with mini/full eval before claiming quality.".to_string(),
        "quality_comparison is a deterministic local equal-byte proxy over synthetic tensor fixtures; it is not held-out BPB/perplexity evidence.".to_string(),
    ];
    if quality_calibration.is_some() {
        notes.push("Quality calibration file was applied to role/group bit-loss estimates; release claims still require held-out BPB/perplexity evidence.".to_string());
    }
    if !feasible {
        notes.push("No candidate assignment fit the artifact byte contract; selected_options is the smallest local fallback.".to_string());
    }
    if lqer_proofs.is_empty() {
        notes.push("No selected option carries LQER bytes under this contract; LQER local proof is skipped.".to_string());
    } else {
        notes.push("lqer_proofs use deterministic quantization residuals and identity-curvature low-rank factors to verify rank-r correction and the activation/logit-error bound.".to_string());
    }
    PackPlannerReport {
        kind: "modelgolf_pack_plan",
        algorithm: "exact_multiple_choice_knapsack_plus_lqer_ranking",
        objective: "minimize additive quality-loss surrogate subject to artifact bytes",
        target_artifact_bytes,
        bucket_bytes,
        feasible,
        selected_bytes,
        selected_quality_loss,
        estimated_bytes_remaining: target_artifact_bytes as isize - selected_bytes as isize,
        selected_options,
        lqer_candidates,
        candidate_count,
        quality_calibration: quality_calibration
            .map(|model| model.report.clone())
            .unwrap_or_else(default_pack_quality_calibration_report),
        quality_comparison,
        lqer_proofs,
        artifact_proof: None,
        notes,
    }
}

fn select_pack_options(
    option_sets: &[Vec<PrecisionOptionReport>],
    target_artifact_bytes: usize,
) -> (bool, Vec<PrecisionOptionReport>, usize, f64) {
    let mut dp = BTreeMap::new();
    dp.insert(
        0usize,
        DpState {
            loss: 0.0,
            bytes: 0,
            option_indices: Vec::new(),
        },
    );
    for options in option_sets {
        let mut next = BTreeMap::<usize, DpState>::new();
        for state in dp.values() {
            for (option_index, option) in options.iter().enumerate() {
                let next_bytes = state.bytes.saturating_add(option.bytes);
                if next_bytes > target_artifact_bytes {
                    continue;
                }
                let mut option_indices = state.option_indices.clone();
                option_indices.push(option_index);
                let candidate = DpState {
                    loss: state.loss + option.estimated_quality_loss,
                    bytes: next_bytes,
                    option_indices,
                };
                if next
                    .get(&next_bytes)
                    .as_ref()
                    .map(|old| better_pack_state(&candidate, old))
                    .unwrap_or(true)
                {
                    next.insert(next_bytes, candidate);
                }
            }
        }
        dp = prune_dominated_pack_states(next);
    }
    let best = dp
        .into_values()
        .filter(|state| state.bytes <= target_artifact_bytes)
        .min_by(|a, b| compare_loss_then_bytes(a.loss, a.bytes, b.loss, b.bytes));
    if let Some(best) = best {
        let selected = best
            .option_indices
            .iter()
            .enumerate()
            .map(|(group_index, &option_index)| option_sets[group_index][option_index].clone())
            .collect::<Vec<_>>();
        (true, selected, best.bytes, best.loss)
    } else {
        let selected = option_sets
            .iter()
            .filter_map(|options| options.iter().min_by_key(|option| option.bytes).cloned())
            .collect::<Vec<_>>();
        let bytes = selected.iter().map(|option| option.bytes).sum();
        let loss = selected
            .iter()
            .map(|option| option.estimated_quality_loss)
            .sum();
        (false, selected, bytes, loss)
    }
}

fn default_pack_quality_calibration_report() -> PackQualityCalibrationReport {
    PackQualityCalibrationReport {
        kind: "modelgolf_pack_quality_calibration",
        source_path: None,
        applied: false,
        points: 0,
        default_scale: 1.0,
        mean_abs_error_before: None,
        mean_abs_error_after: None,
        max_abs_error_after: None,
        factors: Vec::new(),
        notes: vec![
            "No quality calibration file was supplied; Pack uses deterministic role/bit surrogate losses.".to_string(),
        ],
    }
}

fn compare_pack_quality_plans(
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
    option_sets: &[Vec<PrecisionOptionReport>],
    selected_options: &[PrecisionOptionReport],
    selected_bytes: usize,
) -> PackQualityComparisonReport {
    let comparison_budget_bytes = selected_bytes;
    let selected_plan_name = "mixed_plus_lqer_selected";
    let mut rows = Vec::new();

    if let Some(options) = uniform_pack_options(option_sets, 4, false) {
        rows.push(pack_quality_comparison_row(
            "uniform_q4_no_lqer",
            groups,
            spec,
            &options,
            comparison_budget_bytes,
            selected_bytes,
            vec!["Uniform Q4 baseline from the original Pack experiment protocol.".to_string()],
        ));
    }

    if let Some((bits, options)) =
        best_uniform_no_lqer_under_budget(option_sets, comparison_budget_bytes)
    {
        rows.push(pack_quality_comparison_row(
            &format!("uniform_q{bits}_no_lqer_equal_byte"),
            groups,
            spec,
            &options,
            comparison_budget_bytes,
            selected_bytes,
            vec![
                "Highest uniform no-LQER bit width that fits the selected plan byte budget."
                    .to_string(),
            ],
        ));
    }

    let no_lqer_sets = option_sets
        .iter()
        .map(|options| {
            options
                .iter()
                .filter(|option| option.lqer_bytes == 0)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (_, mixed_no_lqer, _, _) = select_pack_options(&no_lqer_sets, comparison_budget_bytes);
    if !mixed_no_lqer.is_empty() {
        rows.push(pack_quality_comparison_row(
            "mixed_no_lqer_equal_byte_dp",
            groups,
            spec,
            &mixed_no_lqer,
            comparison_budget_bytes,
            selected_bytes,
            vec![
                "Exact multiple-choice knapsack baseline with LQER options removed under the selected byte budget."
                    .to_string(),
            ],
        ));
    }

    rows.push(pack_quality_comparison_row(
        selected_plan_name,
        groups,
        spec,
        selected_options,
        comparison_budget_bytes,
        selected_bytes,
        vec![
            "Selected exact Pack plan, including LQER where the byte/quality surrogate chose it."
                .to_string(),
        ],
    ));

    let lqer_control = deterministic_lqer_control_options(option_sets, selected_options);
    if !lqer_control.is_empty() {
        let selected_lqer_count = selected_options
            .iter()
            .filter(|option| option.lqer_bytes > 0)
            .count();
        let control_lqer_count = lqer_control
            .iter()
            .filter(|option| option.lqer_bytes > 0)
            .count();
        let mut notes = vec![
            "Control plan moves the selected LQER group count to deterministic non-selected groups when possible; use as a random-LQER proxy, not as a stochastic average."
                .to_string(),
        ];
        if control_lqer_count < selected_lqer_count {
            notes.push(format!(
                "Only {control_lqer_count} non-selected LQER control groups were available for {selected_lqer_count} selected LQER groups; this row is a conservative no/partial-LQER fallback."
            ));
        }
        let control_name = if control_lqer_count == selected_lqer_count {
            "deterministic_lqer_control_equal_count"
        } else {
            "deterministic_lqer_control_insufficient_candidates"
        };
        rows.push(pack_quality_comparison_row(
            control_name,
            groups,
            spec,
            &lqer_control,
            comparison_budget_bytes,
            selected_bytes,
            notes,
        ));
    }

    rows.sort_by(|a, b| {
        a.plan_name
            .cmp(&b.plan_name)
            .then_with(|| a.artifact_bytes.cmp(&b.artifact_bytes))
    });
    let best_proxy_quality_plan = rows
        .iter()
        .filter(|row| row.fits_comparison_budget)
        .min_by(|a, b| {
            a.local_weighted_residual_mse_proxy
                .total_cmp(&b.local_weighted_residual_mse_proxy)
                .then_with(|| {
                    a.local_activation_ce_bound_proxy
                        .total_cmp(&b.local_activation_ce_bound_proxy)
                })
        })
        .map(|row| row.plan_name.clone());
    let selected_beats_uniform_best_proxy =
        pack_comparison_beats(&rows, selected_plan_name, "uniform_q", |row| {
            row.plan_name.contains("_no_lqer_equal_byte")
        });
    let selected_beats_mixed_no_lqer_proxy = pack_comparison_beats(
        &rows,
        selected_plan_name,
        "mixed_no_lqer_equal_byte_dp",
        |_| true,
    );
    let selected_beats_lqer_control_proxy = pack_comparison_beats(
        &rows,
        selected_plan_name,
        "deterministic_lqer_control",
        |_| true,
    );

    PackQualityComparisonReport {
        kind: "modelgolf_pack_equal_byte_quality_comparison",
        comparison_budget_bytes,
        selected_plan_name,
        rows,
        best_proxy_quality_plan,
        selected_beats_uniform_best_proxy,
        selected_beats_mixed_no_lqer_proxy,
        selected_beats_lqer_control_proxy,
        evidence_boundary: "deterministic local residual/activation-bound proxy on synthetic tensor fixtures; not held-out BPB, perplexity, decode speed, or trained-artifact evidence",
    }
}

fn uniform_pack_options(
    option_sets: &[Vec<PrecisionOptionReport>],
    bits: u8,
    lqer: bool,
) -> Option<Vec<PrecisionOptionReport>> {
    option_sets
        .iter()
        .map(|options| {
            options
                .iter()
                .find(|option| option.bits == bits && (option.lqer_bytes > 0) == lqer)
                .cloned()
        })
        .collect()
}

fn best_uniform_no_lqer_under_budget(
    option_sets: &[Vec<PrecisionOptionReport>],
    budget_bytes: usize,
) -> Option<(u8, Vec<PrecisionOptionReport>)> {
    (4..=8).rev().find_map(|bits| {
        let options = uniform_pack_options(option_sets, bits, false)?;
        let bytes = options.iter().map(|option| option.bytes).sum::<usize>();
        (bytes <= budget_bytes).then_some((bits, options))
    })
}

fn deterministic_lqer_control_options(
    option_sets: &[Vec<PrecisionOptionReport>],
    selected_options: &[PrecisionOptionReport],
) -> Vec<PrecisionOptionReport> {
    let selected_lqer_groups = selected_options
        .iter()
        .filter(|option| option.lqer_bytes > 0)
        .map(|option| option.group.clone())
        .collect::<Vec<_>>();
    let target_lqer_count = selected_lqer_groups.len();
    if target_lqer_count == 0 {
        return selected_options.to_vec();
    }

    let mut control = selected_options
        .iter()
        .map(|selected| {
            if selected.lqer_bytes == 0 {
                return selected.clone();
            }
            option_sets
                .iter()
                .flatten()
                .find(|option| {
                    option.group == selected.group
                        && option.bits == selected.bits
                        && option.lqer_bytes == 0
                })
                .cloned()
                .unwrap_or_else(|| selected.clone())
        })
        .collect::<Vec<_>>();

    let mut candidates = option_sets
        .iter()
        .flatten()
        .filter(|option| {
            option.lqer_bytes > 0
                && !selected_lqer_groups
                    .iter()
                    .any(|selected| selected == &option.group)
        })
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        deterministic_pack_control_hash(&a.group, a.bits)
            .cmp(&deterministic_pack_control_hash(&b.group, b.bits))
            .then_with(|| a.group.cmp(&b.group))
            .then_with(|| a.bits.cmp(&b.bits))
    });

    let mut used_groups = Vec::<String>::new();
    for candidate in candidates {
        if used_groups.iter().any(|group| group == &candidate.group) {
            continue;
        }
        let Some(slot) = control
            .iter_mut()
            .find(|option| option.group == candidate.group)
        else {
            continue;
        };
        *slot = candidate.clone();
        used_groups.push(candidate.group);
        if used_groups.len() >= target_lqer_count {
            break;
        }
    }
    control
}

fn deterministic_pack_control_hash(group: &str, bits: u8) -> u64 {
    group.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    }) ^ bits as u64
}

fn pack_quality_comparison_row(
    plan_name: &str,
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
    options: &[PrecisionOptionReport],
    budget_bytes: usize,
    selected_bytes: usize,
    notes: Vec<String>,
) -> PackQualityComparisonRowReport {
    let artifact_bytes = options.iter().map(|option| option.bytes).sum::<usize>();
    let estimated_quality_loss = options
        .iter()
        .map(|option| option.estimated_quality_loss)
        .sum::<f64>();
    let mut local_weighted_residual_mse_proxy = 0.0;
    let mut local_activation_ce_bound_proxy = 0.0;
    for option in options {
        if let Some(group) = groups.iter().find(|group| group.name == option.group) {
            let (mse, ce_bound) = pack_option_local_quality_proxy(group, spec, option);
            local_weighted_residual_mse_proxy += mse;
            local_activation_ce_bound_proxy += ce_bound;
        }
    }
    let mut lqer_groups = options
        .iter()
        .filter(|option| option.lqer_bytes > 0)
        .map(|option| option.group.clone())
        .collect::<Vec<_>>();
    lqer_groups.sort();
    PackQualityComparisonRowReport {
        plan_name: plan_name.to_string(),
        artifact_bytes,
        byte_delta_vs_selected: artifact_bytes as isize - selected_bytes as isize,
        fits_comparison_budget: artifact_bytes <= budget_bytes,
        estimated_quality_loss,
        local_weighted_residual_mse_proxy,
        local_activation_ce_bound_proxy,
        lqer_group_count: lqer_groups.len(),
        lqer_groups,
        notes,
    }
}

fn pack_option_local_quality_proxy(
    group: &ModelGolfTensorGroup,
    spec: &RunSpec,
    option: &PrecisionOptionReport,
) -> (f64, f64) {
    let requested_rank = spec.quant.lqer.rank.max(1);
    let rows = lqer_proof_dim(group.rows, requested_rank);
    let cols = lqer_proof_dim(group.cols, requested_rank);
    if rows == 0 || cols == 0 {
        return (0.0, 0.0);
    }
    let weights = deterministic_lqer_weights(group, rows, cols, option.bits);
    let quantized = quantize_rows_symmetric(&weights, rows, cols, option.bits);
    let corrected = if option.lqer_bytes > 0 {
        lqer_corrected_weights(&weights, &quantized, rows, cols, requested_rank)
    } else {
        quantized
    };
    let residual_sq = weights
        .iter()
        .zip(corrected.iter())
        .map(|(weight, corrected)| {
            let err = (*weight as f64) - (*corrected as f64);
            err * err
        })
        .sum::<f64>();
    let proof_elems = (rows * cols).max(1) as f64;
    let elems_m = (group.elems as f64 / 1_000_000.0).max(1e-6);
    let weighted_mse = residual_sq / proof_elems * group.sensitivity * elems_m;
    let activation = deterministic_modelgolf_values(cols, 0.211 + option.bits as f32 * 0.017);
    let activation_norm = l2_norm(&activation) as f64;
    let ce_bound = 2.0 * activation_norm * residual_sq.sqrt() * group.sensitivity * elems_m;
    (weighted_mse, ce_bound)
}

fn lqer_corrected_weights(
    weights: &[f32],
    quantized: &[f32],
    rows: usize,
    cols: usize,
    requested_rank: usize,
) -> Vec<f32> {
    use pg_kernels::deltagolf::{deltagolf_materialize_delta, deltagolf_weighted_low_rank_delta};

    let residual = weights
        .iter()
        .zip(quantized.iter())
        .map(|(weight, quantized)| weight - quantized)
        .collect::<Vec<_>>();
    let out_curvature = vec![1.0; rows];
    let in_curvature = vec![1.0; cols];
    let Ok(factors) = deltagolf_weighted_low_rank_delta(
        &residual,
        &out_curvature,
        &in_curvature,
        rows,
        cols,
        requested_rank.min(rows).min(cols),
    ) else {
        return quantized.to_vec();
    };
    let mut correction = vec![0.0f32; residual.len()];
    if deltagolf_materialize_delta(&factors, &mut correction).is_err() {
        return quantized.to_vec();
    }
    quantized
        .iter()
        .zip(correction.iter())
        .map(|(quantized, correction)| quantized + correction)
        .collect()
}

fn pack_comparison_beats(
    rows: &[PackQualityComparisonRowReport],
    selected_name: &str,
    candidate_prefix: &str,
    candidate_filter: impl Fn(&PackQualityComparisonRowReport) -> bool,
) -> Option<bool> {
    let selected = rows.iter().find(|row| row.plan_name == selected_name)?;
    let candidate = rows
        .iter()
        .filter(|row| row.plan_name.starts_with(candidate_prefix))
        .filter(|row| candidate_filter(row))
        .min_by(|a, b| {
            a.local_weighted_residual_mse_proxy
                .total_cmp(&b.local_weighted_residual_mse_proxy)
        })?;
    Some(
        selected.local_weighted_residual_mse_proxy <= candidate.local_weighted_residual_mse_proxy
            && selected.local_activation_ce_bound_proxy
                <= candidate.local_activation_ce_bound_proxy,
    )
}

fn prove_selected_lqer_groups(
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
    selected_options: &[PrecisionOptionReport],
) -> Vec<PackLqerProofReport> {
    selected_options
        .iter()
        .filter(|option| option.lqer_bytes > 0)
        .filter_map(|option| {
            groups
                .iter()
                .find(|group| group.name == option.group)
                .map(|group| prove_lqer_group(group, option, spec))
        })
        .collect()
}

fn prove_lqer_group(
    group: &ModelGolfTensorGroup,
    option: &PrecisionOptionReport,
    spec: &RunSpec,
) -> PackLqerProofReport {
    use pg_kernels::deltagolf::{deltagolf_materialize_delta, deltagolf_weighted_low_rank_delta};

    let requested_rank = spec.quant.lqer.rank.max(1);
    let proof_rows = lqer_proof_dim(group.rows, requested_rank);
    let proof_cols = lqer_proof_dim(group.cols, requested_rank);
    let actual_rank_limit = requested_rank.min(proof_rows).min(proof_cols);
    let weights = deterministic_lqer_weights(group, proof_rows, proof_cols, option.bits);
    let quantized = quantize_rows_symmetric(&weights, proof_rows, proof_cols, option.bits);
    let residual = weights
        .iter()
        .zip(quantized.iter())
        .map(|(weight, quantized)| weight - quantized)
        .collect::<Vec<_>>();
    let ones_rows = vec![1.0f32; proof_rows];
    let ones_cols = vec![1.0f32; proof_cols];
    let delta = deltagolf_weighted_low_rank_delta(
        &residual,
        &ones_rows,
        &ones_cols,
        proof_rows,
        proof_cols,
        actual_rank_limit,
    )
    .expect("valid deterministic LQER residual proof");
    let mut reconstructed = vec![0.0f32; residual.len()];
    deltagolf_materialize_delta(&delta, &mut reconstructed)
        .expect("valid deterministic LQER materialization");

    let residual_frobenius_before = l2_norm(&residual);
    let residual_after = residual
        .iter()
        .zip(reconstructed.iter())
        .map(|(residual, reconstructed)| residual - reconstructed)
        .collect::<Vec<_>>();
    let residual_frobenius_after = l2_norm(&residual_after);
    let lower_rank_residual_frobenius = if actual_rank_limit > 1 {
        let lower = deltagolf_weighted_low_rank_delta(
            &residual,
            &ones_rows,
            &ones_cols,
            proof_rows,
            proof_cols,
            actual_rank_limit - 1,
        )
        .expect("valid deterministic lower-rank LQER proof");
        let mut lower_reconstructed = vec![0.0f32; residual.len()];
        deltagolf_materialize_delta(&lower, &mut lower_reconstructed)
            .expect("valid deterministic lower-rank LQER materialization");
        Some(l2_norm(
            &residual
                .iter()
                .zip(lower_reconstructed.iter())
                .map(|(residual, reconstructed)| residual - reconstructed)
                .collect::<Vec<_>>(),
        ))
    } else {
        Some(residual_frobenius_before)
    };
    let residual_reduction_pct = if residual_frobenius_before > 0.0 {
        100.0 * (1.0 - residual_frobenius_after / residual_frobenius_before)
    } else {
        0.0
    };
    let selected_rank_no_worse_than_lower_rank = lower_rank_residual_frobenius
        .map(|lower| residual_frobenius_after <= lower + 1e-9)
        .unwrap_or(true);
    let activation = deterministic_modelgolf_values(proof_cols, 0.119 + option.bits as f32 * 0.007);
    let activation_l2_norm = l2_norm(&activation);
    let ce_linf_bound_before = 2.0 * activation_l2_norm * residual_frobenius_before;
    let ce_linf_bound_after = 2.0 * activation_l2_norm * residual_frobenius_after;
    let ce_bound_reduction_pct = if ce_linf_bound_before > 0.0 {
        100.0 * (1.0 - ce_linf_bound_after / ce_linf_bound_before)
    } else {
        0.0
    };
    let finite = residual_frobenius_before.is_finite()
        && residual_frobenius_after.is_finite()
        && lower_rank_residual_frobenius
            .map(f64::is_finite)
            .unwrap_or(true)
        && residual_reduction_pct.is_finite()
        && activation_l2_norm.is_finite()
        && ce_linf_bound_before.is_finite()
        && ce_linf_bound_after.is_finite()
        && ce_bound_reduction_pct.is_finite();

    PackLqerProofReport {
        kind: "modelgolf_pack_lqer_local_proof",
        group: group.name.clone(),
        role: group.role,
        bits: option.bits,
        requested_rank,
        actual_rank: delta.rank,
        source_rows: group.rows,
        source_cols: group.cols,
        proof_rows,
        proof_cols,
        residual_frobenius_before,
        residual_frobenius_after,
        lower_rank_residual_frobenius,
        residual_reduction_pct,
        selected_rank_no_worse_than_lower_rank,
        activation_l2_norm,
        ce_linf_bound_before,
        ce_linf_bound_after,
        ce_bound_reduction_pct,
        finite,
        evidence_boundary: "local deterministic LQER residual proof; not calibrated tensor sensitivity, trained-artifact BPB, or production-scale randomized SVD evidence",
    }
}

fn lqer_proof_dim(source_dim: usize, requested_rank: usize) -> usize {
    let source_dim = source_dim.max(1);
    let rank_floor = requested_rank.min(source_dim).max(1);
    source_dim.min(64).max(rank_floor)
}

fn deterministic_lqer_weights(
    group: &ModelGolfTensorGroup,
    rows: usize,
    cols: usize,
    bits: u8,
) -> Vec<f32> {
    let phase = 0.037
        + tensor_role_label(group.role).len() as f32 * 0.003
        + bits as f32 * 0.011
        + (group.name.bytes().fold(0usize, |acc, b| acc + b as usize) % 17) as f32 * 0.002;
    let mut values = deterministic_modelgolf_values(rows * cols, phase);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            values[idx] += 0.04 * ((row + col * 3) % 9) as f32 - 0.16;
        }
    }
    values
}

fn quantize_rows_symmetric(weights: &[f32], rows: usize, cols: usize, bits: u8) -> Vec<f32> {
    let qmax = ((1i32 << (bits.saturating_sub(1))) - 1).max(1) as f32;
    let mut quantized = vec![0.0f32; weights.len()];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &weights[offset..offset + cols];
        let max_abs = row_values
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let scale = max_abs / qmax;
        for col in 0..cols {
            let q = (weights[offset + col] / scale).round().clamp(-qmax, qmax);
            quantized[offset + col] = q * scale;
        }
    }
    quantized
}

fn load_pack_quality_calibration(
    path: &Path,
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
) -> PgResult<PackQualityCalibrationModel> {
    let text = std::fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to parse ModelGolf quality calibration JSON {}: {err}",
            path.display()
        ))
    })?;
    let points = if value.is_array() {
        serde_json::from_value::<Vec<PackQualityCalibrationPoint>>(value)
    } else {
        serde_json::from_value::<PackQualityCalibrationFile>(value).map(|file| file.points)
    }
    .map_err(|err| {
        PgError::InvalidOp(format!(
            "invalid ModelGolf quality calibration schema {}: {err}",
            path.display()
        ))
    })?;

    let evals = points
        .iter()
        .enumerate()
        .map(|(index, point)| evaluate_calibration_point(index, point, groups, spec))
        .collect::<PgResult<Vec<_>>>()?;

    if evals.is_empty() {
        return Ok(PackQualityCalibrationModel {
            report: PackQualityCalibrationReport {
                source_path: Some(path.display().to_string()),
                notes: vec![
                    "Quality calibration file contained no points; Pack uses deterministic role/bit surrogate losses.".to_string(),
                ],
                ..default_pack_quality_calibration_report()
            },
            group_scales: BTreeMap::new(),
            role_scales: BTreeMap::new(),
        });
    }

    let mut group_acc = BTreeMap::<(String, u8, bool), CalibrationAccumulator>::new();
    let mut role_acc = BTreeMap::<(TensorRole, u8, bool), CalibrationAccumulator>::new();
    let mut scale_sum = 0.0;
    for eval in &evals {
        let scale = calibration_scale(eval.estimated, eval.measured)?;
        scale_sum += scale;
        if let Some(group) = &eval.group {
            accumulate_calibration_scale(
                group_acc
                    .entry((group.clone(), eval.bits, eval.lqer))
                    .or_default(),
                scale,
            );
        }
        accumulate_calibration_scale(
            role_acc
                .entry((eval.role, eval.bits, eval.lqer))
                .or_default(),
            scale,
        );
    }
    let default_scale = clamp_calibration_scale(scale_sum / evals.len() as f64);
    let group_scales = group_acc
        .iter()
        .map(|(key, acc)| (key.clone(), average_calibration_scale(acc)))
        .collect::<BTreeMap<_, _>>();
    let role_scales = role_acc
        .iter()
        .map(|(key, acc)| (*key, average_calibration_scale(acc)))
        .collect::<BTreeMap<_, _>>();
    let mut factors = Vec::new();
    factors.push(PackQualityCalibrationFactorReport {
        scope: "global".to_string(),
        bits: 0,
        lqer: false,
        samples: evals.len(),
        scale: default_scale,
    });
    factors.extend(group_acc.iter().map(|((group, bits, lqer), acc)| {
        PackQualityCalibrationFactorReport {
            scope: format!("group:{group}"),
            bits: *bits,
            lqer: *lqer,
            samples: acc.samples,
            scale: average_calibration_scale(acc),
        }
    }));
    factors.extend(role_acc.iter().map(|((role, bits, lqer), acc)| {
        PackQualityCalibrationFactorReport {
            scope: format!("role:{}", tensor_role_label(*role)),
            bits: *bits,
            lqer: *lqer,
            samples: acc.samples,
            scale: average_calibration_scale(acc),
        }
    }));
    factors.sort_by(|a, b| {
        a.scope
            .cmp(&b.scope)
            .then_with(|| a.bits.cmp(&b.bits))
            .then_with(|| a.lqer.cmp(&b.lqer))
    });

    let mut before_sum = 0.0;
    let mut after_sum = 0.0;
    let mut max_after = 0.0f64;
    for eval in &evals {
        let calibrated = apply_pack_calibration_with_maps(
            eval.estimated,
            eval.group.as_deref(),
            eval.role,
            eval.bits,
            eval.lqer,
            default_scale,
            &group_scales,
            &role_scales,
        );
        before_sum += (eval.estimated - eval.measured).abs();
        let after = (calibrated - eval.measured).abs();
        after_sum += after;
        max_after = max_after.max(after);
    }

    Ok(PackQualityCalibrationModel {
        report: PackQualityCalibrationReport {
            kind: "modelgolf_pack_quality_calibration",
            source_path: Some(path.display().to_string()),
            applied: true,
            points: evals.len(),
            default_scale,
            mean_abs_error_before: Some(before_sum / evals.len() as f64),
            mean_abs_error_after: Some(after_sum / evals.len() as f64),
            max_abs_error_after: Some(max_after),
            factors,
            notes: vec![
                "Calibration scales are fit from measured quality-loss deltas divided by planner surrogate deltas.".to_string(),
                "Use held-out full validation BPB/perplexity before turning calibrated Pack scores into release claims.".to_string(),
            ],
        },
        group_scales,
        role_scales,
    })
}

fn evaluate_calibration_point(
    index: usize,
    point: &PackQualityCalibrationPoint,
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
) -> PgResult<CalibrationPointEval> {
    if !(1..=16).contains(&point.bits) {
        return Err(PgError::InvalidOp(format!(
            "quality calibration point {index} has invalid bits {}; expected 1..=16",
            point.bits
        )));
    }
    let measured = point.measured_quality_loss.ok_or_else(|| {
        PgError::InvalidOp(format!(
            "quality calibration point {index} missing measured_quality_loss"
        ))
    })?;
    if !measured.is_finite() || measured < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "quality calibration point {index} measured_quality_loss must be finite and non-negative"
        )));
    }

    let (group_name, role, estimated) = if let Some(group_name) = point.group.as_deref() {
        let group = groups
            .iter()
            .find(|group| group.name == group_name)
            .ok_or_else(|| {
                PgError::InvalidOp(format!(
                    "quality calibration point {index} references unknown group {group_name}"
                ))
            })?;
        if let Some(role) = point.role
            && role != group.role
        {
            return Err(PgError::InvalidOp(format!(
                "quality calibration point {index} role {} does not match group {group_name} role {}",
                tensor_role_label(role),
                tensor_role_label(group.role)
            )));
        }
        (
            Some(group.name.clone()),
            group.role,
            point.estimated_quality_loss.unwrap_or_else(|| {
                quality_loss_surrogate(group, point.bits, point.lqer, spec.quant.lqer.rank)
            }),
        )
    } else {
        let role = point.role.ok_or_else(|| {
            PgError::InvalidOp(format!(
                "quality calibration point {index} without group must include role"
            ))
        })?;
        let estimated = point.estimated_quality_loss.ok_or_else(|| {
            PgError::InvalidOp(format!(
                "quality calibration point {index} without group must include estimated_quality_loss"
            ))
        })?;
        (None, role, estimated)
    };

    if !estimated.is_finite() || estimated < 0.0 {
        return Err(PgError::InvalidOp(format!(
            "quality calibration point {index} estimated_quality_loss must be finite and non-negative"
        )));
    }
    if estimated <= 1e-12 && measured > 1e-12 {
        return Err(PgError::InvalidOp(format!(
            "quality calibration point {index} has nonzero measured loss but zero estimated loss"
        )));
    }

    Ok(CalibrationPointEval {
        group: group_name,
        role,
        bits: point.bits,
        lqer: point.lqer,
        estimated,
        measured,
    })
}

fn calibration_scale(estimated: f64, measured: f64) -> PgResult<f64> {
    if estimated <= 1e-12 {
        return Ok(1.0);
    }
    Ok(clamp_calibration_scale(measured / estimated))
}

fn accumulate_calibration_scale(acc: &mut CalibrationAccumulator, scale: f64) {
    acc.samples += 1;
    acc.scale_sum += scale;
}

fn average_calibration_scale(acc: &CalibrationAccumulator) -> f64 {
    if acc.samples == 0 {
        1.0
    } else {
        clamp_calibration_scale(acc.scale_sum / acc.samples as f64)
    }
}

fn clamp_calibration_scale(scale: f64) -> f64 {
    if !scale.is_finite() {
        1.0
    } else {
        scale.clamp(0.05, 20.0)
    }
}

fn calibrated_quality_loss(
    base_loss: f64,
    group: &ModelGolfTensorGroup,
    bits: u8,
    lqer: bool,
    calibration: Option<&PackQualityCalibrationModel>,
) -> f64 {
    calibration
        .map(|model| {
            apply_pack_calibration_with_maps(
                base_loss,
                Some(group.name.as_str()),
                group.role,
                bits,
                lqer,
                model.report.default_scale,
                &model.group_scales,
                &model.role_scales,
            )
        })
        .unwrap_or(base_loss)
}

#[allow(clippy::too_many_arguments)]
fn apply_pack_calibration_with_maps(
    base_loss: f64,
    group: Option<&str>,
    role: TensorRole,
    bits: u8,
    lqer: bool,
    default_scale: f64,
    group_scales: &BTreeMap<(String, u8, bool), f64>,
    role_scales: &BTreeMap<(TensorRole, u8, bool), f64>,
) -> f64 {
    let group_scale = group.and_then(|group| group_scales.get(&(group.to_string(), bits, lqer)));
    let scale = group_scale
        .copied()
        .or_else(|| role_scales.get(&(role, bits, lqer)).copied())
        .unwrap_or(default_scale);
    base_loss * scale
}

fn tensor_role_label(role: TensorRole) -> &'static str {
    match role {
        TensorRole::AttentionQ => "attention_q",
        TensorRole::AttentionO => "attention_o",
        TensorRole::KeyCacheProjection => "key_cache_projection",
        TensorRole::ValueCacheProjection => "value_cache_projection",
        TensorRole::MlpUp => "mlp_up",
        TensorRole::MlpDown => "mlp_down",
        TensorRole::TokenEmbedding => "token_embedding",
        TensorRole::AttentionGate => "attention_gate",
        TensorRole::Other => "other",
    }
}

fn prove_pack_artifact(
    spec: &RunSpec,
    plan: &ExecutionPlan,
    artifact_path: &Path,
    target_artifact_bytes: usize,
) -> PgResult<PackArtifactProofReport> {
    if let Some(parent) = artifact_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let config = spec.model.to_model_config();
    let mut model = GptModel::new(config.clone());
    model.fill_deterministic();
    let (input_ids, targets) = artifact_proof_tokens(config.vocab_size, config.train_seq_len);
    let pre_export_loss = model_smoke_loss(&model, &input_ids, &targets)?;

    let artifact_bytes = pg_quant::export::export_model_with_spec(
        &model,
        &spec.quant,
        &plan.variant_fingerprint,
        artifact_path,
    )?;

    let mut loaded = GptModel::new(config);
    pg_quant::export::load_artifact_with_spec(artifact_path, &mut loaded, &spec.quant, true)?;
    let post_reload_loss = model_smoke_loss(&loaded, &input_ids, &targets)?;
    let finite_loss = pre_export_loss.is_finite() && post_reload_loss.is_finite();

    Ok(PackArtifactProofReport {
        kind: "modelgolf_pack_artifact_proof",
        artifact_path: artifact_path.display().to_string(),
        artifact_bytes,
        target_artifact_bytes,
        artifact_budget_ok: artifact_bytes <= target_artifact_bytes,
        strict_reload_ok: true,
        variant_fingerprint: plan.variant_fingerprint.clone(),
        smoke_tokens: input_ids.len(),
        pre_export_loss,
        post_reload_loss,
        loss_delta_abs: (post_reload_loss - pre_export_loss).abs(),
        finite_loss,
        evidence_boundary: "local deterministic export/reload format proof; not trained artifact byte proof, full validation BPB, or H100 evidence",
    })
}

fn artifact_proof_tokens(vocab_size: usize, train_seq_len: usize) -> (Vec<u32>, Vec<u32>) {
    let tokens = train_seq_len.clamp(2, 16);
    let vocab = vocab_size.max(2);
    let input_ids = (0..tokens)
        .map(|i| ((i * 7 + 3) % vocab) as u32)
        .collect::<Vec<_>>();
    let targets = (0..tokens)
        .map(|i| ((i * 7 + 4) % vocab) as u32)
        .collect::<Vec<_>>();
    (input_ids, targets)
}

fn model_smoke_loss(model: &GptModel, input_ids: &[u32], targets: &[u32]) -> PgResult<f64> {
    if input_ids.len() != targets.len() {
        return Err(PgError::InvalidOp(format!(
            "artifact proof input/target length mismatch: {} vs {}",
            input_ids.len(),
            targets.len()
        )));
    }
    let mut buffer = ForwardBuffer::new(&model.config, input_ids.len());
    model.forward(input_ids, &mut buffer);
    Ok(model.compute_loss(targets, &buffer) as f64)
}

fn run_pack_measured_experiment(
    spec: &RunSpec,
    spec_path: &Path,
    target_artifact_bytes: usize,
    variant_fingerprint: &str,
) -> PgResult<PackMeasuredExperimentReport> {
    let config = pack_experiment_model_config(spec);
    let mut model = GptModel::new(config.clone());
    model.fill_deterministic();
    let (input_ids, targets) = artifact_proof_tokens(config.vocab_size, config.eval_seq_len);
    let pre_export_loss = model_smoke_loss(&model, &input_ids, &targets)?;
    let variants = pack_measured_experiment_variants(spec, target_artifact_bytes);
    let target_plan_name = "mixed_plus_lqer_top3";
    let mut rows = variants
        .iter()
        .map(|variant| {
            run_pack_measured_experiment_row(
                &model,
                &input_ids,
                &targets,
                pre_export_loss,
                target_artifact_bytes,
                variant_fingerprint,
                variant,
            )
        })
        .collect::<PgResult<Vec<_>>>()?;

    let target_artifact_bytes = rows
        .iter()
        .find(|row| row.plan_name == target_plan_name)
        .map(|row| row.artifact_bytes)
        .unwrap_or(0);
    for row in &mut rows {
        row.byte_delta_vs_target = row.artifact_bytes as isize - target_artifact_bytes as isize;
    }
    rows.sort_by(|a, b| a.plan_name.cmp(&b.plan_name));

    let best_measured_bpb_proxy_plan = rows
        .iter()
        .filter(|row| row.finite)
        .min_by(|a, b| {
            a.measured_bpb_proxy
                .total_cmp(&b.measured_bpb_proxy)
                .then_with(|| a.artifact_bytes.cmp(&b.artifact_bytes))
        })
        .map(|row| row.plan_name.clone());
    let target_beats_uniform_q4_bpb_proxy =
        pack_experiment_beats(&rows, target_plan_name, "uniform_q4");
    let target_beats_mixed_no_lqer_bpb_proxy =
        pack_experiment_beats(&rows, target_plan_name, "mixed_q4_q6_no_lqer");
    let target_beats_random_lqer_control_bpb_proxy =
        pack_experiment_beats(&rows, target_plan_name, "mixed_plus_random_lqer_control");
    let status = if !rows.is_empty() && rows.iter().all(|row| row.finite) {
        "local_measured_experiment_ready"
    } else {
        "local_measured_experiment_failed"
    }
    .to_string();

    Ok(PackMeasuredExperimentReport {
        kind: "modelgolf_pack_measured_lqer_experiment",
        protocol: "uniform_q4_vs_mixed_q4_q6_vs_mixed_lqer_top3_vs_deterministic_random_lqer_control",
        source_spec_path: spec_path.display().to_string(),
        source_spec_name: spec.name.clone(),
        eval_model_shape: format!(
            "layers={} dim={} heads={} kv_heads={} vocab={} tokens={}",
            config.num_layers,
            config.model_dim,
            config.num_heads,
            config.num_kv_heads,
            config.vocab_size,
            input_ids.len()
        ),
        eval_tokens: input_ids.len(),
        target_plan_name,
        rows,
        best_measured_bpb_proxy_plan,
        target_beats_uniform_q4_bpb_proxy,
        target_beats_mixed_no_lqer_bpb_proxy,
        target_beats_random_lqer_control_bpb_proxy,
        status,
        evidence_boundary: "local deterministic export/reload/eval table on a bounded toy model derived from the spec; not a pretrained small-model, held-out BPB/perplexity, or decode-speed release benchmark",
    })
}

fn pack_experiment_model_config(spec: &RunSpec) -> pg_model::ModelConfig {
    let mut config = spec.model.to_model_config();
    config.vocab_size = config.vocab_size.clamp(32, 128);
    config.num_layers = config.num_layers.clamp(1, 2);
    config.model_dim = 32;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.head_dim = config.model_dim / config.num_heads;
    config.mlp_mult = 2.0;
    config.mlp_dim = 64;
    config.rope_dims = 8;
    config.xsa_last_n = 0;
    config.recurrence_enabled = false;
    config.recurrence_start_layer = 0;
    config.recurrence_repeat_layers = 0;
    config.parallel_residual = false;
    config.parallel_residual_start_layer = 0;
    config.attn_out_gate_enabled = false;
    config.sparse_attn_gate_enabled = false;
    config.vrl_enabled = false;
    config.ve_enabled = false;
    config.ve_dim = 0;
    config.ve_layers = Vec::new();
    config.bigram_vocab_size = 0;
    config.bigram_dim = 0;
    config.train_seq_len = 32;
    config.eval_seq_len = 32;
    config
}

#[derive(Debug, Clone)]
struct PackMeasuredExperimentVariant {
    plan_name: &'static str,
    quant_spec: QuantSpec,
    lqer_selection_policy: &'static str,
    forced_lqer_groups: Option<Vec<&'static str>>,
    notes: Vec<String>,
}

fn pack_measured_experiment_variants(
    spec: &RunSpec,
    target_artifact_bytes: usize,
) -> Vec<PackMeasuredExperimentVariant> {
    let mut uniform_q4 = spec.quant.clone();
    uniform_q4.scheme = QuantScheme::GptqLiteInt6;
    uniform_q4.matrix_bits = 4;
    uniform_q4.mlp_bits = 4;
    uniform_q4.embed_bits = 4;
    uniform_q4.attn_gate_bits = 8;
    uniform_q4.target_artifact_bytes = target_artifact_bytes;
    uniform_q4.lqer.enabled = false;
    uniform_q4.lqer.top_k = 0;

    let mut mixed_no_lqer = spec.quant.clone();
    mixed_no_lqer.scheme = QuantScheme::MixedInt5Int6;
    mixed_no_lqer.matrix_bits = 5;
    mixed_no_lqer.mlp_bits = 4;
    mixed_no_lqer.embed_bits = 6;
    mixed_no_lqer.attn_gate_bits = 8;
    mixed_no_lqer.target_artifact_bytes = target_artifact_bytes;
    mixed_no_lqer.lqer.enabled = false;
    mixed_no_lqer.lqer.top_k = 0;

    let mut mixed_lqer = mixed_no_lqer.clone();
    mixed_lqer.lqer.enabled = true;
    mixed_lqer.lqer.rank = mixed_lqer.lqer.rank.clamp(1, 4);
    mixed_lqer.lqer.top_k = 3;
    mixed_lqer.lqer.a_bits = mixed_lqer.lqer.a_bits.clamp(2, 8);
    mixed_lqer.lqer.b_bits = mixed_lqer.lqer.b_bits.clamp(4, 8);
    mixed_lqer.lqer.group_size = mixed_lqer.lqer.group_size.max(16);

    let random_control = mixed_lqer.clone();

    vec![
        PackMeasuredExperimentVariant {
            plan_name: "uniform_q4",
            quant_spec: uniform_q4,
            lqer_selection_policy: "none",
            forced_lqer_groups: None,
            notes: vec!["Uniform 4-bit baseline from the plan experiment table.".to_string()],
        },
        PackMeasuredExperimentVariant {
            plan_name: "mixed_q4_q6_no_lqer",
            quant_spec: mixed_no_lqer,
            lqer_selection_policy: "none",
            forced_lqer_groups: None,
            notes: vec!["Mixed 4/5/6-bit planner baseline without LQER payloads.".to_string()],
        },
        PackMeasuredExperimentVariant {
            plan_name: "mixed_plus_lqer_top3",
            quant_spec: mixed_lqer,
            lqer_selection_policy: "residual_score_top_k",
            forced_lqer_groups: None,
            notes: vec![
                "Target plan: mixed precision plus exporter-selected top-3 residual-score LQER groups.".to_string(),
            ],
        },
        PackMeasuredExperimentVariant {
            plan_name: "mixed_plus_random_lqer_control",
            quant_spec: random_control,
            lqer_selection_policy: "deterministic_hash_control",
            forced_lqer_groups: Some(vec!["kv_bank.v", "mlp_up_bank", "qo_bank.q"]),
            notes: vec![
                "Control plan: same mixed precision and LQER rank/bytes, but with deterministic non-score-selected LQER groups.".to_string(),
            ],
        },
    ]
}

fn run_pack_measured_experiment_row(
    model: &GptModel,
    input_ids: &[u32],
    targets: &[u32],
    pre_export_loss: f64,
    target_artifact_bytes: usize,
    variant_fingerprint: &str,
    variant: &PackMeasuredExperimentVariant,
) -> PgResult<PackMeasuredExperimentRowReport> {
    let artifact_path = std::env::temp_dir().join(format!(
        "modelgolf_pack_experiment_{}_{}.pgrs",
        std::process::id(),
        variant.plan_name
    ));
    let row_fingerprint = format!(
        "{variant_fingerprint}:pack_experiment:{}",
        variant.plan_name
    );
    let artifact_bytes = if let Some(groups) = variant.forced_lqer_groups.as_deref() {
        pg_quant::export::export_model_with_spec_and_lqer_groups(
            model,
            &variant.quant_spec,
            &row_fingerprint,
            &artifact_path,
            groups,
        )?
    } else {
        pg_quant::export::export_model_with_spec(
            model,
            &variant.quant_spec,
            &row_fingerprint,
            &artifact_path,
        )?
    };
    let lqer_groups = pack_experiment_lqer_groups_from_artifact(&artifact_path)?;
    let mut loaded = GptModel::new(model.config.clone());
    pg_quant::export::load_artifact_with_spec(
        &artifact_path,
        &mut loaded,
        &variant.quant_spec,
        true,
    )?;
    let start = Instant::now();
    let post_reload_loss = model_smoke_loss(&loaded, input_ids, targets)?;
    let elapsed = start.elapsed().as_secs_f64();
    let local_decode_tokens_per_second = if elapsed > 0.0 {
        input_ids.len() as f64 / elapsed
    } else {
        f64::INFINITY
    };
    let _ = std::fs::remove_file(&artifact_path);
    let measured_bpb_proxy = post_reload_loss / std::f64::consts::LN_2;
    let finite = pre_export_loss.is_finite()
        && post_reload_loss.is_finite()
        && measured_bpb_proxy.is_finite()
        && local_decode_tokens_per_second.is_finite();

    Ok(PackMeasuredExperimentRowReport {
        plan_name: variant.plan_name.to_string(),
        artifact_bytes,
        byte_delta_vs_target: 0,
        fits_target_artifact_budget: artifact_bytes <= target_artifact_bytes,
        quant_bits: format!(
            "matrix={} mlp={} embed={} attn_gate={}",
            variant.quant_spec.matrix_bits,
            variant.quant_spec.mlp_bits,
            variant.quant_spec.embed_bits,
            variant.quant_spec.attn_gate_bits
        ),
        lqer_enabled: variant.quant_spec.lqer.enabled,
        lqer_rank: variant.quant_spec.lqer.rank,
        lqer_top_k: variant.quant_spec.lqer.top_k,
        lqer_selection_policy: variant.lqer_selection_policy.to_string(),
        lqer_groups,
        pre_export_loss,
        post_reload_loss,
        measured_bpb_proxy,
        loss_delta_abs: (post_reload_loss - pre_export_loss).abs(),
        local_decode_tokens_per_second,
        finite,
        notes: variant.notes.clone(),
    })
}

fn pack_experiment_lqer_groups_from_artifact(path: &Path) -> PgResult<Vec<String>> {
    let metadata = pg_quant::export::artifact_metadata_json(path)?;
    let value = serde_json::from_str::<serde_json::Value>(&metadata).map_err(|err| {
        PgError::DataFormat(format!(
            "ModelGolf pack experiment artifact metadata is invalid JSON: {err}"
        ))
    })?;
    let mut groups = value
        .get("lqer_groups")
        .and_then(|groups| groups.as_array())
        .map(|groups| {
            groups
                .iter()
                .filter_map(|group| group.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    groups.sort();
    Ok(groups)
}

fn pack_experiment_beats(
    rows: &[PackMeasuredExperimentRowReport],
    target_name: &str,
    baseline_name: &str,
) -> Option<bool> {
    let target = rows.iter().find(|row| row.plan_name == target_name)?;
    let baseline = rows.iter().find(|row| row.plan_name == baseline_name)?;
    Some(
        target.finite
            && baseline.finite
            && target.measured_bpb_proxy <= baseline.measured_bpb_proxy,
    )
}

fn precision_options_for_group(
    group: &ModelGolfTensorGroup,
    spec: &RunSpec,
    calibration: Option<&PackQualityCalibrationModel>,
) -> Vec<PrecisionOptionReport> {
    let mut options = Vec::new();
    let compression_factor = compression_factor_for_spec(spec);
    for bits in 4..=8 {
        let weight_bytes = group.elems.saturating_mul(bits as usize).div_ceil(8);
        let scale_bytes = group.rows.saturating_mul(2);
        let raw_bytes = weight_bytes + scale_bytes;
        let base_loss = calibrated_quality_loss(
            quality_loss_surrogate(group, bits, false, spec.quant.lqer.rank),
            group,
            bits,
            false,
            calibration,
        );
        options.push(PrecisionOptionReport {
            group: group.name.clone(),
            role: group.role,
            bits,
            bytes: compressed_byte_estimate(raw_bytes, compression_factor),
            weight_bytes,
            scale_bytes,
            lqer_bytes: 0,
            estimated_quality_loss: base_loss,
            runtime_kernel: format!("{} + per_row_f16_scale_dequant", pack_kernel_for_bits(bits)),
            residual_correction: None,
        });
        if spec.quant.lqer.enabled
            && effective_lqer_rank(group.rows, group.cols, spec.quant.lqer.rank) > 0
            && bits <= 7
        {
            let lqer_bytes = estimate_lqer_bytes(group.rows, group.cols, spec);
            let raw_bytes = weight_bytes + scale_bytes + lqer_bytes;
            let loss = calibrated_quality_loss(
                quality_loss_surrogate(group, bits, true, spec.quant.lqer.rank),
                group,
                bits,
                true,
                calibration,
            );
            options.push(PrecisionOptionReport {
                group: group.name.clone(),
                role: group.role,
                bits,
                bytes: compressed_byte_estimate(raw_bytes, compression_factor),
                weight_bytes,
                scale_bytes,
                lqer_bytes,
                estimated_quality_loss: loss,
                runtime_kernel: format!(
                    "{} + per_row_f16_scale_dequant_lqer_epilogue",
                    pack_kernel_for_bits(bits)
                ),
                residual_correction: Some(format!(
                    "rank{} a{}b{} selective_lqer",
                    spec.quant.lqer.rank, spec.quant.lqer.a_bits, spec.quant.lqer.b_bits
                )),
            });
        }
    }
    options
}

fn lqer_candidates(
    groups: &[ModelGolfTensorGroup],
    spec: &RunSpec,
    selected_options: &[PrecisionOptionReport],
    calibration: Option<&PackQualityCalibrationModel>,
) -> Vec<LqerCandidateReport> {
    if !spec.quant.lqer.enabled || spec.quant.lqer.rank == 0 || spec.quant.lqer.top_k == 0 {
        return Vec::new();
    }
    let mut candidates = groups
        .iter()
        .filter(|group| group.rows > 0 && group.cols > 0)
        .map(|group| {
            let effective_rank = effective_lqer_rank(group.rows, group.cols, spec.quant.lqer.rank);
            let bytes = compressed_byte_estimate(
                estimate_lqer_bytes(group.rows, group.cols, spec),
                compression_factor_for_spec(spec),
            );
            let base = calibrated_quality_loss(
                quality_loss_surrogate(group, group.current_bits, false, spec.quant.lqer.rank),
                group,
                group.current_bits,
                false,
                calibration,
            );
            let corrected = calibrated_quality_loss(
                quality_loss_surrogate(group, group.current_bits, true, spec.quant.lqer.rank),
                group,
                group.current_bits,
                true,
                calibration,
            );
            let reduction = (base - corrected).max(0.0);
            let selected = selected_options
                .iter()
                .any(|option| option.group == group.name && option.lqer_bytes > 0);
            LqerCandidateReport {
                group: group.name.clone(),
                role: group.role,
                rank: spec.quant.lqer.rank,
                effective_rank,
                bytes_added: bytes,
                predicted_loss_reduction: reduction,
                quality_per_byte: if bytes > 0 {
                    reduction / bytes as f64
                } else {
                    0.0
                },
                selected_in_pack_plan: selected,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.quality_per_byte
            .total_cmp(&a.quality_per_byte)
            .then_with(|| a.group.cmp(&b.group))
    });
    candidates.truncate(8);
    candidates
}

fn plan_cache(
    spec: &RunSpec,
    context_tokens: usize,
    batch_sequences: usize,
    memory_budget_bytes: Option<usize>,
) -> PgResult<CachePlannerReport> {
    let cfg = spec.model.to_model_config();
    let head_dim = cfg.head_dim.max(1);
    let kv_elems = batch_sequences
        .saturating_mul(context_tokens)
        .saturating_mul(cfg.num_layers)
        .saturating_mul(cfg.num_kv_heads)
        .saturating_mul(head_dim);
    let fp16_cache_bytes = kv_elems.saturating_mul(2).saturating_mul(2);
    let target_cache_bytes = memory_budget_bytes
        .map(|bytes| (bytes / 2).min(fp16_cache_bytes))
        .unwrap_or_else(|| fp16_cache_bytes.saturating_mul(35).div_ceil(100));
    let mut candidates = Vec::new();
    for block_size_tokens in [64usize, 128, 256] {
        for layout in ["paged", "contiguous"] {
            for k_bits in 2..=8 {
                for v_bits in 2..=8 {
                    let estimated_cache_bytes = cache_policy_estimated_bytes_for_dims(
                        cfg.num_layers,
                        cfg.num_kv_heads,
                        head_dim,
                        context_tokens,
                        batch_sequences,
                        block_size_tokens,
                        layout,
                        k_bits,
                        v_bits,
                    );
                    let bound = attention_error_bound(head_dim, k_bits, v_bits);
                    let compression = fp16_cache_bytes as f64 / estimated_cache_bytes.max(1) as f64;
                    let layout_factor = if layout == "paged" { 0.92 } else { 1.0 };
                    candidates.push(CachePolicyReport {
                        k_bits,
                        v_bits,
                        k_quantization: "per_channel",
                        v_quantization: "per_token",
                        block_size_tokens,
                        layout,
                        estimated_cache_bytes,
                        estimated_attention_error_bound: bound,
                        predicted_attention_speedup: (compression * layout_factor).clamp(0.2, 8.0),
                        budget_ok: estimated_cache_bytes <= target_cache_bytes,
                    });
                }
            }
        }
    }
    candidates.sort_by(|a, b| match (a.budget_ok, b.budget_ok) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a
            .estimated_attention_error_bound
            .total_cmp(&b.estimated_attention_error_bound)
            .then_with(|| a.estimated_cache_bytes.cmp(&b.estimated_cache_bytes)),
    });
    let selected = candidates.first().cloned().unwrap_or(CachePolicyReport {
        k_bits: 8,
        v_bits: 8,
        k_quantization: "per_channel",
        v_quantization: "per_token",
        block_size_tokens: 128,
        layout: "contiguous",
        estimated_cache_bytes: fp16_cache_bytes,
        estimated_attention_error_bound: 0.0,
        predicted_attention_speedup: 1.0,
        budget_ok: fp16_cache_bytes <= target_cache_bytes,
    });
    let feasible = selected.budget_ok;
    candidates.truncate(12);
    let residual_sketch_policy = plan_cache_residual_sketch(
        &cfg,
        context_tokens,
        batch_sequences,
        target_cache_bytes,
        &selected,
    );
    let eviction_policy = plan_cache_eviction(
        &cfg,
        context_tokens,
        batch_sequences,
        target_cache_bytes,
        &selected,
        &residual_sketch_policy,
    );
    let long_context_eval = plan_cache_long_context_eval(
        &cfg,
        context_tokens,
        batch_sequences,
        target_cache_bytes,
        &selected,
        &residual_sketch_policy,
    );
    let selected_policy_proof = prove_cache_policy(spec, &selected, context_tokens)?;
    Ok(CachePlannerReport {
        kind: "cachegolf_kv_plan",
        algorithm: "enumerate_kv_bits_layout_blocks_with_attention_error_bound_plus_residual_sketch_and_eviction_plan",
        context_tokens,
        batch_sequences,
        fp16_cache_bytes,
        target_cache_bytes,
        feasible,
        selected,
        candidates,
        residual_sketch_policy,
        eviction_policy,
        long_context_eval,
        selected_policy_proof,
        bound: "||o_tilde-o||_2 <= epsilon_v + 2*||q||_2*epsilon_k/sqrt(d)*max_j||v_j||_2",
        notes: vec![
            "K uses per-channel quantization and V uses per-token quantization to match asymmetric KV-cache error structure.".to_string(),
            "selected_policy_proof quantizes deterministic K/V through pg-kernels, checks packed dequant-attention parity, and verifies the attention perturbation bound covers observed error.".to_string(),
            "residual_sketch_policy and eviction_policy are deterministic planner outputs for long-context memory pressure; they are not executable fused backend evidence.".to_string(),
            "Fused backend kernels still need backend-specific parity and timing.".to_string(),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn cache_policy_estimated_bytes_for_dims(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_tokens: usize,
    batch_sequences: usize,
    block_size_tokens: usize,
    layout: &str,
    k_bits: u8,
    v_bits: u8,
) -> usize {
    let stored_context_tokens =
        cache_stored_tokens_for_layout(context_tokens, block_size_tokens, layout);
    let kv_elems = batch_sequences
        .saturating_mul(stored_context_tokens)
        .saturating_mul(num_layers)
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim);
    let packed = kv_elems
        .saturating_mul(k_bits as usize + v_bits as usize)
        .div_ceil(8);
    let k_scale_bytes = num_layers
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2);
    let v_scale_bytes = batch_sequences
        .saturating_mul(stored_context_tokens)
        .saturating_mul(num_layers)
        .saturating_mul(num_kv_heads)
        .saturating_mul(2);
    let layout_overhead = if layout == "paged" {
        batch_sequences
            .saturating_mul(context_tokens.div_ceil(block_size_tokens.max(1)))
            .saturating_mul(num_layers)
            .saturating_mul(16)
    } else {
        0
    };
    packed + k_scale_bytes + v_scale_bytes + layout_overhead
}

fn cache_stored_tokens_for_layout(
    context_tokens: usize,
    block_size_tokens: usize,
    layout: &str,
) -> usize {
    if layout == "paged" {
        context_tokens
            .div_ceil(block_size_tokens.max(1))
            .saturating_mul(block_size_tokens.max(1))
    } else {
        context_tokens
    }
}

fn cache_policy_estimated_bytes_for_context(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    policy: &CachePolicyReport,
) -> usize {
    cache_policy_estimated_bytes_for_dims(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim.max(1),
        context_tokens,
        batch_sequences,
        policy.block_size_tokens,
        policy.layout,
        policy.k_bits,
        policy.v_bits,
    )
}

fn plan_cache_residual_sketch(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    target_cache_bytes: usize,
    selected: &CachePolicyReport,
) -> CacheResidualSketchPolicyReport {
    let trigger_attention_error_bound = 1.0;
    let enabled = selected.estimated_attention_error_bound > trigger_attention_error_bound
        || selected.k_bits <= 3
        || selected.v_bits <= 3;
    let protected_recent_tokens = selected
        .block_size_tokens
        .saturating_mul(8)
        .min(context_tokens)
        .max(selected.block_size_tokens.min(context_tokens));
    let sketch_rank = if enabled {
        if selected.k_bits <= 3 || selected.v_bits <= 3 {
            4
        } else {
            2
        }
    } else {
        0
    };
    let sketch_bits = if enabled { 4 } else { 0 };
    let estimated_residual_bytes = if enabled {
        cache_residual_sketch_bytes(
            cfg,
            protected_recent_tokens,
            batch_sequences,
            sketch_rank,
            sketch_bits,
        )
    } else {
        0
    };
    let protected_fraction = if context_tokens > 0 {
        protected_recent_tokens as f64 / context_tokens as f64
    } else {
        0.0
    };
    let estimated_bound_reduction_pct = if enabled {
        (12.0 + sketch_rank as f64 * 7.5 + protected_fraction * 45.0).min(70.0)
    } else {
        0.0
    };
    let selected_reason = if !enabled {
        "selected policy bound/bits do not trigger residual sketching".to_string()
    } else if selected
        .estimated_cache_bytes
        .saturating_add(estimated_residual_bytes)
        <= target_cache_bytes
    {
        "enabled because the selected low-bit cache has sketch budget slack".to_string()
    } else {
        "enabled as a planner recommendation, but eviction or a larger memory budget is needed to keep the sketch under target".to_string()
    };
    CacheResidualSketchPolicyReport {
        kind: "cachegolf_residual_sketch_policy",
        enabled,
        trigger_attention_error_bound,
        protected_recent_tokens,
        sketch_rank,
        sketch_bits,
        estimated_residual_bytes,
        estimated_bound_reduction_pct,
        selected_reason,
        evidence_boundary: "planner byte/error-bound estimate for residual sketches; not an executable residual-cache kernel or long-context quality eval",
    }
}

fn cache_residual_sketch_bytes(
    cfg: &pg_model::ModelConfig,
    protected_recent_tokens: usize,
    batch_sequences: usize,
    sketch_rank: usize,
    sketch_bits: u8,
) -> usize {
    if sketch_rank == 0 || sketch_bits == 0 || protected_recent_tokens == 0 {
        return 0;
    }
    let basis_bytes = cfg
        .num_layers
        .saturating_mul(cfg.num_kv_heads)
        .saturating_mul(cfg.head_dim.max(1))
        .saturating_mul(sketch_rank)
        .saturating_mul(2);
    let coefficient_bytes = batch_sequences
        .saturating_mul(protected_recent_tokens)
        .saturating_mul(cfg.num_layers)
        .saturating_mul(cfg.num_kv_heads)
        .saturating_mul(sketch_rank)
        .saturating_mul(sketch_bits as usize)
        .div_ceil(8);
    let scale_bytes = batch_sequences
        .saturating_mul(protected_recent_tokens)
        .saturating_mul(cfg.num_layers)
        .saturating_mul(cfg.num_kv_heads)
        .saturating_mul(2);
    basis_bytes + coefficient_bytes + scale_bytes
}

fn plan_cache_eviction(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    target_cache_bytes: usize,
    selected: &CachePolicyReport,
    residual: &CacheResidualSketchPolicyReport,
) -> CacheEvictionPolicyReport {
    let total_selected_bytes = selected
        .estimated_cache_bytes
        .saturating_add(residual.estimated_residual_bytes);
    let eviction_needed_for_budget = total_selected_bytes > target_cache_bytes;
    let sink_tokens = selected.block_size_tokens.min(context_tokens);
    let protected_recent_tokens = residual.protected_recent_tokens.min(context_tokens);
    let protected_total = sink_tokens.saturating_add(protected_recent_tokens);
    let evictable_tokens = context_tokens.saturating_sub(protected_total);
    let excess_bytes = total_selected_bytes.saturating_sub(target_cache_bytes);
    let bytes_per_token = cache_policy_estimated_bytes_for_context(
        cfg,
        context_tokens.max(1),
        batch_sequences,
        selected,
    ) as f64
        / context_tokens.max(1) as f64;
    let max_evictable_bytes = (evictable_tokens as f64 * bytes_per_token).floor() as usize;
    let estimated_evicted_bytes_at_budget = if eviction_needed_for_budget {
        excess_bytes.min(max_evictable_bytes)
    } else {
        0
    };
    CacheEvictionPolicyReport {
        kind: "cachegolf_eviction_policy",
        policy: "sink_plus_recent_protected_bound_aware_eviction",
        protected_recent_tokens,
        sink_tokens,
        evictable_tokens,
        estimated_evicted_bytes_at_budget,
        eviction_needed_for_budget,
        score_components: vec![
            "protect sink tokens for global attention anchors".to_string(),
            "protect the most recent residual-sketch window".to_string(),
            "evict oldest low-attention blocks first when target bytes are exceeded".to_string(),
            format!(
                "selected_cache_plus_residual_bytes={total_selected_bytes}; target_cache_bytes={target_cache_bytes}"
            ),
        ],
        evidence_boundary: "planner eviction policy only; not a measured streaming-cache runtime or task-quality retention proof",
    }
}

fn plan_cache_long_context_eval(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    target_cache_bytes: usize,
    selected: &CachePolicyReport,
    residual: &CacheResidualSketchPolicyReport,
) -> CacheLongContextEvalReport {
    let mut contexts = vec![
        context_tokens.saturating_div(4).max(1),
        context_tokens.saturating_div(2).max(1),
        context_tokens.max(1),
        context_tokens.saturating_mul(2).max(1),
    ];
    contexts.sort_unstable();
    contexts.dedup();
    let rows = contexts
        .into_iter()
        .map(|context| {
            let fp16_cache_bytes = cache_fp16_bytes_for_context(cfg, context, batch_sequences);
            let selected_policy_cache_bytes =
                cache_policy_estimated_bytes_for_context(cfg, context, batch_sequences, selected);
            let residual_sketch_bytes = if residual.enabled {
                let protected = residual.protected_recent_tokens.min(context);
                cache_residual_sketch_bytes(
                    cfg,
                    protected,
                    batch_sequences,
                    residual.sketch_rank,
                    residual.sketch_bits,
                )
            } else {
                0
            };
            let total_cache_bytes = selected_policy_cache_bytes + residual_sketch_bytes;
            let reduction = if fp16_cache_bytes > 0 {
                (1.0 - total_cache_bytes as f64 / fp16_cache_bytes as f64) * 100.0
            } else {
                0.0
            };
            let adjusted_bound = selected.estimated_attention_error_bound
                * (1.0 - residual.estimated_bound_reduction_pct / 100.0);
            CacheLongContextEvalRowReport {
                context_tokens: context,
                fp16_cache_bytes,
                selected_policy_cache_bytes,
                residual_sketch_bytes,
                total_cache_bytes,
                memory_reduction_vs_fp16_pct: reduction,
                fits_target_cache_budget: total_cache_bytes <= target_cache_bytes,
                estimated_attention_error_bound: adjusted_bound,
                predicted_attention_speedup: selected.predicted_attention_speedup,
            }
        })
        .collect::<Vec<_>>();
    let selected_context_fits_budget = rows
        .iter()
        .find(|row| row.context_tokens == context_tokens.max(1))
        .map(|row| row.fits_target_cache_budget)
        .unwrap_or(false);
    let max_context_tokens_under_budget = max_cache_context_under_budget(
        cfg,
        batch_sequences,
        target_cache_bytes,
        selected,
        residual,
    );
    CacheLongContextEvalReport {
        kind: "cachegolf_long_context_memory_eval",
        rows,
        selected_context_fits_budget,
        max_context_tokens_under_budget,
        evidence_boundary: "deterministic byte/error-bound scaling model; not long-context BPB/perplexity or fused runtime timing",
    }
}

fn cache_fp16_bytes_for_context(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
) -> usize {
    batch_sequences
        .saturating_mul(context_tokens)
        .saturating_mul(cfg.num_layers)
        .saturating_mul(cfg.num_kv_heads)
        .saturating_mul(cfg.head_dim.max(1))
        .saturating_mul(2)
        .saturating_mul(2)
}

fn max_cache_context_under_budget(
    cfg: &pg_model::ModelConfig,
    batch_sequences: usize,
    target_cache_bytes: usize,
    selected: &CachePolicyReport,
    residual: &CacheResidualSketchPolicyReport,
) -> usize {
    let mut lo = 0usize;
    let mut hi = 1usize;
    while cache_total_bytes_for_context(cfg, hi, batch_sequences, selected, residual)
        <= target_cache_bytes
        && hi < (1usize << 30)
    {
        lo = hi;
        hi = hi.saturating_mul(2);
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if cache_total_bytes_for_context(cfg, mid, batch_sequences, selected, residual)
            <= target_cache_bytes
        {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn cache_total_bytes_for_context(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    selected: &CachePolicyReport,
    residual: &CacheResidualSketchPolicyReport,
) -> usize {
    let selected_bytes =
        cache_policy_estimated_bytes_for_context(cfg, context_tokens, batch_sequences, selected);
    let residual_bytes = if residual.enabled {
        cache_residual_sketch_bytes(
            cfg,
            residual.protected_recent_tokens.min(context_tokens),
            batch_sequences,
            residual.sketch_rank,
            residual.sketch_bits,
        )
    } else {
        0
    };
    selected_bytes.saturating_add(residual_bytes)
}

fn prove_cache_policy(
    spec: &RunSpec,
    selected: &CachePolicyReport,
    context_tokens: usize,
) -> PgResult<CachePolicyProofReport> {
    use pg_kernels::cachegolf::{
        cachegolf_attention_error_bound, cachegolf_causal_attention_forward,
        cachegolf_kv_error_stats, cachegolf_quantize_kv,
    };

    let cfg = spec.model.to_model_config();
    let proof_tokens = context_tokens.clamp(2, 64);
    let num_heads = cfg.num_heads.max(1);
    let num_kv_heads = cfg.num_kv_heads.max(1);
    let head_dim = cfg.head_dim.max(1);
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(PgError::InvalidOp(format!(
            "CacheGolf proof requires num_heads ({num_heads}) to be a multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    let q = deterministic_modelgolf_values(proof_tokens * num_heads * head_dim, 0.19);
    let k = deterministic_modelgolf_values(proof_tokens * num_kv_heads * head_dim, 0.29);
    let v = deterministic_modelgolf_values(proof_tokens * num_kv_heads * head_dim, 0.43);
    let layout = cache_layout_from_label(selected.layout)?;
    let cache = cachegolf_quantize_kv(
        &k,
        &v,
        proof_tokens,
        num_kv_heads,
        head_dim,
        selected.k_bits,
        selected.v_bits,
        selected.block_size_tokens,
        layout,
    )?;

    let mut full = vec![0.0f32; q.len()];
    pg_kernels::attention::causal_attention_forward(
        &q,
        &k,
        &v,
        &mut full,
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    let mut quantized = vec![0.0f32; q.len()];
    cachegolf_causal_attention_forward(&q, &cache, &mut quantized, num_heads)?;

    let mut deq_k = vec![0.0f32; k.len()];
    let mut deq_v = vec![0.0f32; v.len()];
    cache.dequantize_kv(&mut deq_k, &mut deq_v)?;
    let mut explicit_dequant = vec![0.0f32; q.len()];
    pg_kernels::attention::causal_attention_forward(
        &q,
        &deq_k,
        &deq_v,
        &mut explicit_dequant,
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    let stats = cachegolf_kv_error_stats(&cache, &k, &v)?;
    let mut max_query_l2_norm = 0.0f64;
    let mut max_attention_error_bound = 0.0f64;
    let mut max_observed_attention_l2_error = 0.0f64;
    let mut max_dequant_attention_parity_l2_error = 0.0f64;
    for token in 0..proof_tokens {
        for head in 0..num_heads {
            let offset = (token * num_heads + head) * head_dim;
            let q_slice = &q[offset..offset + head_dim];
            let full_slice = &full[offset..offset + head_dim];
            let quantized_slice = &quantized[offset..offset + head_dim];
            let explicit_dequant_slice = &explicit_dequant[offset..offset + head_dim];
            let q_norm = l2_norm(q_slice);
            let bound = cachegolf_attention_error_bound(q_norm as f32, stats, head_dim) as f64;
            let observed = l2_distance(full_slice, quantized_slice);
            let parity = l2_distance(explicit_dequant_slice, quantized_slice);
            max_query_l2_norm = max_query_l2_norm.max(q_norm);
            max_attention_error_bound = max_attention_error_bound.max(bound);
            max_observed_attention_l2_error = max_observed_attention_l2_error.max(observed);
            max_dequant_attention_parity_l2_error =
                max_dequant_attention_parity_l2_error.max(parity);
        }
    }

    let packed_cache_bytes_single_layer = cache.f16_scale_runtime_bytes();
    let fp16_cache_bytes_single_layer = proof_tokens
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(2);
    Ok(CachePolicyProofReport {
        kind: "cachegolf_selected_policy_local_proof",
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        layout: selected.layout,
        k_bits: selected.k_bits,
        v_bits: selected.v_bits,
        block_size_tokens: selected.block_size_tokens,
        packed_cache_bytes_single_layer,
        fp16_cache_bytes_single_layer,
        compression_ratio_single_layer: fp16_cache_bytes_single_layer as f64
            / packed_cache_bytes_single_layer.max(1) as f64,
        max_key_l2_error: stats.max_key_l2_error as f64,
        max_value_l2_error: stats.max_value_l2_error as f64,
        max_value_l2_norm: stats.max_value_l2_norm as f64,
        max_query_l2_norm,
        max_attention_error_bound,
        max_observed_attention_l2_error,
        max_dequant_attention_parity_l2_error,
        bound_covers_observed_error: max_observed_attention_l2_error
            <= max_attention_error_bound + 1e-5,
        dequant_attention_parity_ok: max_dequant_attention_parity_l2_error <= 1e-5,
        evidence_boundary: "local deterministic selected-policy proof; not fused CUDA/Metal timing or long-context validation BPB evidence",
    })
}

fn run_cachegolf_kv_experiment(
    spec: &RunSpec,
    context_tokens: usize,
    batch_sequences: usize,
    target_cache_budget_bytes: usize,
) -> PgResult<CacheGolfKvExperimentReport> {
    let cfg = spec.model.to_model_config();
    let proof_tokens = context_tokens.clamp(4, 64);
    let num_heads = cfg.num_heads.max(1);
    let num_kv_heads = cfg.num_kv_heads.max(1);
    let head_dim = cfg.head_dim.max(1);
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(PgError::InvalidOp(format!(
            "CacheGolf experiment requires num_heads ({num_heads}) to be a multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    let block_size_tokens = 64usize.min(proof_tokens).max(1);
    let layout = "paged";
    let q = deterministic_modelgolf_values(proof_tokens * num_heads * head_dim, 0.191);
    let k = deterministic_modelgolf_values(proof_tokens * num_kv_heads * head_dim, 0.293);
    let v = deterministic_modelgolf_values(proof_tokens * num_kv_heads * head_dim, 0.431);
    let mut full = vec![0.0f32; q.len()];
    pg_kernels::attention::causal_attention_forward(
        &q,
        &k,
        &v,
        &mut full,
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    let mut rows = Vec::new();
    for k_bits in [2u8, 3, 4, 6, 8] {
        for v_bits in [2u8, 3, 4, 6, 8] {
            rows.push(cachegolf_kv_experiment_row(
                &q,
                &k,
                &v,
                &full,
                proof_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size_tokens,
                layout,
                k_bits,
                v_bits,
            )?);
        }
    }
    rows.sort_by(|a, b| {
        a.k_bits
            .cmp(&b.k_bits)
            .then_with(|| a.v_bits.cmp(&b.v_bits))
    });

    let best_observed_error_plan = rows
        .iter()
        .filter(|row| row.bound_covers_observed_error && row.dequant_attention_parity_ok)
        .min_by(|a, b| {
            a.max_observed_attention_l2_error
                .total_cmp(&b.max_observed_attention_l2_error)
                .then_with(|| {
                    a.packed_cache_bytes_single_layer
                        .cmp(&b.packed_cache_bytes_single_layer)
                })
        })
        .map(|row| row.plan_name.clone());
    let best_bound_tightness_plan = rows
        .iter()
        .filter(|row| row.bound_covers_observed_error && row.bound_to_observed_ratio.is_finite())
        .min_by(|a, b| {
            a.bound_to_observed_ratio
                .total_cmp(&b.bound_to_observed_ratio)
                .then_with(|| {
                    a.packed_cache_bytes_single_layer
                        .cmp(&b.packed_cache_bytes_single_layer)
                })
        })
        .map(|row| row.plan_name.clone());
    let requested_fp16_cache_bytes =
        cachegolf_experiment_fp16_cache_bytes(&cfg, context_tokens, batch_sequences);
    let target_cache_budget_pct_of_fp16 = if requested_fp16_cache_bytes > 0 {
        100.0 * target_cache_budget_bytes as f64 / requested_fp16_cache_bytes as f64
    } else {
        0.0
    };
    let best_budgeted_plan = rows
        .iter()
        .filter(|row| row.bound_covers_observed_error && row.dequant_attention_parity_ok)
        .filter(|row| {
            cachegolf_experiment_policy_cache_bytes(
                &cfg,
                context_tokens,
                batch_sequences,
                block_size_tokens,
                layout,
                row,
            ) <= target_cache_budget_bytes
        })
        .min_by(|a, b| {
            a.max_observed_attention_l2_error
                .total_cmp(&b.max_observed_attention_l2_error)
                .then_with(|| {
                    a.packed_cache_bytes_single_layer
                        .cmp(&b.packed_cache_bytes_single_layer)
                })
        })
        .map(|row| row.plan_name.clone());
    let all_bounds_cover_observed_error = rows.iter().all(|row| {
        row.bound_covers_observed_error
            && row.dequant_attention_parity_ok
            && row.max_attention_error_bound.is_finite()
            && row.max_observed_attention_l2_error.is_finite()
    });
    let selected_grid_plan_name = best_budgeted_plan
        .clone()
        .or_else(|| best_observed_error_plan.clone())
        .unwrap_or_else(|| "k8_v8".to_string());
    let selected_grid_policy = rows
        .iter()
        .find(|row| row.plan_name == selected_grid_plan_name)
        .or_else(|| rows.iter().find(|row| row.k_bits == 8 && row.v_bits == 8))
        .expect("CacheGolf experiment row set is non-empty");
    let long_context_memory = cachegolf_experiment_long_context_memory(
        &cfg,
        context_tokens,
        batch_sequences,
        block_size_tokens,
        layout,
        target_cache_budget_bytes,
        selected_grid_policy,
    );

    Ok(CacheGolfKvExperimentReport {
        kind: "cachegolf_kv_bit_grid_experiment",
        protocol: "quantize_kv_bit_grid_measure_attention_error_compare_perturbation_bound_and_contract_budget_memory",
        source_spec_name: spec.name.clone(),
        target_cache_budget_bytes,
        target_cache_budget_pct_of_fp16,
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size_tokens,
        layout,
        rows,
        long_context_memory,
        best_observed_error_plan,
        best_bound_tightness_plan,
        best_budgeted_plan,
        all_bounds_cover_observed_error,
        status: if all_bounds_cover_observed_error {
            "local_experiment_ready".to_string()
        } else {
            "local_experiment_failed".to_string()
        },
        evidence_boundary: "bounded deterministic CPU Experiment 3 harness; proves local K/V quantization error-bound coverage and memory scaling, but not fused runtime timing or long-context BPB/perplexity",
    })
}

#[allow(clippy::too_many_arguments)]
fn cachegolf_kv_experiment_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    full_attention: &[f32],
    proof_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size_tokens: usize,
    layout_label: &'static str,
    k_bits: u8,
    v_bits: u8,
) -> PgResult<CacheGolfKvExperimentRowReport> {
    use pg_kernels::cachegolf::{
        cachegolf_attention_error_bound, cachegolf_causal_attention_forward,
        cachegolf_kv_error_stats, cachegolf_quantize_kv,
    };

    let layout = cache_layout_from_label(layout_label)?;
    let cache = cachegolf_quantize_kv(
        k,
        v,
        proof_tokens,
        num_kv_heads,
        head_dim,
        k_bits,
        v_bits,
        block_size_tokens,
        layout,
    )?;
    let mut quantized = vec![0.0f32; full_attention.len()];
    cachegolf_causal_attention_forward(q, &cache, &mut quantized, num_heads)?;

    let mut deq_k = vec![0.0f32; k.len()];
    let mut deq_v = vec![0.0f32; v.len()];
    cache.dequantize_kv(&mut deq_k, &mut deq_v)?;
    let mut explicit_dequant = vec![0.0f32; full_attention.len()];
    pg_kernels::attention::causal_attention_forward(
        q,
        &deq_k,
        &deq_v,
        &mut explicit_dequant,
        proof_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    let stats = cachegolf_kv_error_stats(&cache, k, v)?;
    let mut max_query_l2_norm = 0.0f64;
    let mut max_attention_error_bound = 0.0f64;
    let mut max_observed_attention_l2_error = 0.0f64;
    let mut max_dequant_attention_parity_l2_error = 0.0f64;
    for token in 0..proof_tokens {
        for head in 0..num_heads {
            let offset = (token * num_heads + head) * head_dim;
            let q_slice = &q[offset..offset + head_dim];
            let full_slice = &full_attention[offset..offset + head_dim];
            let quantized_slice = &quantized[offset..offset + head_dim];
            let explicit_dequant_slice = &explicit_dequant[offset..offset + head_dim];
            let q_norm = l2_norm(q_slice);
            let bound = cachegolf_attention_error_bound(q_norm as f32, stats, head_dim) as f64;
            let observed = l2_distance(full_slice, quantized_slice);
            let parity = l2_distance(explicit_dequant_slice, quantized_slice);
            max_query_l2_norm = max_query_l2_norm.max(q_norm);
            max_attention_error_bound = max_attention_error_bound.max(bound);
            max_observed_attention_l2_error = max_observed_attention_l2_error.max(observed);
            max_dequant_attention_parity_l2_error =
                max_dequant_attention_parity_l2_error.max(parity);
        }
    }
    let packed_cache_bytes_single_layer = cache.f16_scale_runtime_bytes();
    let fp16_cache_bytes_single_layer = proof_tokens
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(2);
    let bound_to_observed_ratio = if max_observed_attention_l2_error > 1e-12 {
        max_attention_error_bound / max_observed_attention_l2_error
    } else {
        f64::INFINITY
    };

    Ok(CacheGolfKvExperimentRowReport {
        plan_name: format!("k{k_bits}_v{v_bits}"),
        k_bits,
        v_bits,
        packed_cache_bytes_single_layer,
        fp16_cache_bytes_single_layer,
        compression_ratio_single_layer: fp16_cache_bytes_single_layer as f64
            / packed_cache_bytes_single_layer.max(1) as f64,
        max_key_l2_error: stats.max_key_l2_error as f64,
        max_value_l2_error: stats.max_value_l2_error as f64,
        max_value_l2_norm: stats.max_value_l2_norm as f64,
        max_query_l2_norm,
        max_attention_error_bound,
        max_observed_attention_l2_error,
        bound_to_observed_ratio,
        max_dequant_attention_parity_l2_error,
        bound_covers_observed_error: max_observed_attention_l2_error
            <= max_attention_error_bound + 1e-5,
        dequant_attention_parity_ok: max_dequant_attention_parity_l2_error <= 1e-5,
        notes: vec![
            "K is quantized per-channel and V per-token using pg-kernels::cachegolf.".to_string(),
            "Bound is epsilon_v + 2*||q||_2*epsilon_k/sqrt(d)*max||v||_2 evaluated from measured quantization errors.".to_string(),
        ],
    })
}

fn cachegolf_experiment_fp16_cache_bytes(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
) -> usize {
    context_tokens
        .max(1)
        .saturating_mul(batch_sequences.max(1))
        .saturating_mul(cfg.num_layers.max(1))
        .saturating_mul(cfg.num_kv_heads.max(1))
        .saturating_mul(cfg.head_dim.max(1))
        .saturating_mul(2)
        .saturating_mul(2)
}

fn cachegolf_experiment_policy_cache_bytes(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    block_size_tokens: usize,
    layout: &'static str,
    selected_row: &CacheGolfKvExperimentRowReport,
) -> usize {
    cache_policy_estimated_bytes_for_dims(
        cfg.num_layers.max(1),
        cfg.num_kv_heads.max(1),
        cfg.head_dim.max(1),
        context_tokens.max(1),
        batch_sequences.max(1),
        block_size_tokens,
        layout,
        selected_row.k_bits,
        selected_row.v_bits,
    )
}

fn cachegolf_experiment_long_context_memory(
    cfg: &pg_model::ModelConfig,
    context_tokens: usize,
    batch_sequences: usize,
    block_size_tokens: usize,
    layout: &'static str,
    target_cache_budget_bytes: usize,
    selected_row: &CacheGolfKvExperimentRowReport,
) -> CacheGolfKvExperimentLongContextReport {
    let mut contexts = vec![
        context_tokens.max(1),
        context_tokens.saturating_mul(2).max(1),
        context_tokens.saturating_mul(4).max(1),
    ];
    contexts.sort_unstable();
    contexts.dedup();
    let rows = contexts
        .into_iter()
        .map(|context| {
            let fp16_cache_bytes =
                cachegolf_experiment_fp16_cache_bytes(cfg, context, batch_sequences);
            let selected_grid_cache_bytes = cachegolf_experiment_policy_cache_bytes(
                cfg,
                context,
                batch_sequences,
                block_size_tokens,
                layout,
                selected_row,
            );
            CacheGolfKvExperimentContextRowReport {
                context_tokens: context,
                batch_sequences: batch_sequences.max(1),
                fp16_cache_bytes,
                target_cache_budget_bytes,
                selected_grid_cache_bytes,
                memory_reduction_vs_fp16_pct: 100.0
                    * (1.0 - selected_grid_cache_bytes as f64 / fp16_cache_bytes.max(1) as f64),
                fits_target_cache_budget: selected_grid_cache_bytes <= target_cache_budget_bytes,
                selected_grid_plan_name: selected_row.plan_name.clone(),
            }
        })
        .collect();
    CacheGolfKvExperimentLongContextReport {
        kind: "cachegolf_kv_bit_grid_long_context_memory",
        rows,
        evidence_boundary: "deterministic memory extrapolation for the selected measured bit-grid policy under the resource contract cache budget; not long-context BPB/perplexity or fused runtime timing",
    }
}

fn cache_layout_from_label(label: &str) -> PgResult<pg_kernels::cachegolf::CacheGolfLayout> {
    match label {
        "contiguous" => Ok(pg_kernels::cachegolf::CacheGolfLayout::Contiguous),
        "paged" => Ok(pg_kernels::cachegolf::CacheGolfLayout::Paged),
        other => Err(PgError::InvalidOp(format!(
            "unknown CacheGolf layout label: {other}"
        ))),
    }
}

fn plan_delta(spec: &RunSpec, delta_budget_bytes: usize) -> PgResult<DeltaPlannerReport> {
    let candidates = delta_candidate_sets(spec);
    let mut dp = BTreeMap::new();
    dp.insert(
        0usize,
        DeltaDpState {
            gain: 0.0,
            bytes: 0,
            option_indices: Vec::new(),
        },
    );
    for options in &candidates {
        let mut next = BTreeMap::<usize, DeltaDpState>::new();
        for state in dp.values() {
            for (option_index, option) in options.iter().enumerate() {
                let next_bytes = state.bytes.saturating_add(option.bytes);
                if next_bytes > delta_budget_bytes {
                    continue;
                }
                let mut option_indices = state.option_indices.clone();
                option_indices.push(option_index);
                let candidate = DeltaDpState {
                    gain: state.gain + option.predicted_domain_gain,
                    bytes: next_bytes,
                    option_indices,
                };
                if next
                    .get(&next_bytes)
                    .as_ref()
                    .map(|old| better_delta_state(&candidate, old))
                    .unwrap_or(true)
                {
                    next.insert(next_bytes, candidate);
                }
            }
        }
        dp = prune_dominated_delta_states(next);
    }
    let best = dp
        .into_values()
        .filter(|state| state.bytes <= delta_budget_bytes)
        .max_by(|a, b| {
            a.gain
                .total_cmp(&b.gain)
                .then_with(|| b.bytes.cmp(&a.bytes))
        });
    let (selected_deltas, selected_bytes, selected_predicted_gain) = if let Some(best) = best {
        let selected = best
            .option_indices
            .iter()
            .enumerate()
            .filter_map(|(family_index, &option_index)| {
                let option = candidates[family_index][option_index].clone();
                (option.bytes > 0).then_some(option)
            })
            .collect::<Vec<_>>();
        (selected, best.bytes, best.gain)
    } else {
        (Vec::new(), 0, 0.0)
    };
    let mut all_candidates = candidates.into_iter().flatten().collect::<Vec<_>>();
    all_candidates.retain(|candidate| candidate.bytes > 0);
    let mut flat_candidates = all_candidates.clone();
    flat_candidates.sort_by(|a, b| {
        b.gain_per_byte
            .total_cmp(&a.gain_per_byte)
            .then_with(|| a.name.cmp(&b.name))
    });
    flat_candidates.truncate(16);
    let selected_low_rank_proofs = prove_selected_delta_low_rank(spec, &selected_deltas)?;
    let score_first_legality_audit = audit_delta_score_first_legality(spec, &selected_deltas);
    let domain_evaluation =
        evaluate_delta_domain_proxy(spec, &all_candidates, &selected_deltas, selected_bytes)?;
    let mut notes = vec![
        "Delta options represent legal local/private adaptation artifacts, not changes to the frozen base model.".to_string(),
        "pg-kernels provides the weighted low-rank Kronecker-curvature CPU reference; this planner still needs measured full-update and curvature statistics before selecting trained deltas by quality.".to_string(),
        "domain_evaluation is a deterministic equal-byte proxy over synthetic domain deltas; it is not private-corpus BPB/perplexity evidence.".to_string(),
    ];
    if selected_low_rank_proofs.is_empty() {
        notes.push("No selected rank-bearing delta fit the current byte contract; weighted low-rank proof is skipped.".to_string());
    } else {
        notes.push("selected_low_rank_proofs use deterministic synthetic full updates and positive curvature diagonals to verify weighted low-rank math for selected delta families.".to_string());
    }
    Ok(DeltaPlannerReport {
        kind: "deltagolf_plan",
        algorithm: "exact_multiple_choice_byte_constrained_delta_allocation",
        delta_budget_bytes,
        selected_bytes,
        selected_predicted_gain,
        selected_deltas,
        candidates: flat_candidates,
        selected_low_rank_proofs,
        score_first_legality_audit,
        domain_evaluation,
        notes,
    })
}

fn audit_delta_score_first_legality(
    spec: &RunSpec,
    selected_deltas: &[DeltaOptionReport],
) -> DeltaScoreFirstLegalityAuditReport {
    let score_first_required = selected_deltas
        .iter()
        .any(|delta| delta.legality.contains("score_first"));
    let artifact_paid_delta_count = selected_deltas
        .iter()
        .filter(|delta| {
            delta.legality.contains("artifact_paid") || delta.legality.contains("paid_in_artifact")
        })
        .count();
    let score_first_or_train_only_delta_count = selected_deltas
        .iter()
        .filter(|delta| {
            delta.legality.contains("score_first")
                || delta.legality.contains("train_only")
                || delta.legality.contains("offline_train")
        })
        .count();
    let illegal_delta_count = selected_deltas
        .iter()
        .filter(|delta| delta.legality.contains("illegal"))
        .count();
    let selected_delta_count = selected_deltas.len();
    let all_paid_or_static = selected_deltas.iter().all(delta_legality_is_artifact_paid);
    let pass = illegal_delta_count == 0
        && (!score_first_required || spec.eval.legal_score_first || all_paid_or_static);
    let mut notes = Vec::new();
    if selected_delta_count == 0 {
        notes.push("No selected delta artifacts under this byte contract.".to_string());
    }
    if score_first_required && spec.eval.legal_score_first {
        notes.push("Spec declares score-first eval legal; selected score-first-capable deltas may be updated only after scoring each token.".to_string());
    } else if score_first_required && all_paid_or_static {
        notes.push("Selected score-first-capable deltas are treated as offline-trained or static artifact-paid deltas for this planner report.".to_string());
    } else if score_first_required {
        notes.push("Selected deltas mention score-first semantics but the spec does not declare score-first eval legal.".to_string());
    }
    if illegal_delta_count > 0 {
        notes.push(
            "At least one selected delta is marked illegal by the local metadata audit."
                .to_string(),
        );
    }
    if pass {
        notes.push("Local metadata audit found no unguarded in-eval mutation claim.".to_string());
    }
    DeltaScoreFirstLegalityAuditReport {
        kind: "deltagolf_score_first_legality_audit",
        score_first_required,
        spec_declares_score_first_legal: spec.eval.legal_score_first,
        selected_delta_count,
        artifact_paid_delta_count,
        score_first_or_train_only_delta_count,
        illegal_delta_count,
        pass,
        notes,
        evidence_boundary: "metadata legality audit only; not an official competition/legal review or runtime score-first trace",
    }
}

fn evaluate_delta_domain_proxy(
    spec: &RunSpec,
    candidates: &[DeltaOptionReport],
    selected_deltas: &[DeltaOptionReport],
    selected_bytes: usize,
) -> PgResult<DeltaDomainEvaluationReport> {
    let comparison_budget_bytes = selected_bytes;
    let selected_plan_name = "selected_delta_plan";
    let mut rows = Vec::new();

    rows.push(delta_domain_evaluation_row(
        spec,
        "no_delta",
        &[],
        comparison_budget_bytes,
        selected_bytes,
    )?);

    rows.push(delta_domain_evaluation_row(
        spec,
        selected_plan_name,
        selected_deltas,
        comparison_budget_bytes,
        selected_bytes,
    )?);

    if let Some(best_single) = candidates
        .iter()
        .filter(|candidate| candidate.bytes <= comparison_budget_bytes)
        .max_by(|a, b| {
            a.predicted_domain_gain
                .total_cmp(&b.predicted_domain_gain)
                .then_with(|| b.bytes.cmp(&a.bytes))
        })
        .cloned()
    {
        rows.push(delta_domain_evaluation_row(
            spec,
            "best_single_delta_equal_byte",
            &[best_single],
            comparison_budget_bytes,
            selected_bytes,
        )?);
    }

    let low_rank_selected = selected_deltas
        .iter()
        .filter(|delta| delta.rank.is_some())
        .cloned()
        .collect::<Vec<_>>();
    if !low_rank_selected.is_empty() {
        rows.push(delta_domain_evaluation_row(
            spec,
            "selected_low_rank_only",
            &low_rank_selected,
            comparison_budget_bytes,
            selected_bytes,
        )?);
    }

    let static_control = best_static_delta_control(candidates, comparison_budget_bytes);
    if !static_control.is_empty() {
        rows.push(delta_domain_evaluation_row(
            spec,
            "static_lookup_control_equal_byte",
            &static_control,
            comparison_budget_bytes,
            selected_bytes,
        )?);
    }

    rows.sort_by(|a, b| {
        a.plan_name
            .cmp(&b.plan_name)
            .then_with(|| a.artifact_bytes.cmp(&b.artifact_bytes))
    });
    let best_proxy_plan = rows
        .iter()
        .filter(|row| row.fits_comparison_budget && row.score_first_legal)
        .min_by(|a, b| {
            a.estimated_domain_loss_proxy
                .total_cmp(&b.estimated_domain_loss_proxy)
        })
        .map(|row| row.plan_name.clone());
    let selected_beats_no_delta_proxy = delta_domain_beats(&rows, selected_plan_name, "no_delta");
    let selected_beats_best_single_proxy =
        delta_domain_beats(&rows, selected_plan_name, "best_single_delta_equal_byte");
    let selected_beats_static_control_proxy = delta_domain_beats(
        &rows,
        selected_plan_name,
        "static_lookup_control_equal_byte",
    );

    Ok(DeltaDomainEvaluationReport {
        kind: "deltagolf_equal_byte_domain_proxy",
        comparison_budget_bytes,
        selected_plan_name,
        rows,
        best_proxy_plan,
        selected_beats_no_delta_proxy,
        selected_beats_best_single_proxy,
        selected_beats_static_control_proxy,
        evidence_boundary: "deterministic local proxy from predicted gain and synthetic weighted-update errors; not trained private-corpus BPB/perplexity evidence",
    })
}

fn delta_domain_evaluation_row(
    spec: &RunSpec,
    plan_name: &str,
    deltas: &[DeltaOptionReport],
    comparison_budget_bytes: usize,
    selected_bytes: usize,
) -> PgResult<DeltaDomainEvaluationRowReport> {
    let artifact_bytes = deltas.iter().map(|delta| delta.bytes).sum::<usize>();
    let predicted_domain_gain = deltas
        .iter()
        .map(|delta| delta.predicted_domain_gain)
        .sum::<f64>();
    let low_rank_weighted_error_proxy = delta_low_rank_weighted_error_proxy(spec, deltas)?;
    let estimated_domain_loss_proxy =
        delta_domain_loss_proxy(predicted_domain_gain, low_rank_weighted_error_proxy);
    let estimated_bpb_delta_proxy = -0.015 * predicted_domain_gain;
    let (score_first_legal, legality_notes) = delta_row_legality(spec, deltas);
    let mut delta_names = deltas
        .iter()
        .map(|delta| delta.name.clone())
        .collect::<Vec<_>>();
    delta_names.sort();
    Ok(DeltaDomainEvaluationRowReport {
        plan_name: plan_name.to_string(),
        artifact_bytes,
        byte_delta_vs_selected: artifact_bytes as isize - selected_bytes as isize,
        fits_comparison_budget: artifact_bytes <= comparison_budget_bytes,
        delta_count: deltas.len(),
        predicted_domain_gain,
        estimated_domain_loss_proxy,
        estimated_bpb_delta_proxy,
        low_rank_weighted_error_proxy,
        score_first_legal,
        legality_notes,
        delta_names,
    })
}

fn delta_low_rank_weighted_error_proxy(
    spec: &RunSpec,
    deltas: &[DeltaOptionReport],
) -> PgResult<Option<f64>> {
    let mut total = 0.0;
    let mut count = 0usize;
    for delta in deltas.iter().filter(|delta| delta.rank.is_some()) {
        let proof = prove_delta_low_rank(spec, delta)?;
        total += proof.selected_rank_weighted_error / proof.zero_delta_weighted_error.max(1e-12);
        count += 1;
    }
    Ok((count > 0).then_some(total / count as f64))
}

fn delta_domain_loss_proxy(
    predicted_domain_gain: f64,
    low_rank_weighted_error_proxy: Option<f64>,
) -> f64 {
    let gain_loss = 1.0 / (1.0 + predicted_domain_gain.max(0.0));
    let low_rank_penalty = low_rank_weighted_error_proxy
        .map(|error| error.clamp(0.0, 4.0) * 0.05)
        .unwrap_or(0.04);
    (gain_loss + low_rank_penalty).max(0.0)
}

fn best_static_delta_control(
    candidates: &[DeltaOptionReport],
    budget_bytes: usize,
) -> Vec<DeltaOptionReport> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate.bytes <= budget_bytes
                && (candidate.delta_type == "ngram_residual" || candidate.delta_type == "bias")
        })
        .max_by(|a, b| {
            a.predicted_domain_gain
                .total_cmp(&b.predicted_domain_gain)
                .then_with(|| b.bytes.cmp(&a.bytes))
        })
        .cloned()
        .into_iter()
        .collect()
}

fn delta_row_legality(spec: &RunSpec, deltas: &[DeltaOptionReport]) -> (bool, Vec<String>) {
    if deltas.is_empty() {
        return (true, vec!["No delta mutation.".to_string()]);
    }
    let mut notes = Vec::new();
    let mut legal = true;
    for delta in deltas {
        if delta.legality.contains("illegal") {
            legal = false;
            notes.push(format!("{} is marked illegal", delta.name));
        } else if delta.legality.contains("score_first") && spec.eval.legal_score_first {
            notes.push(format!(
                "{} requires score-first update ordering",
                delta.name
            ));
        } else if delta_legality_is_artifact_paid(delta) {
            notes.push(format!(
                "{} is treated as offline-trained/static artifact-paid",
                delta.name
            ));
        } else if delta.legality.contains("score_first") {
            legal = false;
            notes.push(format!(
                "{} mentions score-first semantics but spec legal_score_first=false",
                delta.name
            ));
        } else {
            notes.push(format!("{} legality={}", delta.name, delta.legality));
        }
    }
    notes.sort();
    notes.dedup();
    (legal, notes)
}

fn delta_legality_is_artifact_paid(delta: &DeltaOptionReport) -> bool {
    delta.legality == "no_delta"
        || delta.legality.contains("artifact_paid")
        || delta.legality.contains("paid_in_artifact")
        || delta.legality.contains("offline_train_only")
        || delta.legality.contains("train_only_if_paid")
}

fn delta_domain_beats(
    rows: &[DeltaDomainEvaluationRowReport],
    selected_name: &str,
    candidate_name: &str,
) -> Option<bool> {
    let selected = rows.iter().find(|row| row.plan_name == selected_name)?;
    let candidate = rows.iter().find(|row| row.plan_name == candidate_name)?;
    Some(
        selected.estimated_domain_loss_proxy <= candidate.estimated_domain_loss_proxy
            && selected.predicted_domain_gain >= candidate.predicted_domain_gain,
    )
}

fn prove_selected_delta_low_rank(
    spec: &RunSpec,
    selected_deltas: &[DeltaOptionReport],
) -> PgResult<Vec<DeltaLowRankProofReport>> {
    selected_deltas
        .iter()
        .filter(|delta| delta.rank.is_some())
        .map(|delta| prove_delta_low_rank(spec, delta))
        .collect()
}

fn prove_delta_low_rank(
    spec: &RunSpec,
    delta: &DeltaOptionReport,
) -> PgResult<DeltaLowRankProofReport> {
    use pg_kernels::deltagolf::{
        deltagolf_materialize_delta, deltagolf_weighted_error, deltagolf_weighted_low_rank_delta,
    };

    let requested_rank = delta.rank.unwrap_or(0);
    let (source_rows, source_cols) = delta_source_shape(spec, delta);
    let proof_rows = delta_proof_dim(source_rows, requested_rank);
    let proof_cols = delta_proof_dim(source_cols, requested_rank);
    let full_update = deterministic_delta_update(
        proof_rows,
        proof_cols,
        requested_rank,
        delta_proof_phase(delta, source_rows, source_cols),
    );
    let out_curvature_diag = deterministic_curvature_diag(proof_rows, 0.73);
    let in_curvature_diag = deterministic_curvature_diag(proof_cols, 0.41);
    let zero_delta = vec![0.0f32; proof_rows * proof_cols];
    let zero_delta_weighted_error = deltagolf_weighted_error(
        &zero_delta,
        &full_update,
        &out_curvature_diag,
        &in_curvature_diag,
        proof_rows,
        proof_cols,
    )?;

    let selected = deltagolf_weighted_low_rank_delta(
        &full_update,
        &out_curvature_diag,
        &in_curvature_diag,
        proof_rows,
        proof_cols,
        requested_rank,
    )?;
    let mut materialized = vec![0.0f32; proof_rows * proof_cols];
    deltagolf_materialize_delta(&selected, &mut materialized)?;
    let selected_rank_weighted_error = deltagolf_weighted_error(
        &materialized,
        &full_update,
        &out_curvature_diag,
        &in_curvature_diag,
        proof_rows,
        proof_cols,
    )?;

    let lower_rank_weighted_error = if requested_rank > 1 {
        let lower = deltagolf_weighted_low_rank_delta(
            &full_update,
            &out_curvature_diag,
            &in_curvature_diag,
            proof_rows,
            proof_cols,
            requested_rank - 1,
        )?;
        let mut lower_materialized = vec![0.0f32; proof_rows * proof_cols];
        deltagolf_materialize_delta(&lower, &mut lower_materialized)?;
        Some(deltagolf_weighted_error(
            &lower_materialized,
            &full_update,
            &out_curvature_diag,
            &in_curvature_diag,
            proof_rows,
            proof_cols,
        )?)
    } else {
        Some(zero_delta_weighted_error)
    };
    let error_reduction_vs_zero_pct = if zero_delta_weighted_error > 0.0 {
        100.0 * (1.0 - selected_rank_weighted_error / zero_delta_weighted_error)
    } else {
        0.0
    };
    let selected_rank_no_worse_than_lower_rank = lower_rank_weighted_error
        .map(|lower| selected_rank_weighted_error <= lower + 1e-6)
        .unwrap_or(true);
    let curvature_positive = out_curvature_diag
        .iter()
        .chain(in_curvature_diag.iter())
        .all(|value| value.is_finite() && *value > 0.0);
    let finite = zero_delta_weighted_error.is_finite()
        && selected_rank_weighted_error.is_finite()
        && lower_rank_weighted_error
            .map(f64::is_finite)
            .unwrap_or(true)
        && error_reduction_vs_zero_pct.is_finite()
        && selected
            .singular_values
            .iter()
            .all(|value| value.is_finite());

    Ok(DeltaLowRankProofReport {
        kind: "deltagolf_selected_low_rank_local_proof",
        delta_name: delta.name.clone(),
        delta_type: delta.delta_type,
        requested_rank,
        actual_rank: selected.rank,
        source_rows,
        source_cols,
        proof_rows,
        proof_cols,
        factor_a_elems: selected.a.len(),
        factor_b_elems: selected.b.len(),
        top_singular_values: selected
            .singular_values
            .iter()
            .take(4)
            .map(|value| *value as f64)
            .collect(),
        zero_delta_weighted_error,
        selected_rank_weighted_error,
        lower_rank_weighted_error,
        error_reduction_vs_zero_pct,
        selected_rank_no_worse_than_lower_rank,
        finite,
        curvature_positive,
        evidence_boundary: "local deterministic weighted low-rank proof; not trained domain delta, score-first eval, or private-corpus BPB evidence",
    })
}

fn delta_source_shape(spec: &RunSpec, delta: &DeltaOptionReport) -> (usize, usize) {
    let cfg = spec.model.to_model_config();
    match delta.delta_type {
        "lora" if delta.name.starts_with("lora_mlp_down") => (cfg.model_dim, cfg.mlp_dim),
        "lora" => (cfg.model_dim, cfg.model_dim),
        "low_rank_output_correction" => (cfg.vocab_size, cfg.model_dim),
        "kv_memory_adapter" => {
            let kv_dim = cfg.kv_dim().max(1);
            (kv_dim, kv_dim)
        }
        _ => (cfg.model_dim, cfg.model_dim),
    }
}

fn delta_proof_dim(source_dim: usize, requested_rank: usize) -> usize {
    let source_dim = source_dim.max(1);
    let rank_floor = requested_rank.min(source_dim).max(1);
    source_dim.min(64).max(rank_floor)
}

fn delta_proof_phase(delta: &DeltaOptionReport, source_rows: usize, source_cols: usize) -> f32 {
    let name_hash = delta.name.bytes().fold(0usize, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(byte as usize)
    });
    0.053
        + (name_hash % 97) as f32 * 0.0017
        + (source_rows % 29) as f32 * 0.0023
        + (source_cols % 31) as f32 * 0.0019
}

fn deterministic_delta_update(rows: usize, cols: usize, rank: usize, phase: f32) -> Vec<f32> {
    let mut values = deterministic_modelgolf_values(rows * cols, phase + rank as f32 * 0.013);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let ridge = ((row + 3 * col + rank) % 11) as f32 / 37.0;
            values[idx] += ridge - 0.13 + 0.03 * ((row as f32 + 1.0) * phase).cos();
        }
    }
    values
}

fn deterministic_curvature_diag(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 + 1.0;
            0.75 + 0.20 * (x * phase).sin().abs() + 0.05 * ((i % 7) as f32)
        })
        .collect()
}

fn delta_candidate_sets(spec: &RunSpec) -> Vec<Vec<DeltaOptionReport>> {
    let cfg = spec.model.to_model_config();
    let d = cfg.model_dim;
    let mlp = cfg.mlp_dim;
    let n = cfg.num_layers;
    let kv = cfg.kv_dim();
    vec![
        delta_rank_family(
            "lora_attention_qo",
            "lora",
            &[2, 4, 8, 16],
            |rank| n * 2 * (d * rank + rank * d) * 2,
            1.10,
        ),
        delta_rank_family(
            "lora_mlp_down",
            "lora",
            &[2, 4, 8],
            |rank| n * (mlp * rank + rank * d) * 2,
            1.30,
        ),
        delta_rank_family(
            "low_rank_lm_head",
            "low_rank_output_correction",
            &[1, 2, 4],
            |rank| (cfg.vocab_size * rank + rank * d) * 2,
            0.95,
        ),
        vec![
            off_delta("bias_corrections"),
            delta_option(
                "bias_corrections",
                "bias",
                None,
                n * d * 2,
                0.22,
                "score_first_or_train_only_if_paid_in_artifact",
            ),
        ],
        vec![
            off_delta("ngram_residual_table"),
            delta_option(
                "ngram_residual_table_64kb",
                "ngram_residual",
                None,
                64 * 1024,
                0.42,
                "artifact_paid_static_lookup_only",
            ),
            delta_option(
                "ngram_residual_table_256kb",
                "ngram_residual",
                None,
                256 * 1024,
                0.72,
                "artifact_paid_static_lookup_only",
            ),
        ],
        delta_rank_family(
            "kv_memory_adapter",
            "kv_memory_adapter",
            &[2, 4, 8],
            |rank| n * kv * rank * 2 * 2,
            0.45,
        ),
    ]
}

fn delta_rank_family<F>(
    name: &'static str,
    delta_type: &'static str,
    ranks: &[usize],
    bytes_for_rank: F,
    scale: f64,
) -> Vec<DeltaOptionReport>
where
    F: Fn(usize) -> usize,
{
    let mut options = vec![off_delta(name)];
    for &rank in ranks {
        let bytes = bytes_for_rank(rank);
        let gain = scale * ((rank + 1) as f64).ln();
        options.push(delta_option(
            &format!("{name}_rank{rank}"),
            delta_type,
            Some(rank),
            bytes,
            gain,
            "score_first_or_offline_train_only_if_paid_in_artifact",
        ));
    }
    options
}

fn off_delta(name: &'static str) -> DeltaOptionReport {
    DeltaOptionReport {
        name: format!("{name}_off"),
        delta_type: "none",
        rank: None,
        bytes: 0,
        predicted_domain_gain: 0.0,
        gain_per_byte: 0.0,
        legality: "no_delta",
    }
}

fn delta_option(
    name: &str,
    delta_type: &'static str,
    rank: Option<usize>,
    bytes: usize,
    predicted_domain_gain: f64,
    legality: &'static str,
) -> DeltaOptionReport {
    DeltaOptionReport {
        name: name.to_string(),
        delta_type,
        rank,
        bytes,
        predicted_domain_gain,
        gain_per_byte: predicted_domain_gain / bytes.max(1) as f64,
        legality,
    }
}

fn plan_train(spec: &RunSpec, pack: &PackPlannerReport) -> TrainPlannerReport {
    let total_weighted_gap = pack
        .selected_options
        .iter()
        .map(|option| (8usize.saturating_sub(option.bits as usize)) as f64 * option.bytes as f64)
        .sum::<f64>();
    let total_bytes = pack.selected_bytes.max(1) as f64;
    let quantization_distance_proxy = total_weighted_gap / (8.0 * total_bytes);
    let gradient_norm_target = 0.03;
    let smoothness_proxy = if spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
    {
        0.85
    } else {
        1.0
    };
    let export_gap_upper_bound = gradient_norm_target * quantization_distance_proxy
        + 0.5 * smoothness_proxy * quantization_distance_proxy * quantization_distance_proxy;
    let local_proof = prove_traingolf_local(spec, smoothness_proxy);
    TrainPlannerReport {
        kind: "traingolf_artifact_aware_plan",
        objective: "L_full_precision + lambda*quantization_distance + mu*compression_entropy_proxy + nu*residual_byte_cost",
        quantization_distance_proxy,
        gradient_norm_target,
        smoothness_proxy,
        export_gap_upper_bound,
        late_qat_threshold: spec.train.late_qat_threshold,
        compression_entropy_proxy_weight: 0.05,
        residual_byte_cost_weight: 0.01,
        stages: vec![
            TrainStageReport {
                name: "full_precision_warmup",
                trigger: format!("steps < {}", spec.train.warmup_steps),
                purpose: "learn stable representations before compression pressure",
            },
            TrainStageReport {
                name: "artifact_distance_regularization",
                trigger: "after warmup until warmdown".to_string(),
                purpose: "move weights toward exportable quantization cells",
            },
            TrainStageReport {
                name: "late_qat_export_gap_close",
                trigger: format!("lr_scale <= {:.3}", spec.train.late_qat_threshold),
                purpose: "optimize the final compressed artifact rather than only full precision loss",
            },
        ],
        local_proof,
        notes: vec![
            "The export-gap bound follows smoothness: L(Q(W))-L(W) <= ||grad||*epsilon + L_s*epsilon^2/2.".to_string(),
            "local_proof verifies stop-gradient regularizer behavior and the smooth export-gap bound on a deterministic CPU fixture; pg-model can add those gradients to CPU GradBuffers; pg-train can schedule the regularizer in non-CUDA runs; eval calibration remains.".to_string(),
        ],
    }
}

fn prove_traingolf_local(spec: &RunSpec, smoothness_proxy: f64) -> TrainGolfLocalProofReport {
    use pg_kernels::traingolf::{
        TrainGolfQuantizationConfig, traingolf_export_gap_bound,
        traingolf_quantization_regularizer, traingolf_stop_gradient_regularized_loss,
    };

    let bits = spec.train.artifact_regularization_bits.clamp(2, 8);
    let block_size = spec.train.artifact_regularization_block_size.clamp(1, 64);
    let lambda = if spec.train.artifact_regularization_lambda > 0.0 {
        spec.train.artifact_regularization_lambda
    } else {
        0.25
    };
    let weights = deterministic_modelgolf_values(32, 0.157);
    let config = TrainGolfQuantizationConfig {
        bits,
        block_size,
        lambda,
    };
    let report = traingolf_quantization_regularizer(&weights, config)
        .expect("valid deterministic TrainGolf proof config");
    let step_size = 0.4f32;
    let stepped = weights
        .iter()
        .zip(report.gradient.iter())
        .map(|(weight, grad)| weight - step_size * grad)
        .collect::<Vec<_>>();
    let distance_sq_before = squared_distance_f32(&weights, &report.projected);
    let distance_sq_after_fixed_projection_step = squared_distance_f32(&stepped, &report.projected);
    let distance_reduction_pct = if distance_sq_before > 0.0 {
        100.0 * (1.0 - distance_sq_after_fixed_projection_step / distance_sq_before)
    } else {
        0.0
    };
    let gradient_matches_stop_gradient_objective = weights
        .iter()
        .zip(report.projected.iter())
        .zip(report.gradient.iter())
        .all(|((weight, projected), grad)| {
            let expected = 2.0 * lambda * (weight - projected);
            (*grad - expected).abs() <= 1e-6
        });

    let smoothness = smoothness_proxy.max(1e-6);
    let quadratic_full_loss = quadratic_loss_f32(&weights, smoothness);
    let quadratic_projected_loss = quadratic_loss_f32(&report.projected, smoothness);
    let observed_export_gap = quadratic_projected_loss - quadratic_full_loss;
    let gradient_norm = smoothness * l2_norm(&weights);
    let export_gap_bound =
        traingolf_export_gap_bound(gradient_norm, report.distance_norm, smoothness)
            .expect("valid deterministic export-gap inputs");
    let base_loss = quadratic_full_loss.max(0.0);
    let composed_objective_loss =
        traingolf_stop_gradient_regularized_loss(base_loss, report.regularization_loss)
            .expect("valid deterministic regularized loss");
    let finite = distance_sq_before.is_finite()
        && distance_sq_after_fixed_projection_step.is_finite()
        && distance_reduction_pct.is_finite()
        && report.regularization_loss.is_finite()
        && composed_objective_loss.is_finite()
        && quadratic_full_loss.is_finite()
        && quadratic_projected_loss.is_finite()
        && observed_export_gap.is_finite()
        && export_gap_bound.is_finite();

    TrainGolfLocalProofReport {
        kind: "traingolf_local_regularizer_and_export_gap_proof",
        weights: weights.len(),
        bits,
        block_size,
        lambda: lambda as f64,
        step_size: step_size as f64,
        distance_sq_before,
        distance_sq_after_fixed_projection_step,
        distance_reduction_pct,
        regularization_loss: report.regularization_loss,
        composed_objective_loss,
        quadratic_full_loss,
        quadratic_projected_loss,
        observed_export_gap,
        export_gap_bound,
        export_gap_bound_covers_observed: observed_export_gap <= export_gap_bound + 1e-12,
        gradient_matches_stop_gradient_objective,
        finite,
        evidence_boundary: "local deterministic CPU proof; not GPU/backend integration, trained-artifact BPB, or post-export quality comparison evidence",
    }
}

fn plan_optimizer_comm(spec: &RunSpec) -> OptimizerCommReport {
    let world_size = spec.train.world_size.max(1);
    let optimizer_sharded = spec.train.distributed_optimizer_backend
        == DistributedOptimizerBackend::ShardedParallelMuon;
    let bank_payload_bytes_f32 = spec.model.to_model_config().param_count().saturating_mul(4);
    let replicated_all_reduce_wire_bytes_per_rank = if world_size > 1 {
        scaled_collective_bytes(bank_payload_bytes_f32, 2 * (world_size - 1), world_size)
    } else {
        0
    };
    let sharded_reduce_scatter_wire_bytes_per_rank = if world_size > 1 && optimizer_sharded {
        scaled_collective_bytes(bank_payload_bytes_f32, world_size - 1, world_size)
    } else {
        0
    };
    let sharded_param_all_gather_wire_bytes_per_rank = if world_size > 1 && optimizer_sharded {
        scaled_collective_bytes(bank_payload_bytes_f32, world_size - 1, world_size)
    } else {
        0
    };
    let bf16_shadow_all_gather_wire_bytes_per_rank = if world_size > 1
        && optimizer_sharded
        && spec.runtime.sharded_muon_bf16_shadow_all_gather
    {
        scaled_collective_bytes(bank_payload_bytes_f32 / 2, world_size - 1, world_size)
    } else {
        0
    };
    let sharded_total_wire_bytes_per_rank = sharded_reduce_scatter_wire_bytes_per_rank
        .saturating_add(sharded_param_all_gather_wire_bytes_per_rank)
        .saturating_add(bf16_shadow_all_gather_wire_bytes_per_rank);
    let local_proof = prove_sharded_optimizer_equivalence(world_size);
    let shard_separable_update_contract = local_proof.exact_equivalence && local_proof.finite;

    OptimizerCommReport {
        kind: "modelgolf_optimizer_comm_plan",
        compiler_boundary: "choose replicated or sharded optimizer collectives, graph capture, shadow refresh, and communication overlap contracts under the train/runtime spec",
        train_backend: format!("{:?}", spec.train.backend),
        distributed_optimizer_backend: format!("{:?}", spec.train.distributed_optimizer_backend),
        world_size,
        optimizer_sharded,
        nccl_overlap_mode: format!("{:?}", spec.runtime.nccl_overlap_mode),
        bank_payload_bytes_f32,
        replicated_all_reduce_wire_bytes_per_rank,
        sharded_reduce_scatter_wire_bytes_per_rank,
        sharded_param_all_gather_wire_bytes_per_rank,
        bf16_shadow_all_gather_wire_bytes_per_rank,
        sharded_total_wire_bytes_per_rank,
        owned_optimizer_state_reduction_x: if optimizer_sharded {
            world_size as f64
        } else {
            1.0
        },
        parameter_all_gather_required: optimizer_sharded,
        bf16_shadow_all_gather_requested: spec.runtime.sharded_muon_bf16_shadow_all_gather,
        local_graph_requested: spec.runtime.sharded_muon_local_graph,
        pre_norm_graph_requested: spec.runtime.sharded_muon_pre_norm_graph,
        fused_global_clip_requested: spec.runtime.sharded_muon_fused_global_clip,
        parallel_local_requested: spec.runtime.sharded_muon_parallel_local,
        shard_separable_update_contract,
        local_proof,
        notes: vec![
            "Local proof covers the mathematical shard-separable update equivalence; production claims still need NCCL trace timing and update parity on the target distributed backend.".to_string(),
            "Wire-byte estimates use ring-collective formulas over the model parameter payload and are planning estimates, not profiler measurements.".to_string(),
        ],
    }
}

fn scaled_collective_bytes(payload_bytes: usize, numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        return 0;
    }
    payload_bytes
        .saturating_mul(numerator)
        .div_ceil(denominator)
}

fn prove_sharded_optimizer_equivalence(world_size: usize) -> OptimizerCommLocalProofReport {
    let proof_world_size = world_size.clamp(2, 8);
    let proof_parameter_elems = 37;
    let learning_rate = 0.03125f64;
    let weights = deterministic_modelgolf_values(proof_parameter_elems, 0.19)
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let rank_grads = (0..proof_world_size)
        .map(|rank| {
            deterministic_modelgolf_values(proof_parameter_elems, 0.13 + rank as f32 * 0.07)
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let full_grad = (0..proof_parameter_elems)
        .map(|idx| rank_grads.iter().map(|grads| grads[idx]).sum::<f64>())
        .collect::<Vec<_>>();
    let replicated = weights
        .iter()
        .zip(full_grad.iter())
        .map(|(weight, grad)| weight - learning_rate * grad)
        .collect::<Vec<_>>();

    let chunk = proof_parameter_elems.div_ceil(proof_world_size);
    let mut sharded = weights.clone();
    let mut shard_ranges = Vec::with_capacity(proof_world_size);
    for rank in 0..proof_world_size {
        let start = (rank * chunk).min(proof_parameter_elems);
        let end = ((rank + 1) * chunk).min(proof_parameter_elems);
        for idx in start..end {
            let shard_grad = rank_grads.iter().map(|grads| grads[idx]).sum::<f64>();
            sharded[idx] = weights[idx] - learning_rate * shard_grad;
        }
        shard_ranges.push(OptimizerCommShardRangeReport {
            rank,
            start,
            end,
            elements: end.saturating_sub(start),
        });
    }

    let max_abs_diff = replicated
        .iter()
        .zip(sharded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let replicated_checksum = replicated.iter().sum::<f64>();
    let sharded_checksum = sharded.iter().sum::<f64>();
    let finite = max_abs_diff.is_finite()
        && replicated_checksum.is_finite()
        && sharded_checksum.is_finite()
        && replicated.iter().all(|value| value.is_finite())
        && sharded.iter().all(|value| value.is_finite());

    OptimizerCommLocalProofReport {
        kind: "optimizer_comm_shard_separable_equivalence_proof",
        proof_world_size,
        proof_parameter_elems,
        learning_rate,
        shard_ranges,
        replicated_checksum,
        sharded_checksum,
        max_abs_diff,
        exact_equivalence: finite && max_abs_diff <= 1e-12,
        finite,
        assumption: "after an exact gradient reduction, the optimizer update for each parameter shard depends only on that shard's parameters, gradients, and optimizer state",
        evidence_boundary: "deterministic local algebraic proof; not NCCL runtime, overlap, distributed Muon timing, or target-backend parity evidence",
    }
}

fn plan_kernel_forge(spec: &RunSpec) -> KernelForgeReport {
    let cfg = spec.model.to_model_config();
    let rows_m = spec.train.batch_tokens.max(1);
    let vocab_v = cfg.vocab_size.max(1);
    let hidden_d = cfg.model_dim.max(1);
    let tile_t = vocab_v.clamp(1, 512);
    let full_logits_bytes_f32 = rows_m.saturating_mul(vocab_v).saturating_mul(4);
    let tiled_logits_scratch_bytes_f32 = rows_m.saturating_mul(tile_t).saturating_mul(4);
    let row_stats_scratch_bytes_f32 = rows_m.saturating_mul(8);
    let tiled_scratch_bytes_estimate =
        tiled_logits_scratch_bytes_f32.saturating_add(row_stats_scratch_bytes_f32);
    let exact_without_persistent_logits = matches!(
        spec.model.output_ce_backend,
        OutputCeBackend::TiledRepeatedGemm | OutputCeBackend::FusedExactWmma
    );
    let local_proof = prove_exact_tiled_ce(spec, tile_t);
    KernelForgeReport {
        kind: "kernelforge_plan",
        compiler_boundary: "lower typed model/runtime specs into fused kernel contracts plus reference equivalence tests",
        exact_tiled_ce: ExactTiledCeReport {
            backend_requested: format!("{:?}", spec.model.output_ce_backend),
            exact_without_persistent_logits,
            rows_m,
            vocab_v,
            hidden_d,
            tile_t,
            full_logits_bytes_f32,
            tiled_logits_scratch_bytes_f32,
            row_stats_scratch_bytes_f32,
            tiled_scratch_bytes_estimate,
            scratch_reduction_x: full_logits_bytes_f32 as f64
                / tiled_scratch_bytes_estimate.max(1) as f64,
            gradient_formula: "G_iv=(p_iv-1[v=y_i])*phi'(z_iv), dH=G W, dW=G^T H",
            local_proof,
        },
        fusion_primitives: vec![
            fusion_primitive(
                "rmsnorm_projection_prepack",
                &["hidden", "rms_scale", "packed_weight"],
                &["projected_hidden"],
                "planned_contract",
            ),
            fusion_primitive(
                "qk_norm_rope_q_gain",
                &["q", "k", "rope", "q_gain"],
                &["q_rope", "k_rope"],
                "planned_contract",
            ),
            fusion_primitive(
                "exact_softcapped_tiled_ce",
                &["hidden", "lm_head", "target"],
                &["loss", "d_hidden", "d_lm_head"],
                if exact_without_persistent_logits {
                    "runtime_requested"
                } else {
                    "available_plan_boundary"
                },
            ),
            fusion_primitive(
                "lqer_dequant_epilogue",
                &["packed_weight", "scale", "lqer_a", "lqer_b"],
                &["matmul_output"],
                if spec.quant.lqer.enabled {
                    "runtime_requested"
                } else {
                    "available_plan_boundary"
                },
            ),
            fusion_primitive(
                "sharded_muon_reduce_scatter_update",
                &["bank_grad_shards", "muon_state"],
                &["updated_bank_shards"],
                if spec.train.distributed_optimizer_backend
                    == DistributedOptimizerBackend::ShardedParallelMuon
                {
                    "runtime_requested"
                } else {
                    "available_plan_boundary"
                },
            ),
        ],
        equivalence_tests: vec![
            "forward fused output equals unfused function composition on deterministic fixtures".to_string(),
            "backward fused output equals composed Jacobian transpose against reference gradients".to_string(),
            "tiled CE loss/dH/dW matches full-logit CE within backend tolerance".to_string(),
            "reduce-scatter + local update + all-gather matches replicated update for shard-separable optimizers".to_string(),
        ],
        notes: vec![
            "KernelForge is still mostly a contract generator, but exact_tiled_ce.local_proof verifies CPU loss/dH/dW parity through pg-kernels on a deterministic fixture.".to_string(),
            "Exact tiled CE avoids persistent MxV logits while preserving the mathematical logsumexp and gradient.".to_string(),
        ],
    }
}

fn prove_exact_tiled_ce(spec: &RunSpec, planned_tile_t: usize) -> ExactTiledCeProofReport {
    use pg_kernels::cross_entropy::{
        cross_entropy_backward_asym, cross_entropy_forward_asym,
        tiled_output_cross_entropy_backward_asym, tiled_output_cross_entropy_forward_asym,
    };

    let cfg = spec.model.to_model_config();
    let proof_rows_m = spec.train.batch_tokens.clamp(2, 8);
    let proof_vocab_v = cfg.vocab_size.clamp(2, 64);
    let proof_hidden_d = cfg.model_dim.clamp(1, 32);
    let tile_t = planned_tile_t.max(1).min(proof_vocab_v).clamp(1, 16);
    let softcap_pos = 17.0f32;
    let softcap_neg = 29.0f32;
    let loss_scale = 1.0f32 / proof_rows_m as f32;

    let hidden = deterministic_modelgolf_values(proof_rows_m * proof_hidden_d, 0.23);
    let weight = deterministic_modelgolf_values(proof_vocab_v * proof_hidden_d, 0.41);
    let targets = (0..proof_rows_m)
        .map(|row| ((row * 7 + 1) % proof_vocab_v) as u32)
        .collect::<Vec<_>>();

    let logits = output_logits(
        &hidden,
        &weight,
        proof_rows_m,
        proof_hidden_d,
        proof_vocab_v,
    );
    let mut expected_losses = vec![0.0f32; proof_rows_m];
    cross_entropy_forward_asym(
        &logits,
        &targets,
        &mut expected_losses,
        proof_vocab_v,
        softcap_pos,
        softcap_neg,
    );
    let mut tiled_losses = vec![0.0f32; proof_rows_m];
    tiled_output_cross_entropy_forward_asym(
        &hidden,
        &weight,
        &targets,
        &mut tiled_losses,
        proof_rows_m,
        proof_hidden_d,
        proof_vocab_v,
        tile_t,
        softcap_pos,
        softcap_neg,
    );

    let mut grad_logits = vec![0.0f32; proof_rows_m * proof_vocab_v];
    cross_entropy_backward_asym(
        &logits,
        &targets,
        &mut grad_logits,
        proof_vocab_v,
        softcap_pos,
        softcap_neg,
        loss_scale,
    );
    let expected_d_hidden = grad_logits_times_weight(
        &grad_logits,
        &weight,
        proof_rows_m,
        proof_hidden_d,
        proof_vocab_v,
    );
    let expected_d_weight = grad_logits_t_times_hidden(
        &grad_logits,
        &hidden,
        proof_rows_m,
        proof_hidden_d,
        proof_vocab_v,
    );

    let mut tiled_d_hidden = vec![0.0f32; proof_rows_m * proof_hidden_d];
    let mut tiled_d_weight = vec![0.0f32; proof_vocab_v * proof_hidden_d];
    tiled_output_cross_entropy_backward_asym(
        &hidden,
        &weight,
        &targets,
        &mut tiled_d_hidden,
        &mut tiled_d_weight,
        proof_rows_m,
        proof_hidden_d,
        proof_vocab_v,
        tile_t,
        softcap_pos,
        softcap_neg,
        loss_scale,
    );

    let max_loss_abs_diff = max_abs_diff(&tiled_losses, &expected_losses);
    let max_d_hidden_abs_diff = max_abs_diff(&tiled_d_hidden, &expected_d_hidden);
    let max_d_weight_abs_diff = max_abs_diff(&tiled_d_weight, &expected_d_weight);
    let finite = tiled_losses
        .iter()
        .chain(expected_losses.iter())
        .chain(tiled_d_hidden.iter())
        .chain(expected_d_hidden.iter())
        .chain(tiled_d_weight.iter())
        .chain(expected_d_weight.iter())
        .all(|value| value.is_finite());
    let tolerance = 5e-6f64;
    let full_logits_bytes_f32 = proof_rows_m.saturating_mul(proof_vocab_v).saturating_mul(4);
    let tiled_logits_scratch_bytes_f32 = proof_rows_m.saturating_mul(tile_t).saturating_mul(4);
    let row_stats_scratch_bytes_f32 = proof_rows_m.saturating_mul(8);
    let tiled_scratch_bytes_estimate =
        tiled_logits_scratch_bytes_f32.saturating_add(row_stats_scratch_bytes_f32);

    ExactTiledCeProofReport {
        kind: "kernelforge_exact_tiled_ce_local_proof",
        proof_rows_m,
        proof_vocab_v,
        proof_hidden_d,
        tile_t,
        softcap_pos: softcap_pos as f64,
        softcap_neg: softcap_neg as f64,
        loss_scale: loss_scale as f64,
        full_logits_bytes_f32,
        tiled_logits_scratch_bytes_f32,
        row_stats_scratch_bytes_f32,
        tiled_scratch_bytes_estimate,
        scratch_reduction_x: full_logits_bytes_f32 as f64
            / tiled_scratch_bytes_estimate.max(1) as f64,
        max_loss_abs_diff,
        max_d_hidden_abs_diff,
        max_d_weight_abs_diff,
        parity_ok: finite
            && max_loss_abs_diff <= tolerance
            && max_d_hidden_abs_diff <= tolerance
            && max_d_weight_abs_diff <= tolerance,
        finite,
        evidence_boundary: "local deterministic CPU parity proof; not generated CUDA/Metal kernel parity or backend performance evidence",
    }
}

fn run_kernel_forge_ce_experiment(spec: &RunSpec) -> KernelForgeCeExperimentReport {
    use pg_kernels::cross_entropy::{
        cross_entropy_backward_asym, cross_entropy_forward_asym, mean_loss,
        tiled_output_cross_entropy_backward_asym, tiled_output_cross_entropy_forward_asym,
    };

    let cfg = spec.model.to_model_config();
    let rows_m = spec.train.batch_tokens.clamp(4, 16);
    let vocab_v = cfg.vocab_size.clamp(16, 128);
    let hidden_d = cfg.model_dim.clamp(8, 64);
    let tile_t = vocab_v.clamp(1, 32);
    let softcap_pos = 17.0f32;
    let softcap_neg = 29.0f32;
    let loss_scale = 1.0f32 / rows_m as f32;
    let hidden = deterministic_modelgolf_values(rows_m * hidden_d, 0.313);
    let weight = deterministic_modelgolf_values(vocab_v * hidden_d, 0.071);
    let targets = (0..rows_m)
        .map(|row| ((row * 11 + 3) % vocab_v) as u32)
        .collect::<Vec<_>>();

    let full_forward_start = Instant::now();
    let logits = output_logits(&hidden, &weight, rows_m, hidden_d, vocab_v);
    let mut full_losses = vec![0.0f32; rows_m];
    cross_entropy_forward_asym(
        &logits,
        &targets,
        &mut full_losses,
        vocab_v,
        softcap_pos,
        softcap_neg,
    );
    let full_forward_ms = full_forward_start.elapsed().as_secs_f64() * 1000.0;

    let full_backward_start = Instant::now();
    let mut grad_logits = vec![0.0f32; rows_m * vocab_v];
    cross_entropy_backward_asym(
        &logits,
        &targets,
        &mut grad_logits,
        vocab_v,
        softcap_pos,
        softcap_neg,
        loss_scale,
    );
    let full_d_hidden = grad_logits_times_weight(&grad_logits, &weight, rows_m, hidden_d, vocab_v);
    let full_d_weight =
        grad_logits_t_times_hidden(&grad_logits, &hidden, rows_m, hidden_d, vocab_v);
    let full_backward_ms = full_backward_start.elapsed().as_secs_f64() * 1000.0;

    let tiled_forward_start = Instant::now();
    let mut tiled_losses = vec![0.0f32; rows_m];
    tiled_output_cross_entropy_forward_asym(
        &hidden,
        &weight,
        &targets,
        &mut tiled_losses,
        rows_m,
        hidden_d,
        vocab_v,
        tile_t,
        softcap_pos,
        softcap_neg,
    );
    let tiled_forward_ms = tiled_forward_start.elapsed().as_secs_f64() * 1000.0;

    let tiled_backward_start = Instant::now();
    let mut tiled_d_hidden = vec![0.0f32; rows_m * hidden_d];
    let mut tiled_d_weight = vec![0.0f32; vocab_v * hidden_d];
    tiled_output_cross_entropy_backward_asym(
        &hidden,
        &weight,
        &targets,
        &mut tiled_d_hidden,
        &mut tiled_d_weight,
        rows_m,
        hidden_d,
        vocab_v,
        tile_t,
        softcap_pos,
        softcap_neg,
        loss_scale,
    );
    let tiled_backward_ms = tiled_backward_start.elapsed().as_secs_f64() * 1000.0;

    let full_logits_bytes = rows_m.saturating_mul(vocab_v).saturating_mul(4);
    let full_grad_logits_bytes = full_logits_bytes;
    let full_total_scratch = full_logits_bytes.saturating_add(full_grad_logits_bytes);
    let tiled_logits_scratch = rows_m.saturating_mul(tile_t).saturating_mul(4);
    let tiled_total_scratch = tiled_logits_scratch.saturating_add(rows_m.saturating_mul(8));
    let full_total_ms = full_forward_ms + full_backward_ms;
    let tiled_total_ms = tiled_forward_ms + tiled_backward_ms;
    let max_loss_abs_diff = max_abs_diff(&tiled_losses, &full_losses);
    let max_d_hidden_abs_diff = max_abs_diff(&tiled_d_hidden, &full_d_hidden);
    let max_d_weight_abs_diff = max_abs_diff(&tiled_d_weight, &full_d_weight);
    let parity_ok = max_loss_abs_diff <= 5e-6
        && max_d_hidden_abs_diff <= 5e-6
        && max_d_weight_abs_diff <= 5e-6
        && full_losses
            .iter()
            .chain(tiled_losses.iter())
            .chain(full_d_hidden.iter())
            .chain(tiled_d_hidden.iter())
            .chain(full_d_weight.iter())
            .chain(tiled_d_weight.iter())
            .all(|value| value.is_finite());

    let rows = vec![
        KernelForgeCeExperimentRowReport {
            plan_name: "full_logits_reference".to_string(),
            materializes_persistent_logits: true,
            logits_scratch_bytes_f32: full_logits_bytes,
            backward_grad_logits_bytes_f32: full_grad_logits_bytes,
            total_ce_scratch_bytes_f32: full_total_scratch,
            memory_reduction_vs_full_x: 1.0,
            forward_ms_cpu: full_forward_ms,
            backward_ms_cpu: full_backward_ms,
            total_ms_cpu: full_total_ms,
            loss_mean: mean_loss(&full_losses) as f64,
            notes: vec![
                "Materializes full MxV logits and grad-logits for the reference chain-rule path."
                    .to_string(),
            ],
        },
        KernelForgeCeExperimentRowReport {
            plan_name: "tiled_logit_free_ce".to_string(),
            materializes_persistent_logits: false,
            logits_scratch_bytes_f32: tiled_logits_scratch,
            backward_grad_logits_bytes_f32: 0,
            total_ce_scratch_bytes_f32: tiled_total_scratch,
            memory_reduction_vs_full_x: full_total_scratch as f64
                / tiled_total_scratch.max(1) as f64,
            forward_ms_cpu: tiled_forward_ms,
            backward_ms_cpu: tiled_backward_ms,
            total_ms_cpu: tiled_total_ms,
            loss_mean: mean_loss(&tiled_losses) as f64,
            notes: vec![
                "Streams vocabulary tiles and recomputes dot products instead of storing persistent MxV logits."
                    .to_string(),
                "CPU timing is a local reference timing; a generated CUDA/Metal kernel may make a different speed/memory tradeoff."
                    .to_string(),
            ],
        },
    ];

    KernelForgeCeExperimentReport {
        kind: "kernelforge_exact_tiled_ce_experiment",
        protocol: "compare_full_logits_ce_vs_tiled_logit_free_ce_loss_gradients_memory_and_cpu_time",
        source_spec_name: spec.name.clone(),
        rows_m,
        vocab_v,
        hidden_d,
        tile_t,
        rows,
        full_vs_tiled: KernelForgeCeExperimentComparisonReport {
            max_loss_abs_diff,
            max_d_hidden_abs_diff,
            max_d_weight_abs_diff,
            parity_ok,
            tiled_uses_less_scratch: tiled_total_scratch < full_total_scratch,
            tiled_cpu_speedup_vs_full: full_total_ms / tiled_total_ms.max(1e-12),
            evidence_boundary: "local deterministic CPU reference comparison; not generated CUDA/Metal kernel parity or backend performance evidence",
        },
        status: if parity_ok {
            "local_experiment_ready".to_string()
        } else {
            "local_experiment_failed".to_string()
        },
        evidence_boundary: "bounded deterministic CPU Experiment 2 harness; proves local loss/dH/dW parity and reports local memory/timing, but not GPU kernel lowering or production performance",
    }
}

fn output_logits(hidden: &[f32], weight: &[f32], m: usize, d: usize, vocab: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; m * vocab];
    for row in 0..m {
        for v in 0..vocab {
            let mut sum = 0.0f32;
            for col in 0..d {
                sum += hidden[row * d + col] * weight[v * d + col];
            }
            logits[row * vocab + v] = sum;
        }
    }
    logits
}

fn grad_logits_times_weight(
    grad_logits: &[f32],
    weight: &[f32],
    m: usize,
    d: usize,
    vocab: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * d];
    for row in 0..m {
        for v in 0..vocab {
            let grad = grad_logits[row * vocab + v];
            for col in 0..d {
                out[row * d + col] += grad * weight[v * d + col];
            }
        }
    }
    out
}

fn grad_logits_t_times_hidden(
    grad_logits: &[f32],
    hidden: &[f32],
    m: usize,
    d: usize,
    vocab: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab * d];
    for row in 0..m {
        for v in 0..vocab {
            let grad = grad_logits[row * vocab + v];
            for col in 0..d {
                out[v * d + col] += grad * hidden[row * d + col];
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn plan_wind(
    spec: &RunSpec,
    plan: &ExecutionPlan,
    pack: &PackPlannerReport,
    cache: &CachePlannerReport,
    delta: &DeltaPlannerReport,
) -> ModelGolfWindReport {
    let cfg = spec.model.to_model_config();
    let layer_scale = cfg.num_layers as f64 / 11.0;
    let token_scale = spec.train.batch_tokens.max(1) as f64 / 524_288.0;
    let precision_scale = if spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore {
        0.82
    } else {
        1.0
    };
    let ce_scale = match spec.model.output_ce_backend {
        OutputCeBackend::ChunkedBf16Cache => 1.0,
        OutputCeBackend::TiledRepeatedGemm => 0.82,
        OutputCeBackend::FusedExactWmma => 0.62,
    };
    let optimizer_scale = if spec.train.distributed_optimizer_backend
        == DistributedOptimizerBackend::ShardedParallelMuon
    {
        0.78
    } else {
        1.0
    };
    let forward_ms = 42.0 * layer_scale * token_scale * precision_scale;
    let backward_ms = 43.0 * layer_scale * token_scale * precision_scale;
    let ce_ms = 18.0 * token_scale * ce_scale;
    let optimizer_ms = 16.0 * optimizer_scale;
    let train_step_ms_estimate = forward_ms + backward_ms + ce_ms + optimizer_ms;
    let expected_steps_in_600s = (600_000.0 / train_step_ms_estimate.max(1.0)).floor() as usize;
    let expected_train_wall_seconds =
        spec.train.total_iterations as f64 * train_step_ms_estimate / 1_000.0;
    let mut bottlenecks = [
        ("forward_replay", forward_ms),
        ("backward", backward_ms),
        ("output_ce", ce_ms),
        ("optimizer_update", optimizer_ms),
    ];
    bottlenecks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_bottlenecks = bottlenecks
        .iter()
        .take(3)
        .map(|(name, ms)| format!("{name}:{ms:.3}ms"))
        .collect::<Vec<_>>();
    let pareto_candidates = vec![
        WindParetoCandidate {
            name: "ModelGolf Pack selected artifact".to_string(),
            expected_delta_ms_per_step: if pack.selected_options.iter().any(|o| o.bits <= 4) {
                -2.0
            } else {
                -0.5
            },
            expected_delta_bytes: pack.selected_bytes as isize
                - plan
                    .quant_layout
                    .target_artifact_bytes
                    .min(pack.selected_bytes) as isize,
            evidence_required: "final trained export byte proof plus BPB eval",
        },
        WindParetoCandidate {
            name: "CacheGolf KV cache policy".to_string(),
            expected_delta_ms_per_step: -cache.selected.predicted_attention_speedup.max(1.0).ln(),
            expected_delta_bytes: cache.selected.estimated_cache_bytes as isize
                - cache.fp16_cache_bytes as isize,
            evidence_required: "long-context runtime and attention parity",
        },
        WindParetoCandidate {
            name: "DeltaGolf selected private delta".to_string(),
            expected_delta_ms_per_step: 0.2,
            expected_delta_bytes: delta.selected_bytes as isize,
            evidence_required: "domain BPB/perplexity improvement at equal artifact bytes",
        },
        WindParetoCandidate {
            name: "KernelForge exact tiled CE".to_string(),
            expected_delta_ms_per_step: -ce_ms * 0.30,
            expected_delta_bytes: 0,
            evidence_required: "loss/dH/dW parity and GPU timing",
        },
    ];
    let recommendation = pareto_candidates
        .iter()
        .min_by(|a, b| {
            a.expected_delta_ms_per_step
                .total_cmp(&b.expected_delta_ms_per_step)
        })
        .map(|candidate| {
            format!(
                "prioritize {} next; required evidence: {}",
                candidate.name, candidate.evidence_required
            )
        })
        .unwrap_or_else(|| "collect a calibrated trace before choosing the next run".to_string());
    ModelGolfWindReport {
        kind: "modelgolf_wind_tunnel_summary",
        estimate_only: true,
        train_step_ms_estimate,
        expected_steps_in_600s,
        expected_train_wall_seconds,
        artifact_bytes_estimate: pack.selected_bytes,
        top_bottlenecks,
        pareto_candidates,
        recommendation,
    }
}

struct ScaleGolfPlanInputs<'a> {
    spec: &'a RunSpec,
    resource_contract: &'a ResourceContractReport,
    cost_model: &'a ConstraintCostModelReport,
    artifact_ir: &'a ModelArtifactIrReport,
    pack: &'a PackPlannerReport,
    cache: &'a CachePlannerReport,
    delta: &'a DeltaPlannerReport,
    train: &'a TrainPlannerReport,
    optimizer_comm: &'a OptimizerCommReport,
    kernel_forge: &'a KernelForgeReport,
    wind_tunnel: &'a ModelGolfWindReport,
}

fn plan_scale_golf(inputs: ScaleGolfPlanInputs<'_>) -> ScaleGolfReport {
    let ScaleGolfPlanInputs {
        spec,
        resource_contract,
        cost_model,
        artifact_ir,
        pack,
        cache,
        delta,
        train,
        optimizer_comm,
        kernel_forge,
        wind_tunnel,
    } = inputs;
    let shared_compiler_surfaces = vec![
        "resource_contract",
        "model_ir",
        "artifact_compiler",
        "runtime_planner",
        "evaluator",
        "cost_model",
    ];
    let pack_local_ready =
        pack.feasible && pack.selected_bytes <= resource_contract.artifact_budget_bytes;
    let cache_local_ready = cache.feasible
        && cache.selected_policy_proof.bound_covers_observed_error
        && cache.selected_policy_proof.dequant_attention_parity_ok;
    let delta_local_ready = delta.selected_bytes > 0
        && delta.score_first_legality_audit.pass
        && delta
            .domain_evaluation
            .selected_beats_no_delta_proxy
            .unwrap_or(false);
    let train_local_ready = train.local_proof.finite
        && train.local_proof.gradient_matches_stop_gradient_objective
        && train.local_proof.export_gap_bound_covers_observed;
    let optimizer_local_ready = optimizer_comm.local_proof.finite
        && optimizer_comm.local_proof.exact_equivalence
        && optimizer_comm.shard_separable_update_contract;
    let kernel_local_ready = kernel_forge.exact_tiled_ce.local_proof.finite
        && kernel_forge.exact_tiled_ce.local_proof.parity_ok;
    let wind_local_ready = wind_tunnel.estimate_only && !wind_tunnel.pareto_candidates.is_empty();
    let resource_local_ready = resource_contract.artifact_budget_bytes > 0
        && resource_contract.training_time_budget_seconds.is_finite()
        && resource_contract.training_time_budget_seconds > 0.0
        && cost_model.train_energy_budget_joules_proxy.is_finite()
        && cost_model.train_energy_budget_joules_proxy > 0.0
        && artifact_ir.total_parameter_elems_estimate > 0;

    let track_surfaces = shared_compiler_surfaces.clone();
    let tracks = vec![
        ScaleGolfTrackReport {
            track: "PG-Lite",
            target: "laptop / Apple Silicon / tiny artifacts",
            primary_constraint: "local artifact bytes and portable runtime memory",
            hardware_scope: "single local CPU/Metal host".to_string(),
            artifact_budget_bytes: Some(resource_contract.artifact_budget_bytes.min(1_048_576)),
            memory_budget_bytes: resource_contract.memory_budget_bytes,
            context_tokens: resource_contract.context_tokens.min(4096),
            training_time_budget_seconds: Some(resource_contract.training_time_budget_seconds.min(300.0)),
            train_world_size: 1,
            uses_surfaces: track_surfaces.clone(),
            required_modules: vec!["ModelGolf Pack", "TrainGolf", "KernelForge"],
            local_ready: resource_local_ready && pack_local_ready && train_local_ready,
            release_ready: false,
            quality_per_resource_proxy: scale_quality_resource_proxy(
                pack.selected_quality_loss,
                resource_contract.artifact_budget_bytes.min(1_048_576),
                cost_model.train_energy_budget_joules_proxy.min(300.0 * resource_contract.nominal_power_watts_proxy),
            ),
            recommended_next_action:
                "run a PG-Lite-sized export/reload/BPB packet and keep it labeled proxy-only".to_string(),
            remote_evidence_needed: vec![
                "local corpus BPB with artifact byte proof",
                "runtime-specific decode timing",
            ],
        },
        ScaleGolfTrackReport {
            track: "ParameterGolf",
            target: "16MB / 8xH100 / 600s",
            primary_constraint: "artifact bytes, training wall time, and leaderboard BPB",
            hardware_scope: format!("{} / {}", resource_contract.hardware, resource_contract.runtime),
            artifact_budget_bytes: Some(resource_contract.artifact_budget_bytes),
            memory_budget_bytes: resource_contract.memory_budget_bytes,
            context_tokens: resource_contract.context_tokens,
            training_time_budget_seconds: Some(resource_contract.training_time_budget_seconds),
            train_world_size: spec.train.world_size,
            uses_surfaces: track_surfaces.clone(),
            required_modules: vec![
                "ModelGolf Pack",
                "TrainGolf",
                "Optimizer/Comm Compiler",
                "KernelForge",
                "Wind Tunnel",
            ],
            local_ready: resource_local_ready
                && pack_local_ready
                && train_local_ready
                && optimizer_local_ready
                && kernel_local_ready
                && wind_local_ready,
            release_ready: false,
            quality_per_resource_proxy: scale_quality_resource_proxy(
                pack.selected_quality_loss,
                pack.selected_bytes.max(1),
                cost_model.train_energy_budget_joules_proxy,
            ),
            recommended_next_action:
                "collect final artifact, held-out BPB, wall-power, NCCL, kernel, and profiler evidence"
                    .to_string(),
            remote_evidence_needed: vec![
                "final trained artifact byte proof",
                "official held-out BPB/perplexity",
                "measured wall time and power",
                "NCCL and generated-kernel traces",
            ],
        },
        ScaleGolfTrackReport {
            track: "DeltaGolf",
            target: "frozen large base + tiny deltas",
            primary_constraint: "score-first legality and delta bytes",
            hardware_scope: "frozen base runtime plus local/private delta trainer".to_string(),
            artifact_budget_bytes: Some(delta.delta_budget_bytes),
            memory_budget_bytes: resource_contract.memory_budget_bytes,
            context_tokens: resource_contract.context_tokens,
            training_time_budget_seconds: None,
            train_world_size: 1,
            uses_surfaces: track_surfaces.clone(),
            required_modules: vec!["DeltaGolf", "ModelGolf Pack", "CacheGolf"],
            local_ready: resource_local_ready && delta_local_ready,
            release_ready: false,
            quality_per_resource_proxy: scale_delta_resource_proxy(delta),
            recommended_next_action:
                "train a byte-constrained domain delta and bind score-first/legal review evidence"
                    .to_string(),
            remote_evidence_needed: vec![
                "domain full-update or curvature estimates",
                "trained delta bytes",
                "score-first trace or legal review",
                "equal-byte domain BPB/perplexity",
            ],
        },
        ScaleGolfTrackReport {
            track: "LongContext",
            target: "fixed memory / maximum context",
            primary_constraint: "KV-cache memory per batch at target quality",
            hardware_scope: format!("{} long-context runtime", resource_contract.runtime),
            artifact_budget_bytes: Some(pack.selected_bytes),
            memory_budget_bytes: resource_contract.memory_budget_bytes,
            context_tokens: cache
                .long_context_eval
                .max_context_tokens_under_budget
                .max(resource_contract.context_tokens),
            training_time_budget_seconds: None,
            train_world_size: spec.train.world_size.max(1),
            uses_surfaces: track_surfaces.clone(),
            required_modules: vec!["CacheGolf", "ModelGolf Pack", "Wind Tunnel"],
            local_ready: resource_local_ready && cache_local_ready,
            release_ready: false,
            quality_per_resource_proxy: scale_quality_resource_proxy(
                cache.selected.estimated_attention_error_bound,
                cache.selected.estimated_cache_bytes.max(1),
                resource_contract.nominal_power_watts_proxy.max(1.0),
            ),
            recommended_next_action:
                "lower the selected CacheGolf policy to fused backend kernels and run long-context eval"
                    .to_string(),
            remote_evidence_needed: vec![
                "fused dequant-attention parity",
                "long-context BPB/perplexity",
                "decode latency and memory telemetry",
            ],
        },
        ScaleGolfTrackReport {
            track: "ScaleGolf",
            target: "larger cluster / target quality per dollar",
            primary_constraint: "quality improvement per measured dollar/joule",
            hardware_scope: "multi-node or larger accelerator pool".to_string(),
            artifact_budget_bytes: Some(pack.selected_bytes),
            memory_budget_bytes: resource_contract.memory_budget_bytes,
            context_tokens: resource_contract.context_tokens,
            training_time_budget_seconds: Some(resource_contract.training_time_budget_seconds),
            train_world_size: spec.train.world_size.max(8),
            uses_surfaces: track_surfaces.clone(),
            required_modules: vec![
                "ModelGolf Core",
                "ModelGolf Pack",
                "TrainGolf",
                "Optimizer/Comm Compiler",
                "KernelForge",
                "Wind Tunnel",
            ],
            local_ready: resource_local_ready
                && pack_local_ready
                && train_local_ready
                && optimizer_local_ready
                && kernel_local_ready
                && wind_local_ready,
            release_ready: false,
            quality_per_resource_proxy: scale_quality_resource_proxy(
                pack.selected_quality_loss + wind_tunnel.train_step_ms_estimate / 10_000.0,
                pack.selected_bytes.max(1),
                cost_model.train_energy_budget_joules_proxy.max(1.0),
            ),
            recommended_next_action:
                "collect calibrated cost/quality traces before scaling the selected run family"
                    .to_string(),
            remote_evidence_needed: vec![
                "fresh profiler traces",
                "measured energy or dollar cost",
                "held-out quality at scaled size",
                "distributed optimizer and kernel performance traces",
            ],
        },
    ];

    let invariant = scale_golf_invariant(&tracks);
    let current_contract_track =
        scale_current_contract_track(resource_contract, spec, cache, delta).to_string();
    let local_ready = invariant.all_tracks_use_resource_contract
        && invariant.all_tracks_use_model_ir
        && invariant.all_tracks_use_artifact_compiler
        && invariant.all_tracks_use_runtime_planner
        && invariant.all_tracks_use_evaluator
        && invariant.all_tracks_use_cost_model
        && tracks.iter().any(|track| track.local_ready);
    let next_action = tracks
        .iter()
        .find(|track| track.track == current_contract_track)
        .or_else(|| tracks.iter().find(|track| track.local_ready))
        .map(|track| track.recommended_next_action.clone())
        .unwrap_or_else(|| {
            "fix local planner readiness before selecting a scale track".to_string()
        });

    ScaleGolfReport {
        kind: "scalegolf_track_planner",
        thesis: "use constraints as a unifying language across laptop, record, delta, long-context, and larger-cluster LM systems",
        current_contract_track,
        shared_compiler_surfaces,
        tracks,
        invariant,
        local_ready,
        release_ready: false,
        next_action,
        evidence_boundary: "ScaleGolf is a cross-track planning report; release claims require the per-track measured evidence listed on each row",
    }
}

fn scale_golf_invariant(tracks: &[ScaleGolfTrackReport]) -> ScaleGolfInvariantReport {
    let has_surface =
        |track: &ScaleGolfTrackReport, surface: &str| track.uses_surfaces.contains(&surface);
    ScaleGolfInvariantReport {
        all_tracks_use_resource_contract: tracks
            .iter()
            .all(|track| has_surface(track, "resource_contract")),
        all_tracks_use_model_ir: tracks.iter().all(|track| has_surface(track, "model_ir")),
        all_tracks_use_artifact_compiler: tracks
            .iter()
            .all(|track| has_surface(track, "artifact_compiler")),
        all_tracks_use_runtime_planner: tracks
            .iter()
            .all(|track| has_surface(track, "runtime_planner")),
        all_tracks_use_evaluator: tracks.iter().all(|track| has_surface(track, "evaluator")),
        all_tracks_use_cost_model: tracks.iter().all(|track| has_surface(track, "cost_model")),
        track_count: tracks.len(),
        expected_track_count: 5,
    }
}

fn scale_current_contract_track(
    resource_contract: &ResourceContractReport,
    spec: &RunSpec,
    cache: &CachePlannerReport,
    delta: &DeltaPlannerReport,
) -> &'static str {
    if resource_contract.context_tokens >= 32_768
        || cache.long_context_eval.max_context_tokens_under_budget >= 32_768
    {
        "LongContext"
    } else if delta.delta_budget_bytes >= resource_contract.artifact_budget_bytes / 8
        && resource_contract.artifact_budget_bytes > 0
    {
        "DeltaGolf"
    } else if spec.train.world_size > 8 || resource_contract.hardware.contains("cluster") {
        "ScaleGolf"
    } else if resource_contract.artifact_budget_bytes <= 20 * 1024 * 1024
        && resource_contract.training_time_budget_seconds <= 900.0
    {
        "ParameterGolf"
    } else {
        "PG-Lite"
    }
}

fn scale_quality_resource_proxy(loss_proxy: f64, bytes: usize, energy_joules: f64) -> f64 {
    if !loss_proxy.is_finite() || !energy_joules.is_finite() || bytes == 0 || energy_joules <= 0.0 {
        return f64::INFINITY;
    }
    loss_proxy.max(0.0) * (bytes as f64).log2().max(1.0) * energy_joules.log10().max(1.0)
}

fn scale_delta_resource_proxy(delta: &DeltaPlannerReport) -> f64 {
    if !delta.selected_predicted_gain.is_finite() || delta.selected_bytes == 0 {
        return f64::INFINITY;
    }
    -delta.selected_predicted_gain / delta.selected_bytes as f64
}

#[derive(Debug, Clone)]
struct WindExperimentCandidate {
    name: String,
    quant_spec: QuantSpec,
    cache_k_bits: u8,
    cache_v_bits: u8,
    delta_budget_bytes: usize,
    estimated_artifact_bytes: usize,
    estimated_artifact_budget_fit: bool,
    estimated_cache_bytes: usize,
    estimated_memory_budget_fit: Option<bool>,
    estimated_delta_bytes: usize,
    deployment_artifact_ratio: f64,
    deployment_cache_ratio: f64,
    deployment_artifact_overflow_ratio: f64,
    deployment_memory_overflow_ratio: f64,
    cheap_score: f64,
    cheap_quality_loss: f64,
    cheap_cache_error_proxy: f64,
    cheap_delta_gain: f64,
}

#[derive(Debug, Clone)]
struct WindExperimentFullEval {
    artifact_bytes: usize,
    bpb_proxy: f64,
    score: f64,
    loss_delta_abs: f64,
    decode_tokens_per_second: f64,
    lqer_groups: Vec<String>,
}

fn run_modelgolf_wind_experiment(
    spec: &RunSpec,
    variant_fingerprint: &str,
    artifact_budget_bytes: usize,
    context_tokens: usize,
    batch_sequences: usize,
    memory_budget_bytes: Option<usize>,
) -> PgResult<ModelGolfWindExperimentReport> {
    let mut candidates = wind_experiment_candidates(
        spec,
        artifact_budget_bytes,
        context_tokens,
        batch_sequences,
        memory_budget_bytes,
    )?;
    candidates.sort_by(|a, b| {
        b.estimated_artifact_budget_fit
            .cmp(&a.estimated_artifact_budget_fit)
            .then_with(
                || match (a.estimated_memory_budget_fit, b.estimated_memory_budget_fit) {
                    (Some(a_fit), Some(b_fit)) => b_fit.cmp(&a_fit),
                    _ => std::cmp::Ordering::Equal,
                },
            )
            .then_with(|| a.cheap_score.total_cmp(&b.cheap_score))
            .then_with(|| a.estimated_artifact_bytes.cmp(&b.estimated_artifact_bytes))
            .then_with(|| a.name.cmp(&b.name))
    });

    let full_eval_top_k = candidates.len().min(8);
    let config = pack_experiment_model_config(spec);
    let mut model = GptModel::new(config.clone());
    model.fill_deterministic();
    let (input_ids, targets) = artifact_proof_tokens(config.vocab_size, config.eval_seq_len);
    let pre_export_loss = model_smoke_loss(&model, &input_ids, &targets)?;

    let mut rows = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| wind_experiment_row_from_candidate(candidate, index + 1))
        .collect::<Vec<_>>();

    for index in 0..full_eval_top_k {
        let full_eval = wind_experiment_full_eval(
            &model,
            &input_ids,
            &targets,
            pre_export_loss,
            variant_fingerprint,
            &candidates[index],
        )?;
        rows[index].full_evaluated = true;
        rows[index].full_eval_artifact_bytes = Some(full_eval.artifact_bytes);
        rows[index].full_eval_bpb_proxy = Some(full_eval.bpb_proxy);
        rows[index].full_eval_score = Some(full_eval.score);
        rows[index].full_eval_loss_delta_abs = Some(full_eval.loss_delta_abs);
        rows[index].full_eval_decode_tokens_per_second = Some(full_eval.decode_tokens_per_second);
        rows[index].full_eval_lqer_groups = full_eval.lqer_groups;
        rows[index]
            .notes
            .push("Full proxy eval exported, strict-reloaded, smoke-scored, and combined with deployment-scale artifact/cache budget pressure.".to_string());
    }

    assign_wind_experiment_eval_ranks(&mut rows);
    let evaluated = rows
        .iter()
        .filter(|row| row.full_evaluated)
        .collect::<Vec<_>>();
    let cheap_scores = evaluated
        .iter()
        .map(|row| row.cheap_rank as f64)
        .collect::<Vec<_>>();
    let full_scores = evaluated
        .iter()
        .filter_map(|row| row.full_eval_score)
        .collect::<Vec<_>>();
    let holdout_spearman = wind_spearman_rank_correlation(&cheap_scores, &full_scores);
    let rank_errors = evaluated
        .iter()
        .filter_map(|row| row.rank_error.map(|error| error.unsigned_abs() as f64))
        .collect::<Vec<_>>();
    let mean_abs_rank_error = mean_nonempty(&rank_errors);
    let best_cheap_candidate = rows.first().map(|row| row.candidate_name.clone());
    let best_full_candidate = rows
        .iter()
        .filter_map(|row| {
            row.full_eval_score
                .map(|score| (score, row.candidate_name.clone()))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, name)| name);
    let top_3_overlap_count = wind_top_k_overlap_count(&rows, 3);
    let calibration_pass = holdout_spearman.is_some_and(|value| value.is_finite() && value >= 0.50);
    let metrics_finite = full_eval_top_k > 1
        && holdout_spearman.is_some_and(f64::is_finite)
        && rows
            .iter()
            .take(full_eval_top_k)
            .all(|row| row.full_eval_score.is_some_and(f64::is_finite));
    let status = if metrics_finite && calibration_pass {
        "local_wind_experiment_ready"
    } else if metrics_finite {
        "local_wind_experiment_needs_calibration"
    } else {
        "local_wind_experiment_failed"
    }
    .to_string();

    Ok(ModelGolfWindExperimentReport {
        kind: "modelgolf_wind_experiment_candidate_ranking",
        protocol: "generate_24_modelgolf_candidates_rank_by_cheap_proxy_export_reload_eval_top_8_and_measure_rank_correlation",
        source_spec_name: spec.name.clone(),
        candidate_count: rows.len(),
        cheap_ranked_count: rows.len(),
        full_evaluated_count: full_eval_top_k,
        full_eval_top_k,
        holdout_spearman,
        mean_abs_rank_error,
        calibration_pass,
        top_3_overlap_count,
        best_cheap_candidate,
        best_full_candidate,
        rows,
        status,
        evidence_boundary: "local deterministic Experiment 5 harness; ranks ModelGolf pack/cache/delta candidates and validates top candidates by bounded export/reload smoke eval, but not fresh profiler traces or full held-out BPB/runtime evidence",
    })
}

fn wind_experiment_candidates(
    spec: &RunSpec,
    artifact_budget_bytes: usize,
    context_tokens: usize,
    batch_sequences: usize,
    memory_budget_bytes: Option<usize>,
) -> PgResult<Vec<WindExperimentCandidate>> {
    let cfg = spec.model.to_model_config();
    let mut candidates = Vec::new();
    let mut index = 0usize;
    for matrix_bits in [4u8, 5, 6] {
        for mlp_bits in [4u8, 5, 6, 7] {
            for embed_bits in [4u8, 6] {
                let mut candidate_spec = spec.clone();
                candidate_spec.quant.scheme = if matrix_bits <= 4 && mlp_bits <= 4 {
                    QuantScheme::Aggressive
                } else if matrix_bits >= 6 || mlp_bits >= 6 {
                    QuantScheme::MixedInt5Int6
                } else {
                    QuantScheme::GptqLiteInt6
                };
                candidate_spec.quant.matrix_bits = matrix_bits;
                candidate_spec.quant.mlp_bits = mlp_bits;
                candidate_spec.quant.embed_bits = embed_bits;
                candidate_spec.quant.attn_gate_bits = 8;
                candidate_spec.quant.target_artifact_bytes = artifact_budget_bytes;
                candidate_spec.quant.lqer.enabled =
                    !(index + matrix_bits as usize).is_multiple_of(3);
                candidate_spec.quant.lqer.rank = if candidate_spec.quant.lqer.enabled {
                    1 + (index % 4)
                } else {
                    spec.quant.lqer.rank.max(1)
                };
                candidate_spec.quant.lqer.top_k = if candidate_spec.quant.lqer.enabled {
                    1 + ((index / 2) % 4)
                } else {
                    0
                };
                candidate_spec.quant.lqer.a_bits = candidate_spec.quant.lqer.a_bits.clamp(2, 8);
                candidate_spec.quant.lqer.b_bits = candidate_spec.quant.lqer.b_bits.clamp(4, 8);
                candidate_spec.quant.lqer.group_size = candidate_spec.quant.lqer.group_size.max(16);

                let manifest = pg_quant::compile_quant_layout_manifest(
                    &candidate_spec.quant,
                    Some(&candidate_spec.model.to_model_config()),
                )?;
                let groups = modelgolf_tensor_groups(&manifest.groups);
                let (estimated_artifact_bytes, cheap_quality_loss) =
                    wind_pack_candidate_summary(&groups, &candidate_spec);

                let cache_k_bits = [3u8, 4, 5, 6][index % 4];
                let cache_v_bits = [3u8, 4, 5, 6][(index / 3) % 4];
                let estimated_cache_bytes = cache_policy_estimated_bytes_for_dims(
                    cfg.num_layers.max(1),
                    cfg.num_kv_heads.max(1),
                    cfg.head_dim.max(1),
                    context_tokens.max(1),
                    batch_sequences.max(1),
                    128,
                    "paged",
                    cache_k_bits,
                    cache_v_bits,
                );
                let delta_budget_bytes = [0usize, 64 * 1024, 256 * 1024, 1_000_000][index % 4]
                    .min(artifact_budget_bytes.saturating_div(2).max(64 * 1024));
                let (estimated_delta_bytes, cheap_delta_gain) =
                    wind_delta_candidate_summary(&candidate_spec, delta_budget_bytes);
                let cheap_cache_error_proxy =
                    wind_cache_error_proxy(cache_k_bits, cache_v_bits, cfg.head_dim.max(1));
                let artifact_ratio =
                    estimated_artifact_bytes as f64 / artifact_budget_bytes.max(1) as f64;
                let estimated_artifact_budget_fit =
                    estimated_artifact_bytes <= artifact_budget_bytes;
                let cache_ratio = memory_budget_bytes
                    .map(|budget| estimated_cache_bytes as f64 / budget.max(1) as f64)
                    .unwrap_or_else(|| {
                        let fp16 = cachegolf_experiment_fp16_cache_bytes(
                            &cfg,
                            context_tokens.max(1),
                            batch_sequences.max(1),
                        );
                        estimated_cache_bytes as f64 / fp16.max(1) as f64
                    });
                let estimated_memory_budget_fit =
                    memory_budget_bytes.map(|budget| estimated_cache_bytes <= budget);
                let artifact_overflow =
                    estimated_artifact_bytes.saturating_sub(artifact_budget_bytes) as f64
                        / artifact_budget_bytes.max(1) as f64;
                let memory_overflow = memory_budget_bytes
                    .map(|_| (cache_ratio - 1.0).max(0.0))
                    .unwrap_or(0.0);
                let lqer_runtime_penalty = if candidate_spec.quant.lqer.enabled {
                    0.0025 * candidate_spec.quant.lqer.top_k as f64
                } else {
                    0.0
                };
                let cheap_score = cheap_quality_loss
                    + cheap_cache_error_proxy
                    + 0.02 * artifact_ratio
                    + 0.015 * cache_ratio.min(10.0)
                    + 100.0 * artifact_overflow
                    + 50.0 * memory_overflow
                    + lqer_runtime_penalty
                    - 0.015 * cheap_delta_gain;
                candidates.push(WindExperimentCandidate {
                    name: format!(
                        "m{matrix_bits}_mlp{mlp_bits}_e{embed_bits}_lqer{}_k{cache_k_bits}_v{cache_v_bits}_d{}kb",
                        candidate_spec.quant.lqer.top_k,
                        delta_budget_bytes / 1024
                    ),
                    quant_spec: candidate_spec.quant,
                    cache_k_bits,
                    cache_v_bits,
                    delta_budget_bytes,
                    estimated_artifact_bytes,
                    estimated_artifact_budget_fit,
                    estimated_cache_bytes,
                    estimated_memory_budget_fit,
                    estimated_delta_bytes,
                    deployment_artifact_ratio: artifact_ratio,
                    deployment_cache_ratio: cache_ratio,
                    deployment_artifact_overflow_ratio: artifact_overflow,
                    deployment_memory_overflow_ratio: memory_overflow,
                    cheap_score,
                    cheap_quality_loss,
                    cheap_cache_error_proxy,
                    cheap_delta_gain,
                });
                index += 1;
            }
        }
    }
    Ok(candidates)
}

fn wind_pack_candidate_summary(groups: &[ModelGolfTensorGroup], spec: &RunSpec) -> (usize, f64) {
    let lqer_groups = wind_selected_lqer_group_names(groups, spec);
    let raw_bytes = groups
        .iter()
        .map(|group| {
            let selected_lqer = lqer_groups.iter().any(|name| name == &group.name);
            group
                .current_weight_bytes
                .saturating_add(group.scale_bytes)
                .saturating_add(if selected_lqer {
                    estimate_lqer_bytes(group.rows, group.cols, spec)
                } else {
                    0
                })
        })
        .sum::<usize>();
    let compressed_bytes = compressed_byte_estimate(raw_bytes, compression_factor_for_spec(spec));
    let quality_loss = groups
        .iter()
        .map(|group| {
            let selected_lqer = lqer_groups.iter().any(|name| name == &group.name);
            quality_loss_surrogate(
                group,
                group.current_bits,
                selected_lqer,
                spec.quant.lqer.rank,
            )
        })
        .sum::<f64>();
    (compressed_bytes, quality_loss)
}

fn wind_selected_lqer_group_names(groups: &[ModelGolfTensorGroup], spec: &RunSpec) -> Vec<String> {
    if !spec.quant.lqer.enabled || spec.quant.lqer.rank == 0 || spec.quant.lqer.top_k == 0 {
        return Vec::new();
    }
    let mut candidates = groups
        .iter()
        .filter_map(|group| {
            let effective_rank = effective_lqer_rank(group.rows, group.cols, spec.quant.lqer.rank);
            if effective_rank == 0 {
                return None;
            }
            let bytes = estimate_lqer_bytes(group.rows, group.cols, spec).max(1);
            let base =
                quality_loss_surrogate(group, group.current_bits, false, spec.quant.lqer.rank);
            let corrected =
                quality_loss_surrogate(group, group.current_bits, true, spec.quant.lqer.rank);
            Some((
                group.name.clone(),
                (base - corrected).max(0.0) / bytes as f64,
            ))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    candidates
        .into_iter()
        .take(spec.quant.lqer.top_k)
        .map(|(name, _)| name)
        .collect()
}

fn wind_delta_candidate_summary(spec: &RunSpec, delta_budget_bytes: usize) -> (usize, f64) {
    let candidates = delta_candidate_sets(spec);
    let mut dp = BTreeMap::new();
    dp.insert(
        0usize,
        DeltaDpState {
            gain: 0.0,
            bytes: 0,
            option_indices: Vec::new(),
        },
    );
    for options in &candidates {
        let mut next = BTreeMap::<usize, DeltaDpState>::new();
        for state in dp.values() {
            for (option_index, option) in options.iter().enumerate() {
                let next_bytes = state.bytes.saturating_add(option.bytes);
                if next_bytes > delta_budget_bytes {
                    continue;
                }
                let mut option_indices = state.option_indices.clone();
                option_indices.push(option_index);
                let candidate = DeltaDpState {
                    gain: state.gain + option.predicted_domain_gain,
                    bytes: next_bytes,
                    option_indices,
                };
                if next
                    .get(&next_bytes)
                    .as_ref()
                    .map(|old| better_delta_state(&candidate, old))
                    .unwrap_or(true)
                {
                    next.insert(next_bytes, candidate);
                }
            }
        }
        dp = prune_dominated_delta_states(next);
    }
    dp.into_values()
        .filter(|state| state.bytes <= delta_budget_bytes)
        .max_by(|a, b| {
            a.gain
                .total_cmp(&b.gain)
                .then_with(|| b.bytes.cmp(&a.bytes))
        })
        .map(|state| (state.bytes, state.gain))
        .unwrap_or((0, 0.0))
}

fn wind_experiment_row_from_candidate(
    candidate: &WindExperimentCandidate,
    cheap_rank: usize,
) -> ModelGolfWindExperimentRowReport {
    ModelGolfWindExperimentRowReport {
        candidate_name: candidate.name.clone(),
        cheap_rank,
        evaluated_cheap_rank: None,
        full_rank: None,
        quant_bits: format!(
            "matrix={} mlp={} embed={} attn_gate={}",
            candidate.quant_spec.matrix_bits,
            candidate.quant_spec.mlp_bits,
            candidate.quant_spec.embed_bits,
            candidate.quant_spec.attn_gate_bits
        ),
        lqer_enabled: candidate.quant_spec.lqer.enabled,
        lqer_rank: candidate.quant_spec.lqer.rank,
        lqer_top_k: candidate.quant_spec.lqer.top_k,
        cache_k_bits: candidate.cache_k_bits,
        cache_v_bits: candidate.cache_v_bits,
        delta_budget_bytes: candidate.delta_budget_bytes,
        estimated_artifact_bytes: candidate.estimated_artifact_bytes,
        estimated_artifact_budget_fit: candidate.estimated_artifact_budget_fit,
        estimated_cache_bytes: candidate.estimated_cache_bytes,
        estimated_memory_budget_fit: candidate.estimated_memory_budget_fit,
        estimated_delta_bytes: candidate.estimated_delta_bytes,
        cheap_score: candidate.cheap_score,
        cheap_quality_loss: candidate.cheap_quality_loss,
        cheap_cache_error_proxy: candidate.cheap_cache_error_proxy,
        cheap_delta_gain: candidate.cheap_delta_gain,
        full_evaluated: false,
        full_eval_artifact_bytes: None,
        full_eval_bpb_proxy: None,
        full_eval_score: None,
        full_eval_loss_delta_abs: None,
        full_eval_decode_tokens_per_second: None,
        full_eval_lqer_groups: Vec::new(),
        rank_error: None,
        notes: vec![
            "Cheap score uses Pack quality surrogate, CacheGolf error/memory proxy, DeltaGolf predicted gain, and deployment-scale resource pressure.".to_string(),
        ],
    }
}

fn wind_experiment_full_eval(
    model: &GptModel,
    input_ids: &[u32],
    targets: &[u32],
    pre_export_loss: f64,
    variant_fingerprint: &str,
    candidate: &WindExperimentCandidate,
) -> PgResult<WindExperimentFullEval> {
    let artifact_path = std::env::temp_dir().join(format!(
        "modelgolf_wind_experiment_{}_{}.pgrs",
        std::process::id(),
        candidate.name
    ));
    let row_fingerprint = format!("{variant_fingerprint}:wind_experiment:{}", candidate.name);
    let artifact_bytes = pg_quant::export::export_model_with_spec(
        model,
        &candidate.quant_spec,
        &row_fingerprint,
        &artifact_path,
    )?;
    let lqer_groups = pack_experiment_lqer_groups_from_artifact(&artifact_path)?;
    let mut loaded = GptModel::new(model.config.clone());
    pg_quant::export::load_artifact_with_spec(
        &artifact_path,
        &mut loaded,
        &candidate.quant_spec,
        true,
    )?;
    let start = Instant::now();
    let post_reload_loss = model_smoke_loss(&loaded, input_ids, targets)?;
    let elapsed = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&artifact_path);
    let decode_tokens_per_second = if elapsed > 0.0 {
        input_ids.len() as f64 / elapsed
    } else {
        f64::INFINITY
    };
    let bpb_proxy = post_reload_loss / std::f64::consts::LN_2;
    let pre_export_bpb_proxy = pre_export_loss / std::f64::consts::LN_2;
    let lqer_runtime_penalty = if candidate.quant_spec.lqer.enabled {
        0.001 * candidate.quant_spec.lqer.top_k as f64
    } else {
        0.0
    };
    let score = (bpb_proxy - pre_export_bpb_proxy)
        + 0.02 * candidate.deployment_artifact_ratio
        + 0.015 * candidate.deployment_cache_ratio.min(10.0)
        + 100.0 * candidate.deployment_artifact_overflow_ratio
        + 50.0 * candidate.deployment_memory_overflow_ratio
        + 0.75 * candidate.cheap_cache_error_proxy
        - 0.015 * candidate.cheap_delta_gain
        + lqer_runtime_penalty;
    Ok(WindExperimentFullEval {
        artifact_bytes,
        bpb_proxy,
        score,
        loss_delta_abs: (post_reload_loss - pre_export_loss).abs(),
        decode_tokens_per_second,
        lqer_groups,
    })
}

fn assign_wind_experiment_eval_ranks(rows: &mut [ModelGolfWindExperimentRowReport]) {
    let evaluated_indices = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.full_evaluated.then_some(index))
        .collect::<Vec<_>>();
    let mut cheap_order = evaluated_indices.clone();
    cheap_order.sort_by(|&a, &b| {
        rows[a]
            .cheap_rank
            .cmp(&rows[b].cheap_rank)
            .then_with(|| rows[a].candidate_name.cmp(&rows[b].candidate_name))
    });
    for (rank, &index) in cheap_order.iter().enumerate() {
        rows[index].evaluated_cheap_rank = Some(rank + 1);
    }
    let mut full_order = evaluated_indices;
    full_order.sort_by(|&a, &b| {
        rows[a]
            .full_eval_score
            .unwrap_or(f64::INFINITY)
            .total_cmp(&rows[b].full_eval_score.unwrap_or(f64::INFINITY))
            .then_with(|| rows[a].candidate_name.cmp(&rows[b].candidate_name))
    });
    for (rank, &index) in full_order.iter().enumerate() {
        let full_rank = rank + 1;
        rows[index].full_rank = Some(full_rank);
        if let Some(cheap_rank) = rows[index].evaluated_cheap_rank {
            rows[index].rank_error = Some(cheap_rank as isize - full_rank as isize);
        }
    }
}

fn wind_top_k_overlap_count(rows: &[ModelGolfWindExperimentRowReport], k: usize) -> usize {
    let cheap_top = rows
        .iter()
        .filter(|row| row.evaluated_cheap_rank.is_some_and(|rank| rank <= k))
        .map(|row| row.candidate_name.as_str())
        .collect::<Vec<_>>();
    rows.iter()
        .filter(|row| row.full_rank.is_some_and(|rank| rank <= k))
        .filter(|row| cheap_top.iter().any(|name| *name == row.candidate_name))
        .count()
}

fn wind_cache_error_proxy(k_bits: u8, v_bits: u8, head_dim: usize) -> f64 {
    let dim_scale = (head_dim.max(1) as f64).sqrt().recip();
    let key_term = 1.0 / ((1u32 << k_bits.min(8)) as f64 - 1.0);
    let value_term = 1.0 / ((1u32 << v_bits.min(8)) as f64 - 1.0);
    2.0 * key_term * dim_scale + value_term
}

fn wind_spearman_rank_correlation(predicted: &[f64], observed: &[f64]) -> Option<f64> {
    if predicted.len() != observed.len() || predicted.len() < 2 {
        return None;
    }
    let predicted_ranks = wind_rank_finite_values(predicted)?;
    let observed_ranks = wind_rank_finite_values(observed)?;
    wind_pearson_correlation(&predicted_ranks, &observed_ranks)
}

fn wind_rank_finite_values(values: &[f64]) -> Option<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0; values.len()];
    let mut idx = 0;
    while idx < indexed.len() {
        let start = idx;
        let value = indexed[idx].1;
        while idx + 1 < indexed.len() && (indexed[idx + 1].1 - value).abs() <= 1e-12 {
            idx += 1;
        }
        let end = idx;
        let rank = (start + 1 + end + 1) as f64 * 0.5;
        for (original_index, _) in &indexed[start..=end] {
            ranks[*original_index] = rank;
        }
        idx += 1;
    }
    Some(ranks)
}

fn wind_pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = mean_nonempty(x)?;
    let mean_y = mean_nonempty(y)?;
    let mut numerator = 0.0;
    let mut x_sq = 0.0;
    let mut y_sq = 0.0;
    for (&x_value, &y_value) in x.iter().zip(y.iter()) {
        let dx = x_value - mean_x;
        let dy = y_value - mean_y;
        numerator += dx * dy;
        x_sq += dx * dx;
        y_sq += dy * dy;
    }
    let denom = (x_sq * y_sq).sqrt();
    (denom > 0.0).then_some(numerator / denom)
}

fn mean_nonempty(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn platform_statuses() -> Vec<PlatformModuleStatus> {
    vec![
        PlatformModuleStatus {
            module: "ModelGolf Core",
            implemented_surface: "ResourceContract, ModelArtifactIR, TensorRoleIR, runtime metadata, unified JSON report",
            remaining_remote_evidence: "none for local planning; external model importers still future work",
        },
        PlatformModuleStatus {
            module: "ModelGolf Pack",
            implemented_surface: "budgeted mixed-precision DP, local equal-byte quality comparison proxy, optional measured quality-loss calibration, bounded measured pack-experiment export/reload/eval table, LQER candidate ranking/proofs, byte/quality report, and optional deterministic export/strict-reload artifact proof",
            remaining_remote_evidence: "trained artifact export, decode speed, and held-out BPB/perplexity calibration on real validation data",
        },
        PlatformModuleStatus {
            module: "CacheGolf",
            implemented_surface: "KV bit/layout/block planner, bounded cache-experiment K/V bit-grid report with resource-contract budgeted selection, residual-sketch planner, sink/recent eviction planner, long-context memory scaling rows, packed per-channel-K/per-token-V CPU cache format, selected-policy local proof, on-the-fly dequant-attention reference, and attention perturbation bound",
            remaining_remote_evidence: "fused CUDA/Metal dequant-attention kernels plus measured long-context quality/timing eval",
        },
        PlatformModuleStatus {
            module: "DeltaGolf",
            implemented_surface: "byte-constrained delta allocation, selected low-rank local proofs, static score-first legality audit, equal-byte domain proxy rows, and weighted low-rank Kronecker-curvature CPU reference across LoRA/output-style factors",
            remaining_remote_evidence: "measured full-update/curvature estimation, domain corpus training, runtime score-first ordering trace or formal competition/legal review, and real equal-byte domain BPB/perplexity",
        },
        PlatformModuleStatus {
            module: "TrainGolf",
            implemented_surface: "artifact-aware objective report, local regularizer/export-gap proof, CPU quantization-distance regularizer, CPU GradBuffers integration seam, opt-in non-CUDA train-loop scheduling, and smooth export-gap bound reference",
            remaining_remote_evidence: "proxy calibration, GPU/backend integration, and post-export quality comparisons",
        },
        PlatformModuleStatus {
            module: "Optimizer/Comm Compiler",
            implemented_surface: "sharded Parallel Muon communication planner, ring collective byte estimates, graph/shadow/overlap contract reporting, and deterministic reduce-scatter/local-update/all-gather equivalence proof for shard-separable updates",
            remaining_remote_evidence: "distributed backend parity, NCCL trace timing, overlap validation, and target-cluster communication measurements",
        },
        PlatformModuleStatus {
            module: "KernelForge",
            implemented_surface: "fusion primitive contracts, exact tiled CE memory/gradient report, CPU full-logit parity reference, report-level tiled CE local proof, and bounded kernel-experiment full-vs-tiled CE timing/memory table",
            remaining_remote_evidence: "generated CUDA/Metal kernels plus backend parity/perf tests",
        },
        PlatformModuleStatus {
            module: "Wind Tunnel",
            implemented_surface: "Pareto candidate summary tied to pack/cache/delta/kernel plans, ModelGolf wind-experiment candidate ranking with top-8 export/reload proxy validation, plus pg-local reviewer packets with train-only holdout prediction error and rank correlation",
            remaining_remote_evidence: "fresh profiler traces and external full train/eval timing validation",
        },
        PlatformModuleStatus {
            module: "ScaleGolf",
            implemented_surface: "cross-track planner for PG-Lite, ParameterGolf, DeltaGolf, LongContext, and larger-cluster ScaleGolf using the shared resource contract, model IR, artifact compiler, runtime planner, evaluator, and cost model surfaces",
            remaining_remote_evidence: "per-track measured artifact, quality, runtime, energy/cost, distributed communication, kernel, and profiler evidence before cross-scale release claims",
        },
    ]
}

fn load_modelgolf_release_evidence(path: &Path) -> PgResult<ModelGolfReleaseEvidenceFile> {
    let raw = std::fs::read_to_string(path).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to read ModelGolf release evidence {}: {err}",
            path.display()
        ))
    })?;
    let evidence: ModelGolfReleaseEvidenceFile = serde_json::from_str(&raw).map_err(|err| {
        PgError::InvalidOp(format!(
            "failed to parse ModelGolf release evidence JSON {}: {err}",
            path.display()
        ))
    })?;
    if evidence
        .kind
        .as_deref()
        .is_some_and(|kind| kind != "modelgolf_release_evidence")
    {
        return Err(PgError::InvalidOp(format!(
            "invalid ModelGolf release evidence kind in {}; expected modelgolf_release_evidence",
            path.display()
        )));
    }
    Ok(evidence)
}

fn release_status_for(present: bool, satisfied: bool) -> &'static str {
    if satisfied {
        "satisfied"
    } else if present {
        "invalid_release_evidence"
    } else {
        "missing_release_evidence"
    }
}

fn release_nonempty(value: Option<&String>) -> bool {
    value.is_some_and(|value| !value.trim().is_empty())
}

fn release_nonempty_vec(value: Option<&Vec<String>>) -> bool {
    value.is_some_and(|items| !items.is_empty() && items.iter().all(|item| !item.trim().is_empty()))
}

fn release_evidence_base_dir(release_evidence_path: Option<&Path>) -> PathBuf {
    release_evidence_path
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn release_portable_evidence_path(path: &Path, base_dir: &Path) -> String {
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let base_dir = std::fs::canonicalize(base_dir).unwrap_or_else(|_| base_dir.to_path_buf());
    let portable = path
        .strip_prefix(&base_dir)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .or_else(|| release_relative_path_between(&path, &base_dir))
        .unwrap_or(path);
    portable.display().to_string()
}

fn release_relative_path_between(path: &Path, base_dir: &Path) -> Option<PathBuf> {
    let path_components = path.components().collect::<Vec<_>>();
    let base_components = base_dir.components().collect::<Vec<_>>();
    if path_components.is_empty() || base_components.is_empty() {
        return None;
    }

    let common_len = path_components
        .iter()
        .zip(base_components.iter())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count();
    if common_len == 0 {
        return None;
    }

    let mut relative = PathBuf::new();
    for component in &base_components[common_len..] {
        match component {
            std::path::Component::Normal(_) => relative.push(".."),
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    for component in &path_components[common_len..] {
        match component {
            std::path::Component::Normal(part) => relative.push(Path::new(part)),
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    if relative.as_os_str().is_empty() {
        relative.push(".");
    }
    Some(relative)
}

fn release_resolve_evidence_path(base_dir: &Path, raw_path: &str) -> Option<PathBuf> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    Some(if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    })
}

fn release_sha256_string_valid(value: &str) -> bool {
    let Some(hex) = value.trim().strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

fn validate_modelgolf_source_reports(
    evidence: Option<&ModelGolfReleaseEvidenceFile>,
    spec_name: &str,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    release_evidence_path: Option<&Path>,
) -> ModelGolfSourceReportValidation {
    let mut claims_by_pillar = BTreeMap::new();
    let Some(evidence) = evidence else {
        return ModelGolfSourceReportValidation {
            valid: false,
            details: "release evidence file not supplied".to_string(),
            claims_by_pillar,
        };
    };
    let Some(source_reports) = evidence.source_reports.as_ref() else {
        return ModelGolfSourceReportValidation {
            valid: false,
            details: "source_reports missing".to_string(),
            claims_by_pillar,
        };
    };
    if source_reports.is_empty() {
        return ModelGolfSourceReportValidation {
            valid: false,
            details: "source_reports empty".to_string(),
            claims_by_pillar,
        };
    }

    let base_dir = release_evidence_base_dir(release_evidence_path);
    let required_pillars = modelgolf_required_release_pillars();
    let mut details = Vec::new();
    let mut valid = true;
    for source in source_reports {
        let pillar = source.pillar.trim();
        let kind = source.kind.trim();
        if pillar.is_empty() {
            valid = false;
            details.push("source report has empty pillar".to_string());
            continue;
        }
        if claims_by_pillar.contains_key(pillar) {
            valid = false;
            details.push(format!("duplicate source report pillar={pillar}"));
            continue;
        }
        if kind != "modelgolf_release_source_report" {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} has kind={kind:?}, expected modelgolf_release_source_report"
            ));
            continue;
        }
        if !release_sha256_string_valid(&source.sha256) {
            valid = false;
            details.push(format!("source report pillar={pillar} has invalid sha256"));
            continue;
        }
        let Some(path) = release_resolve_evidence_path(&base_dir, &source.path) else {
            valid = false;
            details.push(format!("source report pillar={pillar} has empty path"));
            continue;
        };
        let actual_sha256 = match modelgolf_sha256_file(&path) {
            Ok(actual) => format!("sha256:{actual}"),
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} failed to read {}: {err}",
                    path.display()
                ));
                continue;
            }
        };
        if actual_sha256 != source.sha256 {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} sha256 mismatch for {}",
                path.display()
            ));
            continue;
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} failed to read JSON {}: {err}",
                    path.display()
                ));
                continue;
            }
        };
        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(value) => value,
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} invalid JSON {}: {err}",
                    path.display()
                ));
                continue;
            }
        };
        let report_kind = value.get("kind").and_then(serde_json::Value::as_str);
        let report_pillar = value.get("pillar").and_then(serde_json::Value::as_str);
        let report_spec_name = value.get("spec_name").and_then(serde_json::Value::as_str);
        let report_spec_fingerprint = value
            .get("spec_fingerprint")
            .and_then(serde_json::Value::as_str);
        let report_hardware = value.get("hardware").and_then(serde_json::Value::as_str);
        let report_runtime = value.get("runtime").and_then(serde_json::Value::as_str);
        if report_kind != Some(kind)
            || report_pillar != Some(pillar)
            || report_spec_name != Some(spec_name)
            || report_spec_fingerprint != Some(spec_fingerprint)
            || report_hardware != Some(resource_contract.hardware.as_str())
            || report_runtime != Some(resource_contract.runtime.as_str())
        {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} does not match kind/spec/runtime binding"
            ));
            continue;
        }
        let Some(claims) = value.get("claims").and_then(serde_json::Value::as_object) else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} missing claims object"
            ));
            continue;
        };
        let Some(raw_evidence) = claims
            .get("raw_evidence")
            .and_then(serde_json::Value::as_object)
        else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} missing raw_evidence object"
            ));
            continue;
        };
        let raw_base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let Some(raw_path_value) = raw_evidence.get("path").and_then(serde_json::Value::as_str)
        else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence missing path"
            ));
            continue;
        };
        let Some(raw_sha256) = raw_evidence
            .get("sha256")
            .and_then(serde_json::Value::as_str)
        else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence missing sha256"
            ));
            continue;
        };
        if !release_sha256_string_valid(raw_sha256) {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence has invalid sha256"
            ));
            continue;
        }
        let Some(raw_path) = release_resolve_evidence_path(raw_base_dir, raw_path_value) else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence has empty path"
            ));
            continue;
        };
        let raw_metadata = match std::fs::metadata(&raw_path) {
            Ok(metadata) => metadata,
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} raw_evidence failed to stat {}: {err}",
                    raw_path.display()
                ));
                continue;
            }
        };
        if !raw_metadata.is_file() || raw_metadata.len() == 0 {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence {} is not a non-empty file",
                raw_path.display()
            ));
            continue;
        }
        if raw_evidence
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            != Some(raw_metadata.len())
        {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence byte count mismatch for {}",
                raw_path.display()
            ));
            continue;
        }
        let actual_raw_sha256 = match modelgolf_sha256_file(&raw_path) {
            Ok(actual) => format!("sha256:{actual}"),
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} raw_evidence failed to hash {}: {err}",
                    raw_path.display()
                ));
                continue;
            }
        };
        if actual_raw_sha256 != raw_sha256 {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence sha256 mismatch for {}",
                raw_path.display()
            ));
            continue;
        }
        let raw_body = match std::fs::read_to_string(&raw_path) {
            Ok(raw_body) => raw_body,
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} raw_evidence failed to read JSON {}: {err}",
                    raw_path.display()
                ));
                continue;
            }
        };
        let raw_value: serde_json::Value = match serde_json::from_str(&raw_body) {
            Ok(raw_value) => raw_value,
            Err(err) => {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} raw_evidence invalid JSON {}: {err}",
                    raw_path.display()
                ));
                continue;
            }
        };
        if raw_value.get("kind").and_then(serde_json::Value::as_str)
            != Some("modelgolf_raw_evidence")
            || raw_value.get("pillar").and_then(serde_json::Value::as_str) != Some(pillar)
        {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence does not match modelgolf_raw_evidence kind/pillar"
            ));
            continue;
        }
        let Some(raw_claims) = raw_value
            .get("claims")
            .and_then(serde_json::Value::as_object)
        else {
            valid = false;
            details.push(format!(
                "source report pillar={pillar} raw_evidence missing claims object"
            ));
            continue;
        };
        for key in modelgolf_raw_claim_keys_for_pillar(pillar) {
            let Some(source_claim) = claims.get(*key) else {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} missing source claim required by raw_evidence: {key}"
                ));
                continue;
            };
            if !raw_evidence_claim_value_matches(raw_claims.get(*key), source_claim) {
                valid = false;
                details.push(format!(
                    "source report pillar={pillar} raw_evidence claim mismatch for {key}"
                ));
                continue;
            }
        }
        claims_by_pillar.insert(
            pillar.to_string(),
            serde_json::Value::Object(claims.clone()),
        );
    }

    for pillar in required_pillars {
        if !claims_by_pillar.contains_key(pillar) {
            valid = false;
            details.push(format!("source report pillar={pillar} missing"));
        }
    }
    if details.is_empty() {
        details.push(format!(
            "validated {} source reports with SHA-256 and spec/runtime binding",
            claims_by_pillar.len()
        ));
    }

    ModelGolfSourceReportValidation {
        valid,
        details: details.join("; "),
        claims_by_pillar,
    }
}

fn release_source_claim<'a>(
    source_reports: &'a ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
) -> Option<&'a serde_json::Value> {
    source_reports
        .claims_by_pillar
        .get(pillar)
        .and_then(|claims| claims.get(key))
}

fn release_source_claim_string_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<&String>,
) -> bool {
    let Some(expected) = expected else {
        return false;
    };
    release_source_claim(source_reports, pillar, key).and_then(serde_json::Value::as_str)
        == Some(expected.as_str())
}

fn release_source_claim_bool_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<bool>,
) -> bool {
    release_source_claim(source_reports, pillar, key).and_then(serde_json::Value::as_bool)
        == expected
}

fn release_source_claim_usize_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<usize>,
) -> bool {
    release_source_claim(source_reports, pillar, key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        == expected
}

fn release_source_claim_u8_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<u8>,
) -> bool {
    release_source_claim(source_reports, pillar, key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u8::try_from(value).ok())
        == expected
}

fn release_source_claim_f64_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<f64>,
) -> bool {
    let Some(expected) = expected else {
        return false;
    };
    release_source_claim(source_reports, pillar, key)
        .and_then(serde_json::Value::as_f64)
        .is_some_and(|value| value.is_finite() && (value - expected).abs() <= 1e-9)
}

fn release_source_claim_string_vec_matches(
    source_reports: &ModelGolfSourceReportValidation,
    pillar: &str,
    key: &str,
    expected: Option<&Vec<String>>,
) -> bool {
    let Some(expected) = expected else {
        return false;
    };
    let Some(values) =
        release_source_claim(source_reports, pillar, key).and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if values.len() != expected.len() {
        return false;
    }
    expected.iter().all(|expected_value| {
        values
            .iter()
            .any(|value| value.as_str() == Some(expected_value.as_str()))
    })
}

fn release_finite_positive(value: Option<f64>) -> bool {
    value.is_some_and(|value| value.is_finite() && value > 0.0)
}

fn release_finite_at_least(value: Option<f64>, min_value: f64) -> bool {
    value.is_some_and(|value| value.is_finite() && value >= min_value)
}

fn release_finite_at_most(value: Option<f64>, max_value: f64) -> bool {
    value.is_some_and(|value| value.is_finite() && value <= max_value)
}

fn release_quality_within_budget(
    measured_bpb: Option<f64>,
    baseline_bpb: Option<f64>,
    relative_bpb_increase_pct: Option<f64>,
    quality_budget_pct: Option<f64>,
) -> bool {
    let Some(measured_bpb) = measured_bpb else {
        return false;
    };
    if !measured_bpb.is_finite() || measured_bpb <= 0.0 {
        return false;
    }
    let Some(quality_budget_pct) = quality_budget_pct else {
        return true;
    };
    if !quality_budget_pct.is_finite() {
        return false;
    }
    let Some(baseline_bpb) = baseline_bpb else {
        return false;
    };
    if !baseline_bpb.is_finite() || baseline_bpb <= 0.0 {
        return false;
    }
    let computed_relative = 100.0 * (measured_bpb - baseline_bpb) / baseline_bpb;
    if let Some(relative_bpb_increase_pct) = relative_bpb_increase_pct {
        return relative_bpb_increase_pct.is_finite()
            && (relative_bpb_increase_pct - computed_relative).abs() <= 1e-6
            && computed_relative <= quality_budget_pct;
    }
    computed_relative <= quality_budget_pct
}

fn release_quality_delta_pct_within_budget(
    delta_pct: Option<f64>,
    quality_budget_pct: Option<f64>,
) -> bool {
    let Some(delta_pct) = delta_pct else {
        return false;
    };
    if !delta_pct.is_finite() {
        return false;
    }
    quality_budget_pct
        .filter(|quality_budget_pct| quality_budget_pct.is_finite())
        .is_none_or(|quality_budget_pct| delta_pct <= quality_budget_pct)
}

const MODELGOLF_SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
struct ModelGolfSha256State {
    h: [u32; 8],
    len: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl ModelGolfSha256State {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            len: 0,
            buf: [0; 64],
            buf_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len = self.len.wrapping_add(input.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while input.len() >= 64 {
            self.compress(&input[..64]);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buf[..input.len()].copy_from_slice(input);
            self.buf_len = input.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.h) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(MODELGOLF_SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = (*slot).wrapping_add(value);
        }
    }
}

fn modelgolf_hex_digest(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn modelgolf_sha256_file(path: &Path) -> PgResult<String> {
    let mut file = std::fs::File::open(path)?;
    let mut state = ModelGolfSha256State::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        state.update(&buf[..n]);
    }
    Ok(modelgolf_hex_digest(state.finalize()))
}

fn release_evidence_global_binding_ok(
    evidence: Option<&ModelGolfReleaseEvidenceFile>,
    spec_name: &str,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    source_reports: &ModelGolfSourceReportValidation,
) -> bool {
    let Some(evidence) = evidence else {
        return false;
    };
    evidence.kind.as_deref() == Some("modelgolf_release_evidence")
        && evidence.spec_name.as_deref() == Some(spec_name)
        && evidence.spec_fingerprint.as_deref() == Some(spec_fingerprint)
        && evidence.hardware.as_deref() == Some(resource_contract.hardware.as_str())
        && evidence.runtime.as_deref() == Some(resource_contract.runtime.as_str())
        && release_nonempty(evidence.evidence_id.as_ref())
        && release_nonempty(evidence.generated_at.as_ref())
        && source_reports.valid
}

fn release_artifact_sha256_matches(
    base_dir: &Path,
    path: Option<&String>,
    expected_sha256: Option<&String>,
) -> bool {
    let (Some(path), Some(expected_sha256)) = (path, expected_sha256) else {
        return false;
    };
    if !release_sha256_string_valid(expected_sha256) {
        return false;
    }
    let Some(path) = release_resolve_evidence_path(base_dir, path) else {
        return false;
    };
    modelgolf_sha256_file(&path)
        .map(|actual| format!("sha256:{actual}") == *expected_sha256)
        .unwrap_or(false)
}

fn release_selected_delta_names_match(
    evidence_names: Option<&Vec<String>>,
    selected_deltas: &[DeltaOptionReport],
) -> bool {
    let Some(evidence_names) = evidence_names else {
        return false;
    };
    if evidence_names.len() != selected_deltas.len() {
        return false;
    }
    selected_deltas.iter().all(|delta| {
        evidence_names
            .iter()
            .any(|name| name.as_str() == delta.name.as_str())
    })
}

#[allow(clippy::too_many_arguments)]
fn modelgolf_release_readiness(
    spec_name: &str,
    spec_fingerprint: &str,
    resource_contract: &ResourceContractReport,
    pack: &PackPlannerReport,
    cache: &CachePlannerReport,
    delta: &DeltaPlannerReport,
    train: &TrainPlannerReport,
    optimizer_comm: &OptimizerCommReport,
    kernel_forge: &KernelForgeReport,
    wind_tunnel: &ModelGolfWindReport,
    release_evidence_path: Option<&Path>,
    release_evidence: Option<&ModelGolfReleaseEvidenceFile>,
) -> ModelGolfReleaseReadinessReport {
    let local_planner_ready = pack.feasible
        && cache.feasible
        && cache.selected_policy_proof.bound_covers_observed_error
        && cache.selected_policy_proof.dequant_attention_parity_ok
        && delta.score_first_legality_audit.pass
        && train.local_proof.finite
        && train.local_proof.export_gap_bound_covers_observed
        && optimizer_comm.local_proof.finite
        && optimizer_comm.local_proof.exact_equivalence
        && kernel_forge.exact_tiled_ce.local_proof.parity_ok
        && kernel_forge.exact_tiled_ce.local_proof.finite
        && wind_tunnel.estimate_only;
    let release_evidence_base_dir = release_evidence_base_dir(release_evidence_path);
    let source_report_validation = validate_modelgolf_source_reports(
        release_evidence,
        spec_name,
        spec_fingerprint,
        resource_contract,
        release_evidence_path,
    );
    let release_evidence_binding_ok = release_evidence_global_binding_ok(
        release_evidence,
        spec_name,
        spec_fingerprint,
        resource_contract,
        &source_report_validation,
    );
    let selected_lqer_group_names = pack
        .lqer_proofs
        .iter()
        .map(|proof| proof.group.clone())
        .collect::<Vec<_>>();
    let resource_cost_evidence =
        release_evidence.and_then(|evidence| evidence.resource_cost.as_ref());
    let resource_cost_release_ok = resource_cost_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "resource_cost",
                    "measurement_run_id",
                    evidence.measurement_run_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "resource_cost",
                    "power_meter_id",
                    evidence.power_meter_id.as_ref(),
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "wall_time_seconds",
                    evidence.wall_time_seconds,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "average_power_watts",
                    evidence.average_power_watts,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "energy_joules",
                    evidence.energy_joules,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "resource_cost",
                    "telemetry_validated",
                    evidence.telemetry_validated,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "wall_time_budget_seconds",
                    evidence.wall_time_budget_seconds,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "energy_budget_joules",
                    evidence.energy_budget_joules,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "resource_cost",
                    "wall_time_budget_fit",
                    evidence.wall_time_budget_fit,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "resource_cost",
                    "energy_budget_fit",
                    evidence.energy_budget_fit,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "resource_cost",
                    "energy_consistency_error_pct",
                    evidence.energy_consistency_error_pct,
                )
                && release_nonempty(evidence.measurement_run_id.as_ref())
                && release_nonempty(evidence.power_meter_id.as_ref())
                && release_finite_positive(evidence.wall_time_seconds)
                && release_finite_positive(evidence.average_power_watts)
                && release_finite_positive(evidence.energy_joules)
                && evidence.telemetry_validated == Some(true)
                && evidence.wall_time_budget_seconds
                    == Some(resource_contract.training_time_budget_seconds)
                && evidence.energy_budget_joules
                    == Some(resource_contract.train_energy_budget_joules_proxy)
                && evidence.wall_time_budget_fit == Some(true)
                && evidence.energy_budget_fit == Some(true)
                && evidence.wall_time_seconds.is_some_and(|seconds| {
                    seconds <= resource_contract.training_time_budget_seconds
                })
                && evidence.energy_joules.is_some_and(|joules| {
                    joules <= resource_contract.train_energy_budget_joules_proxy
                })
                && release_finite_at_most(evidence.energy_consistency_error_pct, 5.0)
        })
        .unwrap_or(false);
    let pack_evidence = release_evidence.and_then(|evidence| evidence.pack.as_ref());
    let pack_release_ok = pack_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "pack",
                    "artifact_sha256",
                    evidence.artifact_sha256.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "pack",
                    "validation_dataset_id",
                    evidence.validation_dataset_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "pack",
                    "validation_command",
                    evidence.validation_command.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "pack",
                    "artifact_bytes",
                    evidence.artifact_bytes,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "pack",
                    "strict_reload_pass",
                    evidence.strict_reload_pass,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "pack",
                    "heldout_bpb",
                    evidence.heldout_bpb,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "pack",
                    "baseline_bpb",
                    evidence.baseline_bpb,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "pack",
                    "relative_bpb_increase_pct",
                    evidence.relative_bpb_increase_pct,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "pack",
                    "decode_tokens_per_second",
                    evidence.decode_tokens_per_second,
                )
                && evidence.strict_reload_pass == Some(true)
                && release_nonempty(evidence.validation_dataset_id.as_ref())
                && release_nonempty(evidence.validation_command.as_ref())
                && release_artifact_sha256_matches(
                    &release_evidence_base_dir,
                    evidence.artifact_path.as_ref(),
                    evidence.artifact_sha256.as_ref(),
                )
                && evidence
                    .artifact_bytes
                    .is_some_and(|bytes| bytes > 0 && bytes <= pack.target_artifact_bytes)
                && release_quality_within_budget(
                    evidence.heldout_bpb,
                    evidence.baseline_bpb,
                    evidence.relative_bpb_increase_pct,
                    resource_contract.quality_budget_ppl_pct,
                )
                && release_finite_positive(evidence.decode_tokens_per_second)
        })
        .unwrap_or(false);
    let lqer_evidence = release_evidence.and_then(|evidence| evidence.lqer.as_ref());
    let lqer_release_ok = lqer_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "lqer",
                    "calibration_dataset_id",
                    evidence.calibration_dataset_id.as_ref(),
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "lqer",
                    "production_svd_validated",
                    evidence.production_svd_validated,
                )
                && release_source_claim_string_vec_matches(
                    &source_report_validation,
                    "lqer",
                    "selected_lqer_group_names",
                    evidence.selected_lqer_group_names.as_ref(),
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "lqer",
                    "calibrated_tensor_sensitivity",
                    evidence.calibrated_tensor_sensitivity,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "lqer",
                    "equal_byte_bpb_delta",
                    evidence.equal_byte_bpb_delta,
                )
                && release_nonempty(evidence.calibration_dataset_id.as_ref())
                && !selected_lqer_group_names.is_empty()
                && evidence
                    .selected_lqer_group_names
                    .as_ref()
                    .is_some_and(|names| {
                        names.len() == selected_lqer_group_names.len()
                            && selected_lqer_group_names.iter().all(|selected| {
                                names.iter().any(|name| name.as_str() == selected.as_str())
                            })
                    })
                && evidence.production_svd_validated == Some(true)
                && evidence.calibrated_tensor_sensitivity == Some(true)
                && release_finite_at_most(evidence.equal_byte_bpb_delta, 0.0)
        })
        .unwrap_or(false);
    let cache_evidence = release_evidence.and_then(|evidence| evidence.cache.as_ref());
    let cache_release_ok = cache_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "cache",
                    "backend_id",
                    evidence.backend_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "cache",
                    "kernel_id",
                    evidence.kernel_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "cache",
                    "long_context_dataset_id",
                    evidence.long_context_dataset_id.as_ref(),
                )
                && release_source_claim_u8_matches(
                    &source_report_validation,
                    "cache",
                    "k_bits",
                    evidence.k_bits,
                )
                && release_source_claim_u8_matches(
                    &source_report_validation,
                    "cache",
                    "v_bits",
                    evidence.v_bits,
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "cache",
                    "block_size_tokens",
                    evidence.block_size_tokens,
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "cache",
                    "layout",
                    evidence.layout.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "cache",
                    "context_tokens",
                    evidence.context_tokens,
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "cache",
                    "batch_sequences",
                    evidence.batch_sequences,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "cache",
                    "fused_runtime",
                    evidence.fused_runtime,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "cache",
                    "parity_pass",
                    evidence.parity_pass,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "cache",
                    "memory_budget_fit",
                    evidence.memory_budget_fit,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "cache",
                    "long_context_bpb_delta_pct",
                    evidence.long_context_bpb_delta_pct,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "cache",
                    "speedup_x",
                    evidence.speedup_x,
                )
                && release_nonempty(evidence.backend_id.as_ref())
                && release_nonempty(evidence.kernel_id.as_ref())
                && release_nonempty(evidence.long_context_dataset_id.as_ref())
                && evidence.k_bits == Some(cache.selected.k_bits)
                && evidence.v_bits == Some(cache.selected.v_bits)
                && evidence.block_size_tokens == Some(cache.selected.block_size_tokens)
                && evidence.layout.as_deref() == Some(cache.selected.layout)
                && evidence.context_tokens == Some(cache.context_tokens)
                && evidence.batch_sequences == Some(cache.batch_sequences)
                && evidence.fused_runtime == Some(true)
                && evidence.parity_pass == Some(true)
                && evidence.memory_budget_fit == Some(true)
                && release_quality_delta_pct_within_budget(
                    evidence.long_context_bpb_delta_pct,
                    resource_contract.quality_budget_ppl_pct,
                )
                && release_finite_at_least(evidence.speedup_x, 1.0)
        })
        .unwrap_or(false);
    let delta_evidence = release_evidence.and_then(|evidence| evidence.delta.as_ref());
    let delta_release_ok = delta_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "delta",
                    "domain_dataset_id",
                    evidence.domain_dataset_id.as_ref(),
                )
                && release_source_claim_string_vec_matches(
                    &source_report_validation,
                    "delta",
                    "selected_delta_names",
                    evidence.selected_delta_names.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "delta",
                    "trained_delta_bytes",
                    evidence.trained_delta_bytes,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "delta",
                    "legality_pass",
                    evidence.legality_pass,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "delta",
                    "score_first_trace_or_review",
                    evidence.score_first_trace_or_review,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "delta",
                    "equal_byte_domain_bpb_delta",
                    evidence.equal_byte_domain_bpb_delta,
                )
                && release_nonempty(evidence.domain_dataset_id.as_ref())
                && release_selected_delta_names_match(
                    evidence.selected_delta_names.as_ref(),
                    &delta.selected_deltas,
                )
                && evidence.legality_pass == Some(true)
                && evidence.score_first_trace_or_review == Some(true)
                && delta.selected_bytes > 0
                && evidence
                    .trained_delta_bytes
                    .is_some_and(|bytes| bytes > 0 && bytes <= delta.selected_bytes)
                && release_finite_at_most(evidence.equal_byte_domain_bpb_delta, 0.0)
        })
        .unwrap_or(false);
    let train_evidence = release_evidence.and_then(|evidence| evidence.train.as_ref());
    let train_release_ok = train_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "train",
                    "training_run_id",
                    evidence.training_run_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "train",
                    "backend_id",
                    evidence.backend_id.as_ref(),
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "train",
                    "gpu_backend_integrated",
                    evidence.gpu_backend_integrated,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "train",
                    "proxy_calibrated",
                    evidence.proxy_calibrated,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "train",
                    "post_export_bpb_delta_vs_posthoc",
                    evidence.post_export_bpb_delta_vs_posthoc,
                )
                && release_nonempty(evidence.training_run_id.as_ref())
                && release_nonempty(evidence.backend_id.as_ref())
                && evidence.gpu_backend_integrated == Some(true)
                && evidence.proxy_calibrated == Some(true)
                && release_finite_at_most(evidence.post_export_bpb_delta_vs_posthoc, 0.0)
        })
        .unwrap_or(false);
    let optimizer_comm_evidence =
        release_evidence.and_then(|evidence| evidence.optimizer_comm.as_ref());
    let optimizer_comm_release_ok = optimizer_comm_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "distributed_backend_id",
                    evidence.distributed_backend_id.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "world_size",
                    evidence.world_size,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "optimizer_sharded",
                    evidence.optimizer_sharded,
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "nccl_overlap_mode",
                    evidence.nccl_overlap_mode.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "sharded_total_wire_bytes_per_rank",
                    evidence.sharded_total_wire_bytes_per_rank,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "owned_optimizer_state_reduction_x",
                    evidence.owned_optimizer_state_reduction_x,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "reduce_scatter_parity_pass",
                    evidence.reduce_scatter_parity_pass,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "all_gather_parity_pass",
                    evidence.all_gather_parity_pass,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "optimizer_update_parity_pass",
                    evidence.optimizer_update_parity_pass,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "nccl_trace_validated",
                    evidence.nccl_trace_validated,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "overlap_validated",
                    evidence.overlap_validated,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "measured_comm_time_ms",
                    evidence.measured_comm_time_ms,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "measured_step_time_ms",
                    evidence.measured_step_time_ms,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "optimizer_comm",
                    "communication_speedup_x",
                    evidence.communication_speedup_x,
                )
                && release_nonempty(evidence.distributed_backend_id.as_ref())
                && evidence.world_size == Some(optimizer_comm.world_size)
                && evidence.optimizer_sharded == Some(optimizer_comm.optimizer_sharded)
                && evidence.nccl_overlap_mode.as_deref()
                    == Some(optimizer_comm.nccl_overlap_mode.as_str())
                && evidence.sharded_total_wire_bytes_per_rank
                    == Some(optimizer_comm.sharded_total_wire_bytes_per_rank)
                && evidence.owned_optimizer_state_reduction_x
                    == Some(optimizer_comm.owned_optimizer_state_reduction_x)
                && evidence.reduce_scatter_parity_pass == Some(true)
                && evidence.all_gather_parity_pass == Some(true)
                && evidence.optimizer_update_parity_pass == Some(true)
                && evidence.nccl_trace_validated == Some(true)
                && evidence.overlap_validated == Some(true)
                && release_finite_positive(evidence.measured_comm_time_ms)
                && release_finite_positive(evidence.measured_step_time_ms)
                && evidence
                    .measured_comm_time_ms
                    .zip(evidence.measured_step_time_ms)
                    .is_some_and(|(comm_ms, step_ms)| comm_ms <= step_ms)
                && release_finite_at_least(evidence.communication_speedup_x, 1.0)
        })
        .unwrap_or(false);
    let kernel_evidence = release_evidence.and_then(|evidence| evidence.kernel_forge.as_ref());
    let kernel_release_ok = kernel_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "backend_id",
                    evidence.backend_id.as_ref(),
                )
                && release_source_claim_string_vec_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "generated_kernel_ids",
                    evidence.generated_kernel_ids.as_ref(),
                )
                && release_source_claim_usize_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "tile_t",
                    evidence.tile_t,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "generated_kernels",
                    evidence.generated_kernels,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "parity_pass",
                    evidence.parity_pass,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "speedup_x",
                    evidence.speedup_x,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "kernel_forge",
                    "memory_reduction_x",
                    evidence.memory_reduction_x,
                )
                && release_nonempty(evidence.backend_id.as_ref())
                && release_nonempty_vec(evidence.generated_kernel_ids.as_ref())
                && evidence.tile_t == Some(kernel_forge.exact_tiled_ce.tile_t)
                && evidence.generated_kernels == Some(true)
                && evidence.parity_pass == Some(true)
                && release_finite_at_least(evidence.speedup_x, 1.0)
                && release_finite_at_least(evidence.memory_reduction_x, 1.0)
        })
        .unwrap_or(false);
    let wind_evidence = release_evidence.and_then(|evidence| evidence.wind_tunnel.as_ref());
    let wind_release_ok = wind_evidence
        .map(|evidence| {
            release_evidence_binding_ok
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "trace_corpus_id",
                    evidence.trace_corpus_id.as_ref(),
                )
                && release_source_claim_string_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "calibration_report_id",
                    evidence.calibration_report_id.as_ref(),
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "fresh_profiler_traces",
                    evidence.fresh_profiler_traces,
                )
                && release_source_claim_bool_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "external_timing_validated",
                    evidence.external_timing_validated,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "holdout_spearman",
                    evidence.holdout_spearman,
                )
                && release_source_claim_f64_matches(
                    &source_report_validation,
                    "wind_tunnel",
                    "mean_abs_pct_error",
                    evidence.mean_abs_pct_error,
                )
                && release_nonempty(evidence.trace_corpus_id.as_ref())
                && release_nonempty(evidence.calibration_report_id.as_ref())
                && evidence.fresh_profiler_traces == Some(true)
                && evidence.external_timing_validated == Some(true)
                && release_finite_at_least(evidence.holdout_spearman, 0.50)
                && release_finite_at_most(evidence.mean_abs_pct_error, 25.0)
        })
        .unwrap_or(false);
    let mut checklist = vec![ModelGolfReleaseRequirementReport {
        pillar: "ModelGolf Core",
        requirement: "local planner, proof, and report gates pass for the requested contract",
        status: if local_planner_ready {
            "satisfied"
        } else {
            "failed_local_gate"
        },
        blocking: !local_planner_ready,
        current_evidence: format!(
            "pack_feasible={}, cache_feasible={}, cache_bound_covers={}, cache_dequant_parity={}, delta_static_legality={}, train_export_gap_bound={}, optimizer_comm_equivalence={}, kernel_ce_parity={}, wind_estimate_only={}",
            pack.feasible,
            cache.feasible,
            cache.selected_policy_proof.bound_covers_observed_error,
            cache.selected_policy_proof.dequant_attention_parity_ok,
            delta.score_first_legality_audit.pass,
            train.local_proof.export_gap_bound_covers_observed,
            optimizer_comm.local_proof.exact_equivalence,
            kernel_forge.exact_tiled_ce.local_proof.parity_ok,
            wind_tunnel.estimate_only
        ),
        evidence_needed: "all local deterministic planner/proof gates must pass before release evidence can be interpreted",
    }, ModelGolfReleaseRequirementReport {
        pillar: "Release Evidence Binding",
        requirement: "measured evidence must match spec fingerprint, hardware/runtime labels, run IDs, and source reports",
        status: if release_evidence_binding_ok {
            "satisfied"
        } else if release_evidence.is_some() {
            "invalid_release_evidence"
        } else {
            "missing_release_evidence"
        },
        blocking: !release_evidence_binding_ok,
        current_evidence: release_evidence.map(|evidence| {
            format!(
                "spec_name={:?}, expected_spec_name={}, spec_fingerprint={:?}, expected_spec_fingerprint={}, hardware={:?}, expected_hardware={}, runtime={:?}, expected_runtime={}, evidence_id_present={}, generated_at_present={}, source_report_validation={}",
                evidence.spec_name,
                spec_name,
                evidence.spec_fingerprint,
                spec_fingerprint,
                evidence.hardware,
                resource_contract.hardware,
                evidence.runtime,
                resource_contract.runtime,
                release_nonempty(evidence.evidence_id.as_ref()),
                release_nonempty(evidence.generated_at.as_ref()),
                source_report_validation.details
            )
        }).unwrap_or_else(|| "release evidence file not supplied".to_string()),
        evidence_needed: "provide modelgolf_release_evidence with matching spec_name, spec_fingerprint, hardware, runtime, evidence_id, generated_at, and SHA-256 verified source_reports",
    }];
    checklist.extend([
        ModelGolfReleaseRequirementReport {
            pillar: "Resource Cost Evidence",
            requirement: "measured wall-time, average-power, and energy telemetry must fit the active resource contract",
            status: release_status_for(resource_cost_evidence.is_some(), resource_cost_release_ok),
            blocking: !resource_cost_release_ok,
            current_evidence: resource_cost_evidence.map(|evidence| {
                format!(
                    "measurement_run_id_present={}, power_meter_id_present={}, wall_time_seconds={:?}, wall_time_budget_seconds={}, average_power_watts={:?}, energy_joules={:?}, energy_budget_joules={}, telemetry_validated={:?}, wall_time_budget_fit={:?}, energy_budget_fit={:?}, energy_consistency_error_pct={:?}",
                    release_nonempty(evidence.measurement_run_id.as_ref()),
                    release_nonempty(evidence.power_meter_id.as_ref()),
                    evidence.wall_time_seconds,
                    resource_contract.training_time_budget_seconds,
                    evidence.average_power_watts,
                    evidence.energy_joules,
                    resource_contract.train_energy_budget_joules_proxy,
                    evidence.telemetry_validated,
                    evidence.wall_time_budget_fit,
                    evidence.energy_budget_fit,
                    evidence.energy_consistency_error_pct
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.resource_cost missing; local_training_time_budget_seconds={}, local_energy_budget_joules_proxy={}, nominal_power_watts_proxy={}",
                    resource_contract.training_time_budget_seconds,
                    resource_contract.train_energy_budget_joules_proxy,
                    resource_contract.nominal_power_watts_proxy
                )
            }),
            evidence_needed: "run the target training/eval workload under measured power telemetry, record wall time, average power, total joules, meter identity, and validate budget fit",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "ModelGolf Pack",
            requirement: "final trained artifact byte proof, held-out BPB/perplexity, and decode-speed evidence",
            status: release_status_for(pack_evidence.is_some(), pack_release_ok),
            blocking: !pack_release_ok,
            current_evidence: pack_evidence.map(|evidence| {
                format!(
                    "artifact_path={:?}, artifact_sha256_present={}, validation_dataset_id_present={}, validation_command_present={}, artifact_bytes={:?}, artifact_budget_bytes={}, strict_reload_pass={:?}, heldout_bpb={:?}, baseline_bpb={:?}, relative_bpb_increase_pct={:?}, decode_tokens_per_second={:?}",
                    evidence.artifact_path,
                    release_nonempty(evidence.artifact_sha256.as_ref()),
                    release_nonempty(evidence.validation_dataset_id.as_ref()),
                    release_nonempty(evidence.validation_command.as_ref()),
                    evidence.artifact_bytes,
                    pack.target_artifact_bytes,
                    evidence.strict_reload_pass,
                    evidence.heldout_bpb,
                    evidence.baseline_bpb,
                    evidence.relative_bpb_increase_pct,
                    evidence.decode_tokens_per_second
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.pack missing; local_pack_selected_bytes={}, quality_comparison_rows={}, lqer_proofs={}, proof_boundary=local_format_reload_or_proxy_only",
                    pack.selected_bytes,
                    pack.quality_comparison.rows.len(),
                    pack.lqer_proofs.len()
                )
            }),
            evidence_needed: "export the final trained/pretrained artifact, strict-reload it, measure actual bytes, run held-out validation BPB/perplexity, and record decode-speed timing",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "LQER++",
            requirement: "production-scale residual factorization and calibrated tensor sensitivity",
            status: release_status_for(lqer_evidence.is_some(), lqer_release_ok),
            blocking: !lqer_release_ok,
            current_evidence: lqer_evidence.map(|evidence| {
                format!(
                    "production_svd_validated={:?}, calibrated_tensor_sensitivity={:?}, equal_byte_bpb_delta={:?}, selected_lqer_group_count={}, expected_selected_lqer_group_count={}",
                    evidence.production_svd_validated,
                    evidence.calibrated_tensor_sensitivity,
                    evidence.equal_byte_bpb_delta,
                    evidence.selected_lqer_group_names.as_ref().map(Vec::len).unwrap_or(0),
                    selected_lqer_group_names.len()
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.lqer missing; candidate_count={}, selected_lqer_proofs={}, calibration_status={}",
                    pack.lqer_candidates.len(),
                    pack.lqer_proofs.len(),
                    if pack.quality_calibration.applied {
                        "applied"
                    } else {
                        "not_provided"
                    }
                )
            }),
            evidence_needed: "validate randomized/exact SVD at production tensor sizes and calibrate quality-per-byte scores against held-out model measurements",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "CacheGolf",
            requirement: "fused dequant-attention runtime plus measured long-context quality/timing",
            status: release_status_for(cache_evidence.is_some(), cache_release_ok),
            blocking: !cache_release_ok,
            current_evidence: cache_evidence.map(|evidence| {
                format!(
                    "fused_runtime={:?}, parity_pass={:?}, memory_budget_fit={:?}, long_context_bpb_delta_pct={:?}, speedup_x={:?}",
                    evidence.fused_runtime,
                    evidence.parity_pass,
                    evidence.memory_budget_fit,
                    evidence.long_context_bpb_delta_pct,
                    evidence.speedup_x
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.cache missing; selected_k_bits={}, selected_v_bits={}, selected_policy_bytes={}, selected_context_fits_budget={}, local_bound_covers={}",
                    cache.selected.k_bits,
                    cache.selected.v_bits,
                    cache.selected.estimated_cache_bytes,
                    cache.long_context_eval.selected_context_fits_budget,
                    cache.selected_policy_proof.bound_covers_observed_error
                )
            }),
            evidence_needed: "lower the packed K/V format to CUDA/Metal, test fused parity, then run representative long-context BPB/perplexity and timing",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "DeltaGolf",
            requirement: "trained byte-constrained domain delta with legality and equal-byte quality evidence",
            status: release_status_for(delta_evidence.is_some(), delta_release_ok),
            blocking: !delta_release_ok,
            current_evidence: delta_evidence.map(|evidence| {
                format!(
                    "trained_delta_bytes={:?}, delta_budget_bytes={}, legality_pass={:?}, score_first_trace_or_review={:?}, equal_byte_domain_bpb_delta={:?}",
                    evidence.trained_delta_bytes,
                    delta.selected_bytes,
                    evidence.legality_pass,
                    evidence.score_first_trace_or_review,
                    evidence.equal_byte_domain_bpb_delta
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.delta missing; selected_delta_bytes={}, local_low_rank_proofs={}, static_legality_audit={}, domain_proxy_rows={}",
                    delta.selected_bytes,
                    delta.selected_low_rank_proofs.len(),
                    delta.score_first_legality_audit.pass,
                    delta.domain_evaluation.rows.len()
                )
            }),
            evidence_needed: "estimate full updates/curvature from real domain data, train the delta, record score-first ordering or formal legality review, and compare equal-byte domain BPB/perplexity",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "TrainGolf",
            requirement: "artifact-aware training integrated into production backend with post-export quality comparisons",
            status: release_status_for(train_evidence.is_some(), train_release_ok),
            blocking: !train_release_ok,
            current_evidence: train_evidence.map(|evidence| {
                format!(
                    "gpu_backend_integrated={:?}, proxy_calibrated={:?}, post_export_bpb_delta_vs_posthoc={:?}",
                    evidence.gpu_backend_integrated,
                    evidence.proxy_calibrated,
                    evidence.post_export_bpb_delta_vs_posthoc
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.train missing; local_regularizer_finite={}, export_gap_bound_covers={}, distance_reduction_pct={:.6}",
                    train.local_proof.finite,
                    train.local_proof.export_gap_bound_covers_observed,
                    train.local_proof.distance_reduction_pct
                )
            }),
            evidence_needed: "run backend/GPU artifact-regularized training, calibrate smoothness/gradient proxies, and compare post-export quality against post-hoc quantization",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "Optimizer/Comm Compiler",
            requirement: "distributed optimizer parity, NCCL trace timing, overlap validation, and communication speedup evidence",
            status: release_status_for(optimizer_comm_evidence.is_some(), optimizer_comm_release_ok),
            blocking: !optimizer_comm_release_ok,
            current_evidence: optimizer_comm_evidence.map(|evidence| {
                format!(
                    "distributed_backend_id_present={}, world_size={:?}, expected_world_size={}, optimizer_sharded={:?}, expected_optimizer_sharded={}, nccl_overlap_mode={:?}, expected_nccl_overlap_mode={}, reduce_scatter_parity_pass={:?}, all_gather_parity_pass={:?}, optimizer_update_parity_pass={:?}, nccl_trace_validated={:?}, overlap_validated={:?}, measured_comm_time_ms={:?}, measured_step_time_ms={:?}, communication_speedup_x={:?}",
                    release_nonempty(evidence.distributed_backend_id.as_ref()),
                    evidence.world_size,
                    optimizer_comm.world_size,
                    evidence.optimizer_sharded,
                    optimizer_comm.optimizer_sharded,
                    evidence.nccl_overlap_mode,
                    optimizer_comm.nccl_overlap_mode,
                    evidence.reduce_scatter_parity_pass,
                    evidence.all_gather_parity_pass,
                    evidence.optimizer_update_parity_pass,
                    evidence.nccl_trace_validated,
                    evidence.overlap_validated,
                    evidence.measured_comm_time_ms,
                    evidence.measured_step_time_ms,
                    evidence.communication_speedup_x
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.optimizer_comm missing; local_equivalence={}, world_size={}, optimizer_sharded={}, sharded_total_wire_bytes_per_rank={}",
                    optimizer_comm.local_proof.exact_equivalence,
                    optimizer_comm.world_size,
                    optimizer_comm.optimizer_sharded,
                    optimizer_comm.sharded_total_wire_bytes_per_rank
                )
            }),
            evidence_needed: "run the selected distributed optimizer backend, validate reduce-scatter/local-update/all-gather parity, collect NCCL traces, confirm overlap, and record measured timing/speedup",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "KernelForge",
            requirement: "generated CUDA/Metal kernels with forward/backward parity and backend performance",
            status: release_status_for(kernel_evidence.is_some(), kernel_release_ok),
            blocking: !kernel_release_ok,
            current_evidence: kernel_evidence.map(|evidence| {
                format!(
                    "generated_kernels={:?}, parity_pass={:?}, speedup_x={:?}, memory_reduction_x={:?}",
                    evidence.generated_kernels,
                    evidence.parity_pass,
                    evidence.speedup_x,
                    evidence.memory_reduction_x
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.kernel_forge missing; fusion_primitives={}, exact_tiled_ce_parity={}, scratch_reduction_x={:.6}",
                    kernel_forge.fusion_primitives.len(),
                    kernel_forge.exact_tiled_ce.local_proof.parity_ok,
                    kernel_forge.exact_tiled_ce.scratch_reduction_x
                )
            }),
            evidence_needed: "generate backend kernels, run parity for forward/loss/dH/dW/dWeight, and collect runtime/memory measurements",
        },
        ModelGolfReleaseRequirementReport {
            pillar: "Wind Tunnel",
            requirement: "fresh calibrated profiler traces and external full train/eval timing validation",
            status: release_status_for(wind_evidence.is_some(), wind_release_ok),
            blocking: !wind_release_ok,
            current_evidence: wind_evidence.map(|evidence| {
                format!(
                    "fresh_profiler_traces={:?}, external_timing_validated={:?}, holdout_spearman={:?}, mean_abs_pct_error={:?}",
                    evidence.fresh_profiler_traces,
                    evidence.external_timing_validated,
                    evidence.holdout_spearman,
                    evidence.mean_abs_pct_error
                )
            }).unwrap_or_else(|| {
                format!(
                    "release_evidence.wind_tunnel missing; estimate_only={}, pareto_candidates={}, recommendation={}",
                    wind_tunnel.estimate_only,
                    wind_tunnel.pareto_candidates.len(),
                    wind_tunnel.recommendation
                )
            }),
            evidence_needed: "collect fresh profiler traces, calibrate the cost model, and validate holdout ranking/error on full train/eval runs",
        },
    ]);
    let blocker_count = checklist.iter().filter(|item| item.blocking).count();
    ModelGolfReleaseReadinessReport {
        kind: "modelgolf_release_readiness",
        release_ready: blocker_count == 0,
        local_planner_ready,
        release_evidence_source: release_evidence_path.map(|path| path.display().to_string()),
        blocker_count,
        checklist,
        next_action: if local_planner_ready {
            if blocker_count == 0 {
                "release evidence satisfies all current ModelGolf gates; freeze artifacts, commands, and reviewer packet for release".to_string()
            } else {
                "collect the blocking measured resource-cost, pretrained/held-out, and backend evidence; start with Pack held-out BPB plus CacheGolf fused long-context timing because they gate product claims".to_string()
            }
        } else {
            "fix failing local planner/proof gates before collecting release evidence".to_string()
        },
        evidence_boundary: "fail-closed release gate derived from current local reports plus optional measured release evidence JSON; missing or invalid external evidence remains blocking",
    }
}

fn tensor_role_for_name(name: &str) -> TensorRole {
    match name {
        "qo_bank.q" => TensorRole::AttentionQ,
        "qo_bank.o" => TensorRole::AttentionO,
        "kv_bank.k" => TensorRole::KeyCacheProjection,
        "kv_bank.v" => TensorRole::ValueCacheProjection,
        "mlp_up_bank" => TensorRole::MlpUp,
        "mlp_down_bank" => TensorRole::MlpDown,
        "tok_emb" => TensorRole::TokenEmbedding,
        other if other.contains("gate") => TensorRole::AttentionGate,
        _ => TensorRole::Other,
    }
}

fn role_sensitivity(role: TensorRole) -> f64 {
    match role {
        TensorRole::TokenEmbedding => 1.55,
        TensorRole::AttentionO => 1.25,
        TensorRole::AttentionQ => 1.15,
        TensorRole::KeyCacheProjection => 1.20,
        TensorRole::ValueCacheProjection => 1.05,
        TensorRole::MlpDown => 1.10,
        TensorRole::MlpUp => 0.92,
        TensorRole::AttentionGate => 1.30,
        TensorRole::Other => 1.0,
    }
}

fn quality_loss_surrogate(group: &ModelGolfTensorGroup, bits: u8, lqer: bool, rank: usize) -> f64 {
    let bit_gap = 8u8.saturating_sub(bits) as f64;
    let elems_m = group.elems as f64 / 1_000_000.0;
    let mut loss = group.sensitivity * elems_m * bit_gap * bit_gap / 16.0;
    if lqer {
        let effective_rank = effective_lqer_rank(group.rows, group.cols, rank);
        if effective_rank == 0 {
            return loss;
        }
        let correction = (0.12 * effective_rank as f64).min(0.62);
        let role_bonus = match group.role {
            TensorRole::MlpDown | TensorRole::AttentionO | TensorRole::TokenEmbedding => 1.10,
            TensorRole::MlpUp => 0.92,
            _ => 1.0,
        };
        loss *= 1.0 - (correction * role_bonus).min(0.75);
    }
    loss
}

fn effective_lqer_rank(rows: usize, cols: usize, requested_rank: usize) -> usize {
    requested_rank.min(rows).min(cols)
}

fn estimate_lqer_bytes(rows: usize, cols: usize, spec: &RunSpec) -> usize {
    let rank = spec.quant.lqer.rank;
    if rank == 0 || rows == 0 || cols == 0 {
        return 0;
    }
    let a_weight = rows
        .saturating_mul(rank)
        .saturating_mul(spec.quant.lqer.a_bits as usize)
        .div_ceil(8);
    let b_weight = rank
        .saturating_mul(cols)
        .saturating_mul(spec.quant.lqer.b_bits as usize)
        .div_ceil(8);
    let a_scale = rows.saturating_mul(2);
    let b_scale = rank.saturating_mul(2);
    a_weight + b_weight + a_scale + b_scale
}

fn pack_kernel_for_bits(bits: u8) -> &'static str {
    match bits {
        4 => "pack_signed_i4_per_row_sm90",
        5 => "pack_signed_i5_per_row_sm90",
        6 => "pack_signed_i6_per_row_sm90",
        7 => "pack_signed_i7_per_row_sm90",
        8 => "pack_signed_i8_per_row_sm90",
        _ => "pack_signed_unsupported",
    }
}

fn compression_factor_for_spec(spec: &RunSpec) -> f64 {
    match spec.quant.compression {
        pg_model::CompressionMode::None => 1.0,
        pg_model::CompressionMode::Zstd22 => 0.72,
        pg_model::CompressionMode::Lzma9 => 0.68,
        pg_model::CompressionMode::Pergroup => 0.80,
    }
}

fn compressed_byte_estimate(raw_bytes: usize, factor: f64) -> usize {
    ((raw_bytes as f64) * factor).ceil() as usize
}

fn attention_error_bound(head_dim: usize, k_bits: u8, v_bits: u8) -> f64 {
    let sqrt_d = (head_dim as f64).sqrt();
    let epsilon_k = sqrt_d * 2f64.powi(-(k_bits as i32));
    let epsilon_v = sqrt_d * 2f64.powi(-(v_bits as i32));
    epsilon_v + 2.0 * epsilon_k
}

fn deterministic_modelgolf_values(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32 + 1.0;
            0.8 * (x * phase).sin() + 0.25 * (x * phase * 0.41).cos()
        })
        .collect()
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| {
            let value = *value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let delta = *x as f64 - *y as f64;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn squared_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let delta = *x as f64 - *y as f64;
            delta * delta
        })
        .sum()
}

fn quadratic_loss_f32(weights: &[f32], smoothness: f64) -> f64 {
    0.5 * smoothness
        * weights
            .iter()
            .map(|value| {
                let value = *value as f64;
                value * value
            })
            .sum::<f64>()
}

fn fusion_primitive(
    name: &'static str,
    inputs: &[&'static str],
    outputs: &[&'static str],
    status: &str,
) -> FusionPrimitiveReport {
    FusionPrimitiveReport {
        name,
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
        status: status.to_string(),
        proof_obligation: "fused forward equals function composition and fused backward equals composed Jacobian transpose",
    }
}

fn better_pack_state(candidate: &DpState, old: &DpState) -> bool {
    compare_loss_then_bytes(candidate.loss, candidate.bytes, old.loss, old.bytes)
        == std::cmp::Ordering::Less
}

fn prune_dominated_pack_states(states: BTreeMap<usize, DpState>) -> BTreeMap<usize, DpState> {
    let mut pruned = BTreeMap::new();
    let mut best_loss_at_lower_or_equal_bytes = f64::INFINITY;
    for (bytes, state) in states {
        if state.loss < best_loss_at_lower_or_equal_bytes - 1e-12 {
            best_loss_at_lower_or_equal_bytes = state.loss;
            pruned.insert(bytes, state);
        }
    }
    pruned
}

fn compare_loss_then_bytes(
    loss_a: f64,
    bytes_a: usize,
    loss_b: f64,
    bytes_b: usize,
) -> std::cmp::Ordering {
    loss_a
        .total_cmp(&loss_b)
        .then_with(|| bytes_a.cmp(&bytes_b))
}

fn better_delta_state(candidate: &DeltaDpState, old: &DeltaDpState) -> bool {
    candidate.gain > old.gain + 1e-12
        || ((candidate.gain - old.gain).abs() <= 1e-12 && candidate.bytes < old.bytes)
}

fn prune_dominated_delta_states(
    states: BTreeMap<usize, DeltaDpState>,
) -> BTreeMap<usize, DeltaDpState> {
    let mut pruned = BTreeMap::new();
    let mut best_gain_at_lower_or_equal_bytes = f64::NEG_INFINITY;
    for (bytes, state) in states {
        if state.gain > best_gain_at_lower_or_equal_bytes + 1e-12 {
            best_gain_at_lower_or_equal_bytes = state.gain;
            pruned.insert(bytes, state);
        }
    }
    pruned
}

#[allow(dead_code)]
fn invalid_section(section: &str) -> PgError {
    PgError::InvalidOp(format!(
        "unknown modelgolf section {section}; expected plan, pack, pack-experiment, pack-source-report, lqer-source-report, cache-plan, cache-experiment, cache-source-report, delta-plan, delta-source-report, train-plan, train-source-report, kernel-plan, kernel-experiment, kernel-source-report, wind-plan, wind-experiment, wind-source-report, release-evidence, or release-check"
    ))
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use pg_model::RunSpec;
    use std::fs;
    use std::path::Path;

    fn temp_path(name: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("modelgolf_{name}_{stamp}"))
    }

    fn write_raw_evidence_packet(path: &Path, pillar: &str, claims: serde_json::Value) {
        fs::write(
            path,
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": "modelgolf_raw_evidence",
                "pillar": pillar,
                "claims": claims,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn tiny_spec(path: &Path) -> RunSpec {
        let mut spec = RunSpec::default();
        spec.name = "modelgolf_tiny".to_string();
        spec.model.num_layers = 1;
        spec.model.model_dim = 16;
        spec.model.num_heads = 2;
        spec.model.num_kv_heads = 1;
        spec.model.vocab_size = 64;
        spec.model.mlp_mult = 2.0;
        spec.model.rope.dims = 4;
        spec.model.bigram.enabled = false;
        spec.model.value_embedding.enabled = false;
        spec.train.batch_tokens = 64;
        spec.train.seq_len = 16;
        spec.train.total_iterations = 2;
        spec.quant.target_artifact_bytes = 64 * 1024;
        spec.quant.lqer.enabled = true;
        spec.quant.lqer.rank = 2;
        spec.save(path).unwrap();
        spec
    }

    fn lqer_release_fixture(
        spec_path: &Path,
        artifact_path: &Path,
    ) -> (RunSpec, PackPlannerReport, usize, String) {
        let mut spec = RunSpec::default();
        spec.name = "modelgolf_lqer_fixture_l2_d32_v128".to_string();
        spec.model.num_layers = 2;
        spec.model.model_dim = 32;
        spec.model.num_heads = 4;
        spec.model.num_kv_heads = 2;
        spec.model.vocab_size = 128;
        spec.model.mlp_mult = 2.0;
        spec.model.rope.dims = 8;
        spec.model.bigram.enabled = false;
        spec.model.value_embedding.enabled = false;
        spec.train.batch_tokens = 128;
        spec.train.seq_len = 32;
        spec.train.total_iterations = 2;
        spec.quant.target_artifact_bytes = 8 * 1024;
        spec.quant.lqer.enabled = true;
        spec.quant.lqer.rank = 2;
        spec.quant.lqer.top_k = 3;
        spec.quant.lqer.group_size = 16;
        spec.save(spec_path).unwrap();

        let manifest = pg_quant::compile_quant_layout_manifest(
            &spec.quant,
            Some(&spec.model.to_model_config()),
        )
        .unwrap();
        let tensor_groups = modelgolf_tensor_groups(&manifest.groups);
        let pack = plan_pack(
            &tensor_groups,
            &spec,
            spec.quant.target_artifact_bytes,
            None,
        );
        assert!(pack.feasible);
        assert!(
            !pack.lqer_proofs.is_empty(),
            "release fixture must keep a selected LQER proof"
        );

        let plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        let mut release_model = GptModel::new(spec.model.to_model_config());
        release_model.fill_deterministic();
        let artifact_bytes = pg_quant::export::export_model_with_spec(
            &release_model,
            &spec.quant,
            &plan.variant_fingerprint,
            artifact_path,
        )
        .unwrap();
        assert!(artifact_bytes <= spec.quant.target_artifact_bytes);
        let artifact_sha256 = format!("sha256:{}", modelgolf_sha256_file(artifact_path).unwrap());
        (spec, pack, artifact_bytes, artifact_sha256)
    }

    #[test]
    fn modelgolf_report_covers_all_platform_pillars() {
        let dir = temp_path("report");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("modelgolf.json");
        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::Full,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: None,
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        assert_eq!(report.kind, "modelgolf_constraint_native_platform_plan");
        assert!(report.resource_contract.training_time_budget_seconds > 0.0);
        assert!(report.resource_contract.nominal_power_watts_proxy > 0.0);
        assert!(report.resource_contract.train_energy_budget_joules_proxy > 0.0);
        assert_eq!(report.cost_model.kind, "modelgolf_constraint_cost_model");
        assert_eq!(
            report.cost_model.training_time_budget_seconds,
            report.resource_contract.training_time_budget_seconds
        );
        assert!(report.cost_model.train_tokens_per_budget_second.is_finite());
        assert!(report.cost_model.train_tokens_per_budget_second > 0.0);
        assert!(report.cost_model.train_energy_budget_joules_proxy > 0.0);
        assert!(
            report
                .cost_model
                .evidence_boundary
                .contains("not measured wall energy")
        );
        assert!(report.pack.feasible);
        assert!(report.cache.feasible);
        assert_eq!(
            report.cache.residual_sketch_policy.kind,
            "cachegolf_residual_sketch_policy"
        );
        assert_eq!(
            report.cache.eviction_policy.kind,
            "cachegolf_eviction_policy"
        );
        assert_eq!(
            report.cache.long_context_eval.kind,
            "cachegolf_long_context_memory_eval"
        );
        assert!(
            report
                .cache
                .long_context_eval
                .rows
                .iter()
                .any(|row| row.context_tokens == report.cache.context_tokens)
        );
        assert!(
            report
                .cache
                .long_context_eval
                .max_context_tokens_under_budget
                > 0
        );
        assert_eq!(
            report.pack.quality_comparison.kind,
            "modelgolf_pack_equal_byte_quality_comparison"
        );
        assert!(
            report
                .pack
                .quality_comparison
                .rows
                .iter()
                .any(|row| row.plan_name == report.pack.quality_comparison.selected_plan_name)
        );
        assert!(report.pack.quality_comparison.rows.iter().all(|row| {
            row.local_weighted_residual_mse_proxy.is_finite()
                && row.local_activation_ce_bound_proxy.is_finite()
        }));
        assert!(
            report
                .pack
                .quality_comparison
                .evidence_boundary
                .contains("not held-out BPB")
        );
        assert!(report.pack.lqer_proofs.iter().all(|proof| {
            proof.finite
                && proof.residual_frobenius_after < proof.residual_frobenius_before
                && proof.ce_linf_bound_after < proof.ce_linf_bound_before
                && proof.selected_rank_no_worse_than_lower_rank
        }));
        assert!(!report.delta.candidates.is_empty());
        assert_eq!(
            report.delta.algorithm,
            "exact_multiple_choice_byte_constrained_delta_allocation"
        );
        assert!(
            report
                .platform_status
                .iter()
                .any(|status| status.module == "Optimizer/Comm Compiler")
        );
        assert!(
            report
                .platform_status
                .iter()
                .any(|status| status.module == "KernelForge")
        );
        assert!(
            report
                .platform_status
                .iter()
                .any(|status| status.module == "ScaleGolf")
        );
        assert_eq!(report.scale_golf.kind, "scalegolf_track_planner");
        assert_eq!(report.scale_golf.tracks.len(), 5);
        assert_eq!(report.scale_golf.invariant.track_count, 5);
        assert_eq!(report.scale_golf.invariant.expected_track_count, 5);
        assert!(report.scale_golf.invariant.all_tracks_use_resource_contract);
        assert!(report.scale_golf.invariant.all_tracks_use_model_ir);
        assert!(report.scale_golf.invariant.all_tracks_use_artifact_compiler);
        assert!(report.scale_golf.invariant.all_tracks_use_runtime_planner);
        assert!(report.scale_golf.invariant.all_tracks_use_evaluator);
        assert!(report.scale_golf.invariant.all_tracks_use_cost_model);
        assert!(
            report
                .scale_golf
                .tracks
                .iter()
                .any(|track| track.track == "PG-Lite")
        );
        assert!(
            report
                .scale_golf
                .tracks
                .iter()
                .any(|track| track.track == "ParameterGolf")
        );
        assert!(
            report
                .scale_golf
                .tracks
                .iter()
                .any(|track| track.track == "DeltaGolf")
        );
        assert!(
            report
                .scale_golf
                .tracks
                .iter()
                .any(|track| track.track == "LongContext")
        );
        assert!(
            report
                .scale_golf
                .tracks
                .iter()
                .any(|track| track.track == "ScaleGolf")
        );
        assert!(report.scale_golf.local_ready);
        assert!(!report.scale_golf.release_ready);
        assert!(report.optimizer_comm.local_proof.finite);
        assert!(report.optimizer_comm.local_proof.exact_equivalence);
        assert!(report.optimizer_comm.shard_separable_update_contract);
        assert!(report.kernel_forge.exact_tiled_ce.local_proof.parity_ok);
        assert!(report.kernel_forge.exact_tiled_ce.local_proof.finite);
        assert!(report.train.local_proof.finite);
        assert!(
            report
                .train
                .local_proof
                .gradient_matches_stop_gradient_objective
        );
        assert!(report.train.local_proof.export_gap_bound_covers_observed);
        assert!(
            report
                .train
                .local_proof
                .distance_sq_after_fixed_projection_step
                < report.train.local_proof.distance_sq_before
        );
        assert!(
            report
                .delta
                .selected_low_rank_proofs
                .iter()
                .all(|proof| proof.finite
                    && proof.curvature_positive
                    && proof.selected_rank_no_worse_than_lower_rank)
        );
        assert!(report.delta.score_first_legality_audit.pass);
        assert_eq!(
            report.delta.domain_evaluation.kind,
            "deltagolf_equal_byte_domain_proxy"
        );
        assert!(
            report
                .delta
                .domain_evaluation
                .rows
                .iter()
                .any(|row| row.plan_name == report.delta.domain_evaluation.selected_plan_name)
        );
        assert!(report.delta.domain_evaluation.rows.iter().all(|row| {
            row.estimated_domain_loss_proxy.is_finite() && row.estimated_bpb_delta_proxy.is_finite()
        }));
        assert!(!report.release_readiness.release_ready);
        assert!(report.release_readiness.local_planner_ready);
        assert!(report.release_readiness.blocker_count > 0);
        assert!(
            report
                .release_readiness
                .checklist
                .iter()
                .any(|item| item.pillar == "ModelGolf Pack"
                    && item.status == "missing_release_evidence"
                    && item.blocking)
        );
        assert!(output.exists());
    }

    #[test]
    fn optimizer_comm_plans_sharded_collectives_and_proves_equivalence() {
        let mut spec = RunSpec::default();
        spec.train.world_size = 4;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.runtime.nccl_overlap_mode = pg_model::NcclOverlapMode::BucketedMeasured;
        spec.runtime.sharded_muon_local_graph = true;
        spec.runtime.sharded_muon_pre_norm_graph = true;
        spec.runtime.sharded_muon_bf16_shadow_all_gather = true;

        let report = plan_optimizer_comm(&spec);
        assert_eq!(report.kind, "modelgolf_optimizer_comm_plan");
        assert!(report.optimizer_sharded);
        assert_eq!(report.world_size, 4);
        assert!(report.bank_payload_bytes_f32 > 0);
        assert!(report.sharded_reduce_scatter_wire_bytes_per_rank > 0);
        assert!(report.sharded_param_all_gather_wire_bytes_per_rank > 0);
        assert!(report.bf16_shadow_all_gather_wire_bytes_per_rank > 0);
        assert!(
            report.sharded_total_wire_bytes_per_rank
                > report.sharded_reduce_scatter_wire_bytes_per_rank
        );
        assert_eq!(report.owned_optimizer_state_reduction_x, 4.0);
        assert!(report.local_graph_requested);
        assert!(report.pre_norm_graph_requested);
        assert!(report.bf16_shadow_all_gather_requested);
        assert!(report.local_proof.finite);
        assert!(report.local_proof.exact_equivalence);
        assert!(report.local_proof.max_abs_diff <= 1e-12);
        assert_eq!(report.local_proof.proof_world_size, 4);
        assert_eq!(report.local_proof.shard_ranges.len(), 4);
        assert_eq!(
            report
                .local_proof
                .shard_ranges
                .iter()
                .map(|range| range.elements)
                .sum::<usize>(),
            report.local_proof.proof_parameter_elems
        );
    }

    #[test]
    fn release_check_fails_closed_without_external_evidence() {
        let dir = temp_path("release_check");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("release_check.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ReleaseCheck,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: None,
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let readiness = &report.release_readiness;
        assert_eq!(readiness.kind, "modelgolf_release_readiness");
        assert!(!readiness.release_ready);
        assert!(readiness.local_planner_ready);
        assert!(readiness.blocker_count >= 6);
        assert!(
            readiness
                .checklist
                .iter()
                .any(|item| item.pillar == "CacheGolf"
                    && item.requirement.contains("long-context")
                    && item.blocking)
        );
        let serialized: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output).unwrap()).unwrap();
        assert_eq!(
            serialized["kind"].as_str(),
            Some("modelgolf_release_readiness")
        );
        assert_eq!(serialized["release_ready"].as_bool(), Some(false));
    }

    #[test]
    fn lqer_source_report_requires_selected_lqer_proofs() {
        let dir = temp_path("lqer_source_requires_selected");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let spec = tiny_spec(&spec_path);
        let plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        let manifest = pg_quant::compile_quant_layout_manifest(
            &spec.quant,
            Some(&spec.model.to_model_config()),
        )
        .unwrap();
        let tensor_groups = modelgolf_tensor_groups(&manifest.groups);
        let pack = plan_pack(&tensor_groups, &spec, 1, None);
        assert!(!pack.lqer_candidates.is_empty());
        assert!(pack.lqer_proofs.is_empty());

        let resource_contract = ResourceContractReport {
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            artifact_budget_bytes: 1,
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            training_time_budget_seconds: f64::from(spec.train.max_wallclock_seconds),
            nominal_power_watts_proxy: 65.0,
            train_energy_budget_joules_proxy: f64::from(spec.train.max_wallclock_seconds) * 65.0,
            context_tokens: 128,
            batch_sequences: 2,
            train_batch_tokens: spec.train.batch_tokens,
            eval_stride: spec.eval.stride,
            adaptation_legality: "score_first_adaptation_only".to_string(),
            evidence_boundary: "unit test resource contract",
        };
        let err = run_modelgolf_lqer_source_report(
            &spec,
            &plan.variant_fingerprint,
            &resource_contract,
            &pack,
            &ModelGolfOptions {
                section: ModelGolfSection::LqerSourceReport,
                spec: spec_path,
                output: None,
                hardware: "unit_test".to_string(),
                runtime: "portable_cpu".to_string(),
                memory_budget_bytes: Some(8 * 1024 * 1024),
                latency_target_ms: Some(100.0),
                quality_budget_ppl_pct: Some(3.0),
                context: Some(128),
                batch: Some(2),
                artifact_budget_bytes: Some(1),
                delta_budget_bytes: Some(64 * 1024),
                proof_artifact: None,
                quality_calibration: None,
                release_evidence: None,
                release_artifact: None,
                validation_dataset_id: None,
                validation_command: None,
                heldout_bpb: None,
                baseline_bpb: None,
                decode_tokens_per_second: None,
                backend_id: None,
                kernel_id: None,
                long_context_dataset_id: None,
                fused_runtime: None,
                parity_pass: None,
                long_context_bpb_delta_pct: None,
                speedup_x: None,
                calibration_dataset_id: Some("unit-lqer-calibration".to_string()),
                production_svd_validated: Some(true),
                calibrated_tensor_sensitivity: Some(true),
                equal_byte_bpb_delta: Some(-0.01),
                domain_dataset_id: None,
                trained_delta_bytes: None,
                legality_pass: None,
                score_first_trace_or_review: None,
                equal_byte_domain_bpb_delta: None,
                training_run_id: None,
                gpu_backend_integrated: None,
                proxy_calibrated: None,
                post_export_bpb_delta_vs_posthoc: None,
                measurement_run_id: None,
                power_meter_id: None,
                wall_time_seconds: None,
                average_power_watts: None,
                energy_joules: None,
                telemetry_validated: None,
                distributed_backend_id: None,
                reduce_scatter_parity_pass: None,
                all_gather_parity_pass: None,
                optimizer_update_parity_pass: None,
                nccl_trace_validated: None,
                overlap_validated: None,
                measured_comm_time_ms: None,
                measured_step_time_ms: None,
                communication_speedup_x: None,
                generated_kernel_ids: None,
                generated_kernels: None,
                memory_reduction_x: None,
                trace_corpus_id: None,
                calibration_report_id: None,
                fresh_profiler_traces: None,
                external_timing_validated: None,
                holdout_spearman: None,
                mean_abs_pct_error: None,
                evidence_source: None,
                source_report_dir: None,
                evidence_id: None,
                generated_at: None,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("selected no LQER proofs to bind"),
            "{err}"
        );
    }

    #[test]
    fn kernel_forge_scratch_accounting_excludes_live_inputs() {
        let mut spec = RunSpec::default();
        spec.model.model_dim = 128;
        spec.model.vocab_size = 2048;
        spec.train.batch_tokens = 32;

        let kernel = plan_kernel_forge(&spec);
        let ce = &kernel.exact_tiled_ce;
        let expected_full_logits = ce.rows_m * ce.vocab_v * 4;
        let expected_tile_logits = ce.rows_m * ce.tile_t * 4;
        let expected_row_stats = ce.rows_m * 8;
        let live_input_bytes_f16 = ce.rows_m * ce.hidden_d * 2 + ce.vocab_v * ce.hidden_d * 2;

        assert_eq!(ce.tile_t, 512);
        assert_eq!(ce.full_logits_bytes_f32, expected_full_logits);
        assert_eq!(ce.tiled_logits_scratch_bytes_f32, expected_tile_logits);
        assert_eq!(ce.row_stats_scratch_bytes_f32, expected_row_stats);
        assert_eq!(
            ce.tiled_scratch_bytes_estimate,
            expected_tile_logits + expected_row_stats
        );
        assert!(
            ce.tiled_scratch_bytes_estimate < live_input_bytes_f16,
            "live hidden/lm_head tensors must not be counted as CE scratch"
        );
        assert!(
            (ce.scratch_reduction_x
                - expected_full_logits as f64
                    / (expected_tile_logits + expected_row_stats).max(1) as f64)
                .abs()
                < 1e-12
        );

        let proof = &ce.local_proof;
        assert_eq!(
            proof.tiled_scratch_bytes_estimate,
            proof.tiled_logits_scratch_bytes_f32 + proof.row_stats_scratch_bytes_f32
        );
        assert!(
            (proof.scratch_reduction_x
                - proof.full_logits_bytes_f32 as f64
                    / proof.tiled_scratch_bytes_estimate.max(1) as f64)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn release_check_accepts_complete_measured_evidence() {
        let dir = temp_path("release_check_complete");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let artifact_path = dir.join("release_artifact.pgrs");
        let (spec, pack, release_artifact_bytes, release_artifact_sha256) =
            lqer_release_fixture(&spec_path, &artifact_path);
        let cache = plan_cache(&spec, 128, 2, Some(8 * 1024 * 1024)).unwrap();
        let delta = plan_delta(&spec, 64 * 1024).unwrap();
        let release_artifact_budget_bytes = pack.target_artifact_bytes;
        assert!(!pack.lqer_proofs.is_empty());
        let selected_lqer_group_names = pack
            .lqer_proofs
            .iter()
            .map(|proof| proof.group.clone())
            .collect::<Vec<_>>();
        let selected_delta_names = delta
            .selected_deltas
            .iter()
            .map(|delta| delta.name.clone())
            .collect::<Vec<_>>();
        let kernel_forge = plan_kernel_forge(&spec);
        let optimizer_comm = plan_optimizer_comm(&spec);
        let resource_wall_time_budget_seconds =
            f64::from(spec.train.max_wallclock_seconds.max(1e-6));
        let (resource_nominal_power_watts, _) =
            nominal_power_watts_proxy("unit_test", spec.train.world_size);
        let resource_energy_budget_joules =
            resource_wall_time_budget_seconds * resource_nominal_power_watts;
        let measured_wall_time_seconds = resource_wall_time_budget_seconds * 0.5;
        let measured_average_power_watts = resource_nominal_power_watts * 0.5;
        let measured_energy_joules = measured_wall_time_seconds * measured_average_power_watts;
        assert!(delta.selected_bytes > 0);
        assert!(release_artifact_bytes <= release_artifact_budget_bytes);
        let raw_pack_evidence = dir.join("raw_pack_eval.json");
        let raw_lqer_evidence = dir.join("raw_lqer_calibration.json");
        let raw_cache_evidence = dir.join("raw_cache_backend.json");
        let raw_delta_evidence = dir.join("raw_delta_training.json");
        let raw_train_evidence = dir.join("raw_train_backend.json");
        let raw_resource_cost_evidence = dir.join("raw_resource_cost.json");
        let raw_optimizer_comm_evidence = dir.join("raw_optimizer_comm.json");
        let raw_kernel_evidence = dir.join("raw_kernel_parity.json");
        let raw_wind_evidence = dir.join("raw_wind_traces.json");
        write_raw_evidence_packet(
            &raw_pack_evidence,
            "pack",
            serde_json::json!({
                "artifact_sha256": release_artifact_sha256,
                "validation_dataset_id": "unit-heldout",
                "validation_command": "cargo test -p pg-local release_check",
                "artifact_bytes": release_artifact_bytes,
                "strict_reload_pass": true,
                "heldout_bpb": 1.01,
                "baseline_bpb": 1.0,
                "relative_bpb_increase_pct": 1.0,
                "decode_tokens_per_second": 42.0,
            }),
        );
        write_raw_evidence_packet(
            &raw_lqer_evidence,
            "lqer",
            serde_json::json!({
                "calibration_dataset_id": "unit-lqer-calibration",
                "selected_lqer_group_names": selected_lqer_group_names,
                "production_svd_validated": true,
                "calibrated_tensor_sensitivity": true,
                "equal_byte_bpb_delta": -0.01,
            }),
        );
        write_raw_evidence_packet(
            &raw_cache_evidence,
            "cache",
            serde_json::json!({
                "backend_id": "unit-cache-backend",
                "kernel_id": "unit-cache-kernel",
                "long_context_dataset_id": "unit-long-context",
                "k_bits": cache.selected.k_bits,
                "v_bits": cache.selected.v_bits,
                "block_size_tokens": cache.selected.block_size_tokens,
                "layout": cache.selected.layout,
                "context_tokens": cache.context_tokens,
                "batch_sequences": cache.batch_sequences,
                "fused_runtime": true,
                "parity_pass": true,
                "memory_budget_fit": true,
                "long_context_bpb_delta_pct": 1.0,
                "speedup_x": 1.25,
            }),
        );
        write_raw_evidence_packet(
            &raw_delta_evidence,
            "delta",
            serde_json::json!({
                "domain_dataset_id": "unit-domain",
                "selected_delta_names": selected_delta_names,
                "trained_delta_bytes": delta.selected_bytes,
                "legality_pass": true,
                "score_first_trace_or_review": true,
                "equal_byte_domain_bpb_delta": -0.02,
            }),
        );
        write_raw_evidence_packet(
            &raw_train_evidence,
            "train",
            serde_json::json!({
                "training_run_id": "unit-train-run",
                "backend_id": "unit-train-backend",
                "gpu_backend_integrated": true,
                "proxy_calibrated": true,
                "post_export_bpb_delta_vs_posthoc": -0.01,
            }),
        );
        write_raw_evidence_packet(
            &raw_resource_cost_evidence,
            "resource_cost",
            serde_json::json!({
                "measurement_run_id": "unit-train-run",
                "power_meter_id": "unit-power-meter",
                "wall_time_seconds": measured_wall_time_seconds,
                "average_power_watts": measured_average_power_watts,
                "energy_joules": measured_energy_joules,
                "telemetry_validated": true,
                "wall_time_budget_seconds": resource_wall_time_budget_seconds,
                "energy_budget_joules": resource_energy_budget_joules,
                "wall_time_budget_fit": true,
                "energy_budget_fit": true,
                "energy_consistency_error_pct": 0.0,
            }),
        );
        write_raw_evidence_packet(
            &raw_optimizer_comm_evidence,
            "optimizer_comm",
            serde_json::json!({
                "distributed_backend_id": "unit-distributed-backend",
                "world_size": optimizer_comm.world_size,
                "optimizer_sharded": optimizer_comm.optimizer_sharded,
                "nccl_overlap_mode": optimizer_comm.nccl_overlap_mode,
                "sharded_total_wire_bytes_per_rank": optimizer_comm.sharded_total_wire_bytes_per_rank,
                "owned_optimizer_state_reduction_x": optimizer_comm.owned_optimizer_state_reduction_x,
                "reduce_scatter_parity_pass": true,
                "all_gather_parity_pass": true,
                "optimizer_update_parity_pass": true,
                "nccl_trace_validated": true,
                "overlap_validated": true,
                "measured_comm_time_ms": 1.25,
                "measured_step_time_ms": 10.0,
                "communication_speedup_x": 1.05,
            }),
        );
        write_raw_evidence_packet(
            &raw_kernel_evidence,
            "kernel_forge",
            serde_json::json!({
                "backend_id": "unit-kernel-backend",
                "generated_kernel_ids": ["unit-exact-tiled-ce"],
                "tile_t": kernel_forge.exact_tiled_ce.tile_t,
                "generated_kernels": true,
                "parity_pass": true,
                "speedup_x": 1.10,
                "memory_reduction_x": 2.0,
            }),
        );
        write_raw_evidence_packet(
            &raw_wind_evidence,
            "wind_tunnel",
            serde_json::json!({
                "trace_corpus_id": "unit-trace-corpus",
                "calibration_report_id": "unit-calibration-report",
                "fresh_profiler_traces": true,
                "external_timing_validated": true,
                "holdout_spearman": 0.80,
                "mean_abs_pct_error": 10.0,
            }),
        );
        let pack_source_path = dir.join("source_pack.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::PackSourceReport,
            spec: spec_path.clone(),
            output: Some(pack_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: Some(artifact_path.clone()),
            validation_dataset_id: Some("unit-heldout".to_string()),
            validation_command: Some("cargo test -p pg-local release_check".to_string()),
            heldout_bpb: Some(1.01),
            baseline_bpb: Some(1.0),
            decode_tokens_per_second: Some(42.0),
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_pack_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let pack_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&pack_source_path).unwrap()).unwrap();
        assert_eq!(
            pack_source["kind"].as_str(),
            Some("modelgolf_release_source_report")
        );
        assert_eq!(pack_source["pillar"].as_str(), Some("pack"));
        assert_eq!(
            pack_source["claims"]["strict_reload_pass"].as_bool(),
            Some(true)
        );
        assert!(
            (pack_source["claims"]["relative_bpb_increase_pct"]
                .as_f64()
                .unwrap()
                - 1.0)
                .abs()
                <= 1e-9
        );
        let lqer_source_path = dir.join("source_lqer.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::LqerSourceReport,
            spec: spec_path.clone(),
            output: Some(lqer_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: Some("unit-lqer-calibration".to_string()),
            production_svd_validated: Some(true),
            calibrated_tensor_sensitivity: Some(true),
            equal_byte_bpb_delta: Some(-0.01),
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_lqer_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let lqer_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&lqer_source_path).unwrap()).unwrap();
        assert_eq!(lqer_source["pillar"].as_str(), Some("lqer"));
        assert_eq!(
            lqer_source["claims"]["production_svd_validated"].as_bool(),
            Some(true)
        );
        let cache_source_path = dir.join("source_cache.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::CacheSourceReport,
            spec: spec_path.clone(),
            output: Some(cache_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: Some("unit-cache-backend".to_string()),
            kernel_id: Some("unit-cache-kernel".to_string()),
            long_context_dataset_id: Some("unit-long-context".to_string()),
            fused_runtime: Some(true),
            parity_pass: Some(true),
            long_context_bpb_delta_pct: Some(1.0),
            speedup_x: Some(1.25),
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_cache_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let cache_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&cache_source_path).unwrap()).unwrap();
        assert_eq!(cache_source["pillar"].as_str(), Some("cache"));
        assert_eq!(
            cache_source["claims"]["k_bits"].as_u64(),
            Some(cache.selected.k_bits as u64)
        );
        assert_eq!(
            cache_source["claims"]["memory_budget_fit"].as_bool(),
            Some(true)
        );
        let delta_source_path = dir.join("source_delta.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::DeltaSourceReport,
            spec: spec_path.clone(),
            output: Some(delta_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: Some("unit-domain".to_string()),
            trained_delta_bytes: Some(delta.selected_bytes),
            legality_pass: Some(true),
            score_first_trace_or_review: Some(true),
            equal_byte_domain_bpb_delta: Some(-0.02),
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_delta_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let delta_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&delta_source_path).unwrap()).unwrap();
        assert_eq!(delta_source["pillar"].as_str(), Some("delta"));

        let train_source_path = dir.join("source_train.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::TrainSourceReport,
            spec: spec_path.clone(),
            output: Some(train_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: Some("unit-train-backend".to_string()),
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: Some("unit-train-run".to_string()),
            gpu_backend_integrated: Some(true),
            proxy_calibrated: Some(true),
            post_export_bpb_delta_vs_posthoc: Some(-0.01),
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_train_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let train_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&train_source_path).unwrap()).unwrap();
        assert_eq!(train_source["pillar"].as_str(), Some("train"));

        let resource_cost_source_path = dir.join("source_resource_cost.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ResourceCostSourceReport,
            spec: spec_path.clone(),
            output: Some(resource_cost_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: Some("unit-train-run".to_string()),
            power_meter_id: Some("unit-power-meter".to_string()),
            wall_time_seconds: Some(measured_wall_time_seconds),
            average_power_watts: Some(measured_average_power_watts),
            energy_joules: Some(measured_energy_joules),
            telemetry_validated: Some(true),
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_resource_cost_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let resource_cost_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&resource_cost_source_path).unwrap()).unwrap();
        assert_eq!(
            resource_cost_source["pillar"].as_str(),
            Some("resource_cost")
        );
        assert_eq!(
            resource_cost_source["claims"]["energy_budget_fit"].as_bool(),
            Some(true)
        );

        let optimizer_comm_source_path = dir.join("source_optimizer_comm.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::OptimizerCommSourceReport,
            spec: spec_path.clone(),
            output: Some(optimizer_comm_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: Some("unit-distributed-backend".to_string()),
            reduce_scatter_parity_pass: Some(true),
            all_gather_parity_pass: Some(true),
            optimizer_update_parity_pass: Some(true),
            nccl_trace_validated: Some(true),
            overlap_validated: Some(true),
            measured_comm_time_ms: Some(1.25),
            measured_step_time_ms: Some(10.0),
            communication_speedup_x: Some(1.05),
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_optimizer_comm_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let optimizer_comm_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&optimizer_comm_source_path).unwrap())
                .unwrap();
        assert_eq!(
            optimizer_comm_source["pillar"].as_str(),
            Some("optimizer_comm")
        );
        assert_eq!(
            optimizer_comm_source["claims"]["optimizer_update_parity_pass"].as_bool(),
            Some(true)
        );

        let kernel_source_path = dir.join("source_kernel_forge.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::KernelSourceReport,
            spec: spec_path.clone(),
            output: Some(kernel_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: Some("unit-kernel-backend".to_string()),
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: Some(true),
            long_context_bpb_delta_pct: None,
            speedup_x: Some(1.10),
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: Some("unit-exact-tiled-ce".to_string()),
            generated_kernels: Some(true),
            memory_reduction_x: Some(2.0),
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: Some(raw_kernel_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let kernel_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&kernel_source_path).unwrap()).unwrap();
        assert_eq!(kernel_source["pillar"].as_str(), Some("kernel_forge"));

        let wind_source_path = dir.join("source_wind_tunnel.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::WindSourceReport,
            spec: spec_path.clone(),
            output: Some(wind_source_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: Some("unit-trace-corpus".to_string()),
            calibration_report_id: Some("unit-calibration-report".to_string()),
            fresh_profiler_traces: Some(true),
            external_timing_validated: Some(true),
            holdout_spearman: Some(0.80),
            mean_abs_pct_error: Some(10.0),
            evidence_source: Some(raw_wind_evidence.clone()),
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        let wind_source: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&wind_source_path).unwrap()).unwrap();
        assert_eq!(wind_source["pillar"].as_str(), Some("wind_tunnel"));
        let evidence_path = dir.join("release_evidence.json");
        let output = dir.join("release_check.json");
        run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ReleaseEvidence,
            spec: spec_path.clone(),
            output: Some(evidence_path.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: Some(dir.clone()),
            evidence_id: Some("unit-release-evidence".to_string()),
            generated_at: Some("2026-06-19T00:00:00Z".to_string()),
        })
        .unwrap();
        let evidence: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&evidence_path).unwrap()).unwrap();
        assert_eq!(
            evidence["kind"].as_str(),
            Some("modelgolf_release_evidence")
        );
        assert_eq!(
            evidence["source_reports"].as_array().map(Vec::len),
            Some(modelgolf_required_release_pillars().len())
        );
        let source_reports = evidence["source_reports"].as_array().unwrap();
        assert!(
            source_reports.iter().all(|source| {
                let path = source["path"].as_str().unwrap();
                !Path::new(path).is_absolute()
            }),
            "release-evidence should emit paths portable relative to the evidence file"
        );
        assert!(
            source_reports
                .iter()
                .any(|source| source["pillar"].as_str() == Some("resource_cost"))
        );
        assert!(
            source_reports
                .iter()
                .any(|source| source["pillar"].as_str() == Some("optimizer_comm"))
        );

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ReleaseCheck,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: Some(evidence_path.clone()),
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let readiness = &report.release_readiness;
        assert!(readiness.local_planner_ready);
        assert!(readiness.release_ready);
        assert_eq!(readiness.blocker_count, 0);
        let expected_evidence_source = evidence_path.display().to_string();
        assert_eq!(
            readiness.release_evidence_source.as_deref(),
            Some(expected_evidence_source.as_str())
        );
        assert!(
            readiness
                .checklist
                .iter()
                .all(|item| item.status == "satisfied" && !item.blocking)
        );
        let serialized: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output).unwrap()).unwrap();
        assert_eq!(serialized["release_ready"].as_bool(), Some(true));
        assert_eq!(serialized["blocker_count"].as_u64(), Some(0));

        fs::write(&raw_wind_evidence, "tampered raw profiler trace\n").unwrap();
        let release_evidence_contract = ResourceContractReport {
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            artifact_budget_bytes: release_artifact_budget_bytes,
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            training_time_budget_seconds: f64::from(spec.train.max_wallclock_seconds),
            nominal_power_watts_proxy: 65.0,
            train_energy_budget_joules_proxy: f64::from(spec.train.max_wallclock_seconds) * 65.0,
            context_tokens: 128,
            batch_sequences: 2,
            train_batch_tokens: spec.train.batch_tokens,
            eval_stride: spec.eval.stride,
            adaptation_legality: "score_first_adaptation_only".to_string(),
            evidence_boundary: "unit test resource contract",
        };
        let release_evidence_plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        let generation_rejected = run_modelgolf_release_evidence(
            &spec,
            &release_evidence_plan.variant_fingerprint,
            &release_evidence_contract,
            &ModelGolfOptions {
                section: ModelGolfSection::ReleaseEvidence,
                spec: dir.join("spec.toml"),
                output: None,
                hardware: "unit_test".to_string(),
                runtime: "portable_cpu".to_string(),
                memory_budget_bytes: Some(8 * 1024 * 1024),
                latency_target_ms: Some(100.0),
                quality_budget_ppl_pct: Some(3.0),
                context: Some(128),
                batch: Some(2),
                artifact_budget_bytes: Some(release_artifact_budget_bytes),
                delta_budget_bytes: Some(64 * 1024),
                proof_artifact: None,
                quality_calibration: None,
                release_evidence: None,
                release_artifact: None,
                validation_dataset_id: None,
                validation_command: None,
                heldout_bpb: None,
                baseline_bpb: None,
                decode_tokens_per_second: None,
                backend_id: None,
                kernel_id: None,
                long_context_dataset_id: None,
                fused_runtime: None,
                parity_pass: None,
                long_context_bpb_delta_pct: None,
                speedup_x: None,
                calibration_dataset_id: None,
                production_svd_validated: None,
                calibrated_tensor_sensitivity: None,
                equal_byte_bpb_delta: None,
                domain_dataset_id: None,
                trained_delta_bytes: None,
                legality_pass: None,
                score_first_trace_or_review: None,
                equal_byte_domain_bpb_delta: None,
                training_run_id: None,
                gpu_backend_integrated: None,
                proxy_calibrated: None,
                post_export_bpb_delta_vs_posthoc: None,
                measurement_run_id: None,
                power_meter_id: None,
                wall_time_seconds: None,
                average_power_watts: None,
                energy_joules: None,
                telemetry_validated: None,
                distributed_backend_id: None,
                reduce_scatter_parity_pass: None,
                all_gather_parity_pass: None,
                optimizer_update_parity_pass: None,
                nccl_trace_validated: None,
                overlap_validated: None,
                measured_comm_time_ms: None,
                measured_step_time_ms: None,
                communication_speedup_x: None,
                generated_kernel_ids: None,
                generated_kernels: None,
                memory_reduction_x: None,
                trace_corpus_id: None,
                calibration_report_id: None,
                fresh_profiler_traces: None,
                external_timing_validated: None,
                holdout_spearman: None,
                mean_abs_pct_error: None,
                evidence_source: None,
                source_report_dir: Some(dir.clone()),
                evidence_id: Some("unit-release-evidence-rejected".to_string()),
                generated_at: Some("2026-06-19T00:00:00Z".to_string()),
            },
        )
        .unwrap_err();
        assert!(
            generation_rejected
                .to_string()
                .contains("invalid release source reports"),
            "{generation_rejected}"
        );
        assert!(
            generation_rejected.to_string().contains("raw_evidence"),
            "{generation_rejected}"
        );
        let raw_rejected = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ReleaseCheck,
            spec: dir.join("spec.toml"),
            output: Some(dir.join("release_check_raw_rejected.json")),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: Some(evidence_path.clone()),
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        assert!(!raw_rejected.release_readiness.release_ready);
        assert!(
            raw_rejected
                .release_readiness
                .checklist
                .iter()
                .any(|item| item.pillar == "Release Evidence Binding"
                    && item.status == "invalid_release_evidence"
                    && item.current_evidence.contains("raw_evidence")
                    && (item.current_evidence.contains("sha256 mismatch")
                        || item.current_evidence.contains("byte count mismatch")))
        );
        write_raw_evidence_packet(
            &raw_wind_evidence,
            "wind_tunnel",
            serde_json::json!({
                "trace_corpus_id": "unit-trace-corpus",
                "calibration_report_id": "unit-calibration-report",
                "fresh_profiler_traces": true,
                "external_timing_validated": true,
                "holdout_spearman": 0.80,
                "mean_abs_pct_error": 10.0,
            }),
        );

        let tampered_pack_report = dir.join("source_pack.json");
        let mut tampered: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&tampered_pack_report).unwrap()).unwrap();
        tampered["claims"]["strict_reload_pass"] = serde_json::Value::Bool(false);
        fs::write(
            &tampered_pack_report,
            serde_json::to_string_pretty(&tampered).unwrap(),
        )
        .unwrap();
        let rejected = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::ReleaseCheck,
            spec: dir.join("spec.toml"),
            output: Some(dir.join("release_check_rejected.json")),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(8 * 1024 * 1024),
            latency_target_ms: Some(100.0),
            quality_budget_ppl_pct: Some(3.0),
            context: Some(128),
            batch: Some(2),
            artifact_budget_bytes: Some(release_artifact_budget_bytes),
            delta_budget_bytes: Some(64 * 1024),
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: Some(evidence_path),
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();
        assert!(!rejected.release_readiness.release_ready);
        assert!(
            rejected
                .release_readiness
                .checklist
                .iter()
                .any(|item| item.pillar == "Release Evidence Binding"
                    && item.status == "invalid_release_evidence"
                    && item.current_evidence.contains("sha256 mismatch"))
        );
    }

    #[test]
    fn release_quality_recomputes_relative_bpb() {
        assert!(release_quality_within_budget(
            Some(1.01),
            Some(1.0),
            Some(1.0),
            Some(3.0)
        ));
        assert!(!release_quality_within_budget(
            Some(1.20),
            Some(1.0),
            Some(1.0),
            Some(3.0)
        ));
        assert!(!release_quality_within_budget(
            Some(1.01),
            Some(0.0),
            Some(1.0),
            Some(3.0)
        ));
    }

    #[test]
    fn modelgolf_sha256_matches_independent_vectors() {
        let mut empty = ModelGolfSha256State::new();
        empty.update(b"");
        assert_eq!(
            modelgolf_hex_digest(empty.finalize()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let mut abc = ModelGolfSha256State::new();
        abc.update(b"abc");
        assert_eq!(
            modelgolf_hex_digest(abc.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let mut chunked = ModelGolfSha256State::new();
        chunked.update(b"a");
        chunked.update(b"b");
        chunked.update(b"c");
        assert_eq!(
            modelgolf_hex_digest(chunked.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn delta_planner_emits_weighted_low_rank_proofs() {
        let mut spec = RunSpec::default();
        spec.model.num_layers = 1;
        spec.model.model_dim = 16;
        spec.model.num_heads = 2;
        spec.model.num_kv_heads = 1;
        spec.model.vocab_size = 64;
        spec.model.mlp_mult = 2.0;
        let report = plan_delta(&spec, 64 * 1024).unwrap();

        assert!(!report.selected_deltas.is_empty());
        assert!(!report.selected_low_rank_proofs.is_empty());
        assert!(report.score_first_legality_audit.pass);
        assert_eq!(
            report.domain_evaluation.kind,
            "deltagolf_equal_byte_domain_proxy"
        );
        assert_eq!(
            report.domain_evaluation.selected_beats_no_delta_proxy,
            Some(true)
        );
        assert!(
            report
                .domain_evaluation
                .rows
                .iter()
                .any(|row| row.plan_name == "selected_delta_plan"
                    && row.fits_comparison_budget
                    && row.score_first_legal)
        );
        for proof in &report.selected_low_rank_proofs {
            assert!(proof.finite);
            assert!(proof.curvature_positive);
            assert!(proof.actual_rank <= proof.requested_rank);
            assert_eq!(proof.factor_a_elems, proof.proof_rows * proof.actual_rank);
            assert_eq!(proof.factor_b_elems, proof.actual_rank * proof.proof_cols);
            assert!(proof.selected_rank_weighted_error < proof.zero_delta_weighted_error);
            assert!(proof.error_reduction_vs_zero_pct > 0.0);
            assert!(proof.selected_rank_no_worse_than_lower_rank);
        }
    }

    #[test]
    fn delta_dp_matches_bruteforce_multiple_choice_allocation() {
        let mut spec = RunSpec::default();
        spec.model.num_layers = 1;
        spec.model.model_dim = 12;
        spec.model.num_heads = 2;
        spec.model.num_kv_heads = 1;
        spec.model.vocab_size = 48;
        spec.model.mlp_mult = 2.0;
        let budget = 12 * 1024;

        let report = plan_delta(&spec, budget).unwrap();
        let brute = brute_force_delta_plan(&delta_candidate_sets(&spec), budget);

        assert_eq!(report.selected_bytes, brute.bytes);
        assert!(
            (report.selected_predicted_gain - brute.gain).abs() <= 1e-12,
            "DP gain {} != brute gain {}",
            report.selected_predicted_gain,
            brute.gain
        );
        assert_eq!(
            selected_delta_names(&report.selected_deltas),
            selected_delta_names(&brute.options)
        );
    }

    #[derive(Clone)]
    struct BruteDeltaPlan {
        bytes: usize,
        gain: f64,
        options: Vec<DeltaOptionReport>,
    }

    fn brute_force_delta_plan(
        families: &[Vec<DeltaOptionReport>],
        budget: usize,
    ) -> BruteDeltaPlan {
        fn visit(
            families: &[Vec<DeltaOptionReport>],
            family_index: usize,
            budget: usize,
            bytes: usize,
            gain: f64,
            options: &mut Vec<DeltaOptionReport>,
            best: &mut Option<BruteDeltaPlan>,
        ) {
            if family_index == families.len() {
                let candidate = BruteDeltaPlan {
                    bytes,
                    gain,
                    options: options
                        .iter()
                        .filter(|option| option.bytes > 0)
                        .cloned()
                        .collect(),
                };
                if best
                    .as_ref()
                    .map(|old| brute_delta_better(&candidate, old))
                    .unwrap_or(true)
                {
                    *best = Some(candidate);
                }
                return;
            }

            for option in &families[family_index] {
                let next_bytes = bytes.saturating_add(option.bytes);
                if next_bytes > budget {
                    continue;
                }
                options.push(option.clone());
                visit(
                    families,
                    family_index + 1,
                    budget,
                    next_bytes,
                    gain + option.predicted_domain_gain,
                    options,
                    best,
                );
                options.pop();
            }
        }

        let mut best = None;
        visit(families, 0, budget, 0, 0.0, &mut Vec::new(), &mut best);
        best.expect("delta candidate families always contain zero-byte choices")
    }

    fn brute_delta_better(candidate: &BruteDeltaPlan, old: &BruteDeltaPlan) -> bool {
        candidate.gain > old.gain + 1e-12
            || ((candidate.gain - old.gain).abs() <= 1e-12 && candidate.bytes < old.bytes)
    }

    fn selected_delta_names(options: &[DeltaOptionReport]) -> Vec<String> {
        let mut names = options
            .iter()
            .map(|option| option.name.clone())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn pack_lqer_proof_reduces_residual_and_bound() {
        let mut spec = RunSpec::default();
        spec.quant.lqer.enabled = true;
        spec.quant.lqer.rank = 2;
        let group = ModelGolfTensorGroup {
            name: "mlp_down_bank".to_string(),
            role: TensorRole::MlpDown,
            rows: 64,
            cols: 64,
            elems: 4096,
            current_bits: 4,
            current_weight_bytes: 2048,
            scale_bytes: 128,
            lqer_bytes: 512,
            sensitivity: role_sensitivity(TensorRole::MlpDown),
        };
        let selected = PrecisionOptionReport {
            group: group.name.clone(),
            role: group.role,
            bits: 4,
            bytes: 4096,
            weight_bytes: 2048,
            scale_bytes: 128,
            lqer_bytes: 512,
            estimated_quality_loss: 0.1,
            runtime_kernel: "unit_test_lqer".to_string(),
            residual_correction: Some("rank2 selective_lqer".to_string()),
        };

        let proofs = prove_selected_lqer_groups(&[group], &spec, &[selected]);
        assert_eq!(proofs.len(), 1);
        let proof = &proofs[0];
        assert!(proof.finite);
        assert_eq!(proof.actual_rank, 2);
        assert!(proof.residual_frobenius_after < proof.residual_frobenius_before);
        assert!(proof.ce_linf_bound_after < proof.ce_linf_bound_before);
        assert!(proof.residual_reduction_pct > 0.0);
        assert!(proof.ce_bound_reduction_pct > 0.0);
        assert!(proof.selected_rank_no_worse_than_lower_rank);
    }

    #[test]
    fn pack_quality_comparison_rewards_lqer_residual_proxy() {
        let mut spec = RunSpec::default();
        spec.quant.lqer.enabled = true;
        spec.quant.lqer.rank = 2;
        spec.quant.lqer.top_k = 1;
        spec.quant.compression = pg_model::CompressionMode::None;
        let group = ModelGolfTensorGroup {
            name: "mlp_down_bank".to_string(),
            role: TensorRole::MlpDown,
            rows: 64,
            cols: 64,
            elems: 4096,
            current_bits: 4,
            current_weight_bytes: 2048,
            scale_bytes: 128,
            lqer_bytes: 512,
            sensitivity: role_sensitivity(TensorRole::MlpDown),
        };
        let option_sets = vec![precision_options_for_group(&group, &spec, None)];
        let selected = option_sets[0]
            .iter()
            .find(|option| option.bits == 4 && option.lqer_bytes > 0)
            .cloned()
            .expect("4-bit LQER option should exist");
        let selected_bytes = selected.bytes;

        let comparison =
            compare_pack_quality_plans(&[group], &spec, &option_sets, &[selected], selected_bytes);
        let selected_row = comparison
            .rows
            .iter()
            .find(|row| row.plan_name == comparison.selected_plan_name)
            .expect("selected comparison row");
        let uniform_q4 = comparison
            .rows
            .iter()
            .find(|row| row.plan_name == "uniform_q4_no_lqer")
            .expect("uniform q4 row");

        assert_eq!(selected_row.lqer_group_count, 1);
        assert!(selected_row.fits_comparison_budget);
        assert!(
            selected_row.local_weighted_residual_mse_proxy
                < uniform_q4.local_weighted_residual_mse_proxy
        );
        assert!(
            selected_row.local_activation_ce_bound_proxy
                < uniform_q4.local_activation_ce_bound_proxy
        );
        assert_eq!(comparison.selected_beats_uniform_best_proxy, Some(true));
    }

    #[test]
    fn pack_lqer_candidates_respect_enable_flag_and_effective_rank() {
        let mut spec = RunSpec::default();
        spec.quant.lqer.enabled = false;
        spec.quant.lqer.rank = 12;
        spec.quant.lqer.top_k = 3;
        let group = ModelGolfTensorGroup {
            name: "tok_emb".to_string(),
            role: TensorRole::TokenEmbedding,
            rows: 2,
            cols: 3,
            elems: 6,
            current_bits: 4,
            current_weight_bytes: 3,
            scale_bytes: 4,
            lqer_bytes: 0,
            sensitivity: role_sensitivity(TensorRole::TokenEmbedding),
        };

        let disabled_options = precision_options_for_group(&group, &spec, None);
        assert!(disabled_options.iter().all(|option| option.lqer_bytes == 0));
        assert!(lqer_candidates(std::slice::from_ref(&group), &spec, &[], None).is_empty());

        spec.quant.lqer.enabled = true;
        let rank_limited_loss = quality_loss_surrogate(&group, 4, true, 2);
        let over_requested_loss = quality_loss_surrogate(&group, 4, true, spec.quant.lqer.rank);
        assert_eq!(
            effective_lqer_rank(group.rows, group.cols, spec.quant.lqer.rank),
            2
        );
        assert!(
            (rank_limited_loss - over_requested_loss).abs() < 1e-12,
            "LQER quality correction must be limited by matrix rank, got {rank_limited_loss} vs {over_requested_loss}"
        );

        let candidates = lqer_candidates(&[group], &spec, &[], None);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].rank, 12);
        assert_eq!(candidates[0].effective_rank, 2);
        assert!(candidates[0].predicted_loss_reduction > 0.0);
    }

    #[test]
    fn pack_dp_matches_bruteforce_multiple_choice_knapsack() {
        fn option(group: &str, bytes: usize, loss: f64) -> PrecisionOptionReport {
            PrecisionOptionReport {
                group: group.to_string(),
                role: TensorRole::Other,
                bits: 4,
                bytes,
                weight_bytes: bytes,
                scale_bytes: 0,
                lqer_bytes: 0,
                estimated_quality_loss: loss,
                runtime_kernel: "unit".to_string(),
                residual_correction: None,
            }
        }

        fn brute_force(
            sets: &[Vec<PrecisionOptionReport>],
            budget: usize,
            group_index: usize,
            bytes: usize,
            loss: f64,
        ) -> Option<(usize, f64)> {
            if group_index == sets.len() {
                return Some((bytes, loss));
            }
            sets[group_index]
                .iter()
                .filter_map(|candidate| {
                    let next_bytes = bytes + candidate.bytes;
                    if next_bytes > budget {
                        return None;
                    }
                    brute_force(
                        sets,
                        budget,
                        group_index + 1,
                        next_bytes,
                        loss + candidate.estimated_quality_loss,
                    )
                })
                .min_by(|a, b| compare_loss_then_bytes(a.1, a.0, b.1, b.0))
        }

        let option_sets = vec![
            vec![
                option("a", 6, 1.0),
                option("a", 4, 5.0),
                option("a", 2, 9.0),
            ],
            vec![
                option("b", 5, 1.0),
                option("b", 3, 4.0),
                option("b", 1, 8.0),
            ],
            vec![
                option("c", 5, 1.0),
                option("c", 2, 6.0),
                option("c", 1, 9.0),
            ],
        ];
        let budget = 12;
        let (feasible, selected, selected_bytes, selected_loss) =
            select_pack_options(&option_sets, budget);
        let brute = brute_force(&option_sets, budget, 0, 0, 0.0).unwrap();

        assert!(feasible);
        assert_eq!(selected.len(), option_sets.len());
        assert_eq!(selected_bytes, brute.0);
        assert!((selected_loss - brute.1).abs() < 1e-12);
        assert!(selected_bytes <= budget);
    }

    #[test]
    fn pack_artifact_proof_exports_and_strict_reloads() {
        let dir = temp_path("pack_proof");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let artifact_path = dir.join("modelgolf_proof.pgrs");
        let output = dir.join("pack.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::Pack,
            spec: spec_path,
            output: Some(output),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: None,
            latency_target_ms: None,
            quality_budget_ppl_pct: None,
            context: None,
            batch: None,
            artifact_budget_bytes: None,
            delta_budget_bytes: None,
            proof_artifact: Some(artifact_path.clone()),
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let proof = report
            .pack
            .artifact_proof
            .expect("pack proof should be present");
        assert!(artifact_path.exists());
        assert!(proof.artifact_bytes > 0);
        assert!(proof.artifact_budget_ok);
        assert!(proof.strict_reload_ok);
        assert!(proof.finite_loss);
        assert!(proof.pre_export_loss.is_finite());
        assert!(proof.post_reload_loss.is_finite());
        assert_eq!(proof.smoke_tokens, 16);
    }

    #[test]
    fn pack_experiment_emits_measured_lqer_table() {
        let dir = temp_path("pack_experiment");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("pack_experiment.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::PackExperiment,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: None,
            latency_target_ms: None,
            quality_budget_ppl_pct: None,
            context: None,
            batch: None,
            artifact_budget_bytes: Some(128 * 1024),
            delta_budget_bytes: None,
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let experiment = report.pack_experiment.as_ref().unwrap();
        assert_eq!(experiment.kind, "modelgolf_pack_measured_lqer_experiment");
        assert_eq!(experiment.rows.len(), 4);
        assert_eq!(experiment.target_plan_name, "mixed_plus_lqer_top3");
        assert_eq!(experiment.status, "local_measured_experiment_ready");
        assert!(experiment.best_measured_bpb_proxy_plan.is_some());
        assert!(experiment.rows.iter().all(|row| {
            row.finite
                && row.post_reload_loss.is_finite()
                && row.measured_bpb_proxy.is_finite()
                && row.local_decode_tokens_per_second.is_finite()
        }));
        let target = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == experiment.target_plan_name)
            .unwrap();
        assert!(target.lqer_enabled);
        assert_eq!(target.lqer_groups.len(), 3);
        let control = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == "mixed_plus_random_lqer_control")
            .unwrap();
        assert_eq!(control.lqer_selection_policy, "deterministic_hash_control");
        assert_eq!(control.lqer_groups.len(), 3);
        assert!(output.exists());
    }

    #[test]
    fn kernel_experiment_emits_exact_tiled_ce_table() {
        let dir = temp_path("kernel_experiment");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("kernel_experiment.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::KernelExperiment,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: None,
            latency_target_ms: None,
            quality_budget_ppl_pct: None,
            context: None,
            batch: None,
            artifact_budget_bytes: None,
            delta_budget_bytes: None,
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let experiment = report.kernel_experiment.as_ref().unwrap();
        assert_eq!(experiment.kind, "kernelforge_exact_tiled_ce_experiment");
        assert_eq!(experiment.rows.len(), 2);
        assert_eq!(experiment.status, "local_experiment_ready");
        assert!(experiment.full_vs_tiled.parity_ok);
        assert!(experiment.full_vs_tiled.tiled_uses_less_scratch);
        assert!(
            experiment
                .rows
                .iter()
                .all(|row| row.forward_ms_cpu.is_finite()
                    && row.backward_ms_cpu.is_finite()
                    && row.total_ms_cpu.is_finite()
                    && row.loss_mean.is_finite())
        );
        let full = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == "full_logits_reference")
            .unwrap();
        let tiled = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == "tiled_logit_free_ce")
            .unwrap();
        assert!(full.materializes_persistent_logits);
        assert!(!tiled.materializes_persistent_logits);
        assert!(tiled.total_ce_scratch_bytes_f32 < full.total_ce_scratch_bytes_f32);
        assert!(output.exists());
    }

    #[test]
    fn wind_experiment_ranks_candidates_and_evaluates_top_plans() {
        let dir = temp_path("wind_experiment");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("wind_experiment.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::WindExperiment,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: Some(16 * 1024 * 1024),
            latency_target_ms: None,
            quality_budget_ppl_pct: None,
            context: Some(64),
            batch: Some(2),
            artifact_budget_bytes: Some(128 * 1024),
            delta_budget_bytes: None,
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let experiment = report.wind_experiment.as_ref().unwrap();
        assert_eq!(
            experiment.kind,
            "modelgolf_wind_experiment_candidate_ranking"
        );
        assert_eq!(experiment.candidate_count, 24);
        assert_eq!(experiment.cheap_ranked_count, 24);
        assert_eq!(experiment.full_eval_top_k, 8);
        assert_eq!(experiment.full_evaluated_count, 8);
        assert!(experiment.holdout_spearman.is_some_and(f64::is_finite));
        assert!(experiment.mean_abs_rank_error.is_some());
        let calibration_pass = experiment.holdout_spearman.unwrap() >= 0.50;
        assert_eq!(experiment.calibration_pass, calibration_pass);
        assert_eq!(
            experiment.status,
            if calibration_pass {
                "local_wind_experiment_ready"
            } else {
                "local_wind_experiment_needs_calibration"
            }
        );
        assert!(experiment.best_cheap_candidate.is_some());
        assert!(experiment.best_full_candidate.is_some());
        assert!(experiment.rows.windows(2).all(|pair| {
            pair[0].cheap_rank < pair[1].cheap_rank
                && (!pair[1].estimated_artifact_budget_fit || pair[0].estimated_artifact_budget_fit)
                && (pair[0].estimated_artifact_budget_fit != pair[1].estimated_artifact_budget_fit
                    || pair[0].cheap_score <= pair[1].cheap_score + 1e-12)
        }));
        assert!(experiment.rows.iter().take(8).all(|row| {
            row.full_evaluated
                && row.full_eval_score.is_some_and(f64::is_finite)
                && row.full_eval_bpb_proxy.is_some_and(f64::is_finite)
                && row
                    .full_eval_decode_tokens_per_second
                    .is_some_and(f64::is_finite)
                && row.evaluated_cheap_rank.is_some()
                && row.full_rank.is_some()
                && row.rank_error.is_some()
        }));
        assert!(
            experiment
                .rows
                .iter()
                .skip(8)
                .all(|row| !row.full_evaluated)
        );
        let serialized: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output).unwrap()).unwrap();
        assert_eq!(
            serialized["kind"].as_str(),
            Some("modelgolf_wind_experiment_candidate_ranking")
        );
    }

    #[test]
    fn wind_full_proxy_penalizes_deployment_budget_overflow() {
        let dir = temp_path("wind_budget_proxy");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let spec = tiny_spec(&spec_path);
        let config = pack_experiment_model_config(&spec);
        let mut model = GptModel::new(config.clone());
        model.fill_deterministic();
        let (input_ids, targets) = artifact_proof_tokens(config.vocab_size, config.eval_seq_len);
        let pre_export_loss = model_smoke_loss(&model, &input_ids, &targets).unwrap();

        let feasible = WindExperimentCandidate {
            name: "feasible_budget_candidate".to_string(),
            quant_spec: spec.quant.clone(),
            cache_k_bits: 4,
            cache_v_bits: 4,
            delta_budget_bytes: 0,
            estimated_artifact_bytes: 1024,
            estimated_artifact_budget_fit: true,
            estimated_cache_bytes: 1024,
            estimated_memory_budget_fit: Some(true),
            estimated_delta_bytes: 0,
            deployment_artifact_ratio: 0.10,
            deployment_cache_ratio: 0.10,
            deployment_artifact_overflow_ratio: 0.0,
            deployment_memory_overflow_ratio: 0.0,
            cheap_score: 0.0,
            cheap_quality_loss: 0.0,
            cheap_cache_error_proxy: 0.0,
            cheap_delta_gain: 0.0,
        };
        let overflow = WindExperimentCandidate {
            name: "overflow_budget_candidate".to_string(),
            estimated_artifact_bytes: 4096,
            estimated_artifact_budget_fit: false,
            deployment_artifact_ratio: 2.0,
            deployment_artifact_overflow_ratio: 1.0,
            ..feasible.clone()
        };

        let feasible_eval = wind_experiment_full_eval(
            &model,
            &input_ids,
            &targets,
            pre_export_loss,
            "wind_budget_proxy",
            &feasible,
        )
        .unwrap();
        let overflow_eval = wind_experiment_full_eval(
            &model,
            &input_ids,
            &targets,
            pre_export_loss,
            "wind_budget_proxy",
            &overflow,
        )
        .unwrap();

        assert_eq!(feasible_eval.artifact_bytes, overflow_eval.artifact_bytes);
        assert!(overflow_eval.score > feasible_eval.score + 90.0);
    }

    #[test]
    fn cache_experiment_emits_kv_bit_grid() {
        let dir = temp_path("cache_experiment");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_spec(&spec_path);
        let output = dir.join("cache_experiment.json");

        let report = run_modelgolf_plan(ModelGolfOptions {
            section: ModelGolfSection::CacheExperiment,
            spec: spec_path,
            output: Some(output.clone()),
            hardware: "unit_test".to_string(),
            runtime: "portable_cpu".to_string(),
            memory_budget_bytes: None,
            latency_target_ms: None,
            quality_budget_ppl_pct: None,
            context: Some(32),
            batch: Some(2),
            artifact_budget_bytes: None,
            delta_budget_bytes: None,
            proof_artifact: None,
            quality_calibration: None,
            release_evidence: None,
            release_artifact: None,
            validation_dataset_id: None,
            validation_command: None,
            heldout_bpb: None,
            baseline_bpb: None,
            decode_tokens_per_second: None,
            backend_id: None,
            kernel_id: None,
            long_context_dataset_id: None,
            fused_runtime: None,
            parity_pass: None,
            long_context_bpb_delta_pct: None,
            speedup_x: None,
            calibration_dataset_id: None,
            production_svd_validated: None,
            calibrated_tensor_sensitivity: None,
            equal_byte_bpb_delta: None,
            domain_dataset_id: None,
            trained_delta_bytes: None,
            legality_pass: None,
            score_first_trace_or_review: None,
            equal_byte_domain_bpb_delta: None,
            training_run_id: None,
            gpu_backend_integrated: None,
            proxy_calibrated: None,
            post_export_bpb_delta_vs_posthoc: None,
            measurement_run_id: None,
            power_meter_id: None,
            wall_time_seconds: None,
            average_power_watts: None,
            energy_joules: None,
            telemetry_validated: None,
            distributed_backend_id: None,
            reduce_scatter_parity_pass: None,
            all_gather_parity_pass: None,
            optimizer_update_parity_pass: None,
            nccl_trace_validated: None,
            overlap_validated: None,
            measured_comm_time_ms: None,
            measured_step_time_ms: None,
            communication_speedup_x: None,
            generated_kernel_ids: None,
            generated_kernels: None,
            memory_reduction_x: None,
            trace_corpus_id: None,
            calibration_report_id: None,
            fresh_profiler_traces: None,
            external_timing_validated: None,
            holdout_spearman: None,
            mean_abs_pct_error: None,
            evidence_source: None,
            source_report_dir: None,
            evidence_id: None,
            generated_at: None,
        })
        .unwrap();

        let experiment = report.cache_experiment.as_ref().unwrap();
        assert_eq!(experiment.kind, "cachegolf_kv_bit_grid_experiment");
        assert_eq!(experiment.rows.len(), 25);
        assert_eq!(experiment.status, "local_experiment_ready");
        assert!(experiment.all_bounds_cover_observed_error);
        assert!(experiment.best_observed_error_plan.is_some());
        assert!(experiment.best_bound_tightness_plan.is_some());
        let budgeted_plan = experiment.best_budgeted_plan.as_ref().unwrap();
        assert_eq!(
            experiment.long_context_memory.kind,
            "cachegolf_kv_bit_grid_long_context_memory"
        );
        assert!(!experiment.long_context_memory.rows.is_empty());
        assert!(experiment.target_cache_budget_bytes > 0);
        assert!(experiment.target_cache_budget_pct_of_fp16.is_finite());
        assert!(
            experiment
                .long_context_memory
                .rows
                .iter()
                .any(|row| row.selected_grid_plan_name == *budgeted_plan
                    && row.context_tokens == 32
                    && row.fits_target_cache_budget)
        );
        assert!(experiment.rows.iter().all(|row| {
            row.bound_covers_observed_error
                && row.dequant_attention_parity_ok
                && row.max_attention_error_bound.is_finite()
                && row.max_observed_attention_l2_error.is_finite()
                && row.bound_to_observed_ratio.is_finite()
                && row.compression_ratio_single_layer > 0.0
        }));
        let low = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == "k2_v2")
            .unwrap();
        let high = experiment
            .rows
            .iter()
            .find(|row| row.plan_name == "k8_v8")
            .unwrap();
        assert!(high.max_observed_attention_l2_error <= low.max_observed_attention_l2_error);
        assert!(output.exists());
    }

    #[test]
    fn pack_dp_spends_bits_on_sensitive_groups_under_budget() {
        let mut spec = RunSpec::default();
        spec.quant.lqer.enabled = false;
        spec.quant.compression = pg_model::CompressionMode::None;
        let sensitive = ModelGolfTensorGroup {
            name: "tok_emb".to_string(),
            role: TensorRole::TokenEmbedding,
            rows: 100,
            cols: 100,
            elems: 10_000,
            current_bits: 8,
            current_weight_bytes: 10_000,
            scale_bytes: 200,
            lqer_bytes: 0,
            sensitivity: role_sensitivity(TensorRole::TokenEmbedding),
        };
        let cheap = ModelGolfTensorGroup {
            name: "mlp_up_bank".to_string(),
            role: TensorRole::MlpUp,
            rows: 100,
            cols: 100,
            elems: 10_000,
            current_bits: 8,
            current_weight_bytes: 10_000,
            scale_bytes: 200,
            lqer_bytes: 0,
            sensitivity: role_sensitivity(TensorRole::MlpUp),
        };
        let target = 5 * 4096;
        let report = plan_pack(&[sensitive, cheap], &spec, target, None);
        assert!(report.feasible);
        assert!(!report.quality_calibration.applied);
        assert_eq!(report.bucket_bytes, 1);
        assert_eq!(
            report.algorithm,
            "exact_multiple_choice_knapsack_plus_lqer_ranking"
        );
        let emb_bits = report
            .selected_options
            .iter()
            .find(|option| option.group == "tok_emb")
            .unwrap()
            .bits;
        let mlp_bits = report
            .selected_options
            .iter()
            .find(|option| option.group == "mlp_up_bank")
            .unwrap()
            .bits;
        assert!(emb_bits >= mlp_bits);
    }

    #[test]
    fn pack_quality_calibration_scales_group_loss() {
        let dir = temp_path("pack_calibration");
        fs::create_dir_all(&dir).unwrap();
        let calibration_path = dir.join("quality_calibration.json");
        let mut spec = RunSpec::default();
        spec.quant.lqer.enabled = false;
        let group = ModelGolfTensorGroup {
            name: "tok_emb".to_string(),
            role: TensorRole::TokenEmbedding,
            rows: 100,
            cols: 100,
            elems: 10_000,
            current_bits: 8,
            current_weight_bytes: 10_000,
            scale_bytes: 200,
            lqer_bytes: 0,
            sensitivity: role_sensitivity(TensorRole::TokenEmbedding),
        };
        let base_loss = quality_loss_surrogate(&group, 4, false, spec.quant.lqer.rank);
        let measured_loss = base_loss * 2.5;
        fs::write(
            &calibration_path,
            format!(r#"[{{"group":"tok_emb","bits":4,"measured_quality_loss":{measured_loss}}}]"#),
        )
        .unwrap();

        let calibration =
            load_pack_quality_calibration(&calibration_path, std::slice::from_ref(&group), &spec)
                .unwrap();
        let options = precision_options_for_group(&group, &spec, Some(&calibration));
        let calibrated = options
            .iter()
            .find(|option| option.bits == 4 && option.lqer_bytes == 0)
            .unwrap();

        assert!(calibration.report.applied);
        assert_eq!(calibration.report.points, 1);
        assert!(
            (calibrated.estimated_quality_loss - measured_loss).abs() < 1e-12,
            "calibrated loss {} should match measured {}",
            calibrated.estimated_quality_loss,
            measured_loss
        );
        assert!(
            calibration
                .report
                .mean_abs_error_after
                .expect("after error should be present")
                < calibration
                    .report
                    .mean_abs_error_before
                    .expect("before error should be present")
        );
    }

    #[test]
    fn cache_bound_improves_with_more_bits() {
        let low = attention_error_bound(64, 2, 2);
        let high = attention_error_bound(64, 6, 6);
        assert!(high < low);
    }

    #[test]
    fn cache_planner_bytes_match_reference_format_for_contiguous_policy() {
        use pg_kernels::cachegolf::{CacheGolfLayout, cachegolf_quantize_kv};

        let mut spec = RunSpec::default();
        spec.model.num_layers = 3;
        spec.model.num_kv_heads = 2;
        spec.model.num_heads = 4;
        spec.model.model_dim = 16;
        let context_tokens = 7;
        let batch_sequences = 1;
        let report = plan_cache(&spec, context_tokens, batch_sequences, Some(usize::MAX)).unwrap();
        assert_eq!(report.selected.layout, "contiguous");
        assert!(report.selected_policy_proof.bound_covers_observed_error);
        assert!(report.selected_policy_proof.dequant_attention_parity_ok);
        assert_eq!(
            report.residual_sketch_policy.kind,
            "cachegolf_residual_sketch_policy"
        );
        assert_eq!(
            report.eviction_policy.policy,
            "sink_plus_recent_protected_bound_aware_eviction"
        );
        assert!(!report.eviction_policy.eviction_needed_for_budget);
        assert!(
            report
                .long_context_eval
                .rows
                .iter()
                .any(|row| row.context_tokens == context_tokens)
        );
        assert!(report.long_context_eval.selected_context_fits_budget);
        assert!(report.long_context_eval.max_context_tokens_under_budget >= context_tokens);

        let head_dim = spec.model.to_model_config().head_dim;
        let elems = context_tokens * spec.model.num_kv_heads * head_dim;
        let k = vec![0.0f32; elems];
        let v = vec![0.0f32; elems];
        let cache = cachegolf_quantize_kv(
            &k,
            &v,
            context_tokens,
            spec.model.num_kv_heads,
            head_dim,
            report.selected.k_bits,
            report.selected.v_bits,
            report.selected.block_size_tokens,
            CacheGolfLayout::Contiguous,
        )
        .unwrap();
        assert_eq!(
            report.selected.estimated_cache_bytes,
            cache.f16_scale_artifact_bytes() * spec.model.num_layers
        );
        let helper_bytes = cache_policy_estimated_bytes_for_context(
            &spec.model.to_model_config(),
            context_tokens,
            batch_sequences,
            &report.selected,
        );
        assert_eq!(helper_bytes, report.selected.estimated_cache_bytes);
    }

    #[test]
    fn cache_planner_bytes_match_paged_reference_format() {
        use pg_kernels::cachegolf::{CacheGolfLayout, cachegolf_quantize_kv};

        let mut spec = RunSpec::default();
        spec.model.num_layers = 2;
        spec.model.num_kv_heads = 2;
        spec.model.num_heads = 4;
        spec.model.model_dim = 16;
        let cfg = spec.model.to_model_config();
        let context_tokens = 129;
        let batch_sequences = 1;
        let block_size_tokens = 64;
        let k_bits = 3;
        let v_bits = 4;
        let elems = context_tokens * spec.model.num_kv_heads * cfg.head_dim;
        let k = vec![0.0f32; elems];
        let v = vec![0.0f32; elems];

        let cache = cachegolf_quantize_kv(
            &k,
            &v,
            context_tokens,
            spec.model.num_kv_heads,
            cfg.head_dim,
            k_bits,
            v_bits,
            block_size_tokens,
            CacheGolfLayout::Paged,
        )
        .unwrap();
        let estimated = cache_policy_estimated_bytes_for_dims(
            spec.model.num_layers,
            spec.model.num_kv_heads,
            cfg.head_dim,
            context_tokens,
            batch_sequences,
            block_size_tokens,
            "paged",
            k_bits,
            v_bits,
        );

        assert_eq!(cache.stored_tokens(), 192);
        assert_eq!(
            estimated,
            cache.f16_scale_runtime_bytes() * spec.model.num_layers
        );
        assert_eq!(
            cache_stored_tokens_for_layout(context_tokens, 128, "paged"),
            256
        );
        assert_ne!(
            estimated,
            cache_policy_estimated_bytes_for_dims(
                spec.model.num_layers,
                spec.model.num_kv_heads,
                cfg.head_dim,
                context_tokens,
                batch_sequences,
                128,
                "paged",
                k_bits,
                v_bits,
            )
        );
    }
}
