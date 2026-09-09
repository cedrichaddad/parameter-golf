use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pg_core::{PgError, PgResult};
use pg_data::bpb::compute_bpb;
use pg_model::{ExecutionPlan, GptModel, RecurrentBackwardProfile, RunSpec};
use pg_optim::muon::Muon;
use pg_quant::{CompiledQuantKernelSet, compile_quant_layout_manifest};
use serde::{Deserialize, Serialize};

pub mod modelgolf;
pub use modelgolf::{ModelGolfOptions, ModelGolfReport, ModelGolfSection, run_modelgolf_plan};

#[cfg(feature = "metal_apple")]
mod metal_lite;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyOptions {
    pub spec: PathBuf,
    pub artifact: Option<PathBuf>,
    pub artifact_audit: Option<PathBuf>,
    pub score_first_log: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub kind: &'static str,
    pub spec_path: String,
    pub spec_name: String,
    pub spec_fingerprint: String,
    pub artifact_manifest_ok: bool,
    pub artifact_budget_known: bool,
    pub artifact_budget_ok: bool,
    pub caseops_sidecar_ok: bool,
    pub data_preflight: DataPreflightReport,
    pub eval_legality: EvalLegalityReport,
    pub bpb_smoke_ok: bool,
    pub score_first_legal: bool,
    pub artifact_audit_ok: bool,
    pub score_first_log_ok: bool,
    pub proposal_claims_ok: bool,
    pub proposal_features: ProposalFeatureReport,
    pub notes: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataPreflightReport {
    pub kind: &'static str,
    pub canonical_target: bool,
    pub train_pattern: Option<String>,
    pub validation_pattern: Option<String>,
    pub caseops_sidecar_pattern: Option<String>,
    pub caseops_enabled: bool,
    pub caseops_byte_sidecar_required: bool,
    pub train_shards_found: usize,
    pub train_shards_required: Option<usize>,
    pub train_shards_ok: Option<bool>,
    pub validation_shards_found: usize,
    pub validation_tokens_found: Option<usize>,
    pub validation_tokens_required: Option<usize>,
    pub validation_tokens_ok: Option<bool>,
    pub validation_docs_found: Option<usize>,
    pub validation_docs_required: Option<usize>,
    pub validation_docs_ok: Option<bool>,
    pub sidecar_shards_found: usize,
    pub sidecar_tokens_found: Option<usize>,
    pub sidecar_tokens_match_validation: Option<bool>,
    pub sidecar_structural_ok: bool,
    pub ready: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalLegalityReport {
    pub kind: &'static str,
    pub bpb_byte_accounting_ok: bool,
    pub score_before_update_order_ok: bool,
    pub score_first_tokens_scored: usize,
    pub score_first_update_count: usize,
    pub score_first_loss_not_worse_than_static: bool,
    pub no_future_token_access: bool,
    pub document_boundary_reset_ok: bool,
    pub artifact_decode_eval_equivalence_ok: bool,
    pub artifact_fingerprint: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProposalFeatureReport {
    pub bf16_direct_compact: String,
    pub device_resident_sampler: String,
    pub no_loss_backward_graph: String,
    pub full_train_step_graph: String,
    pub persistent_cta_block_backward: String,
    pub xsa_inside_sdpa: String,
    pub adjacent_sparse_xsa: String,
    pub quant_layout_compiler: String,
    pub bigramhash_fusion: String,
    pub pg_lite: String,
    pub pg_lite_metal_backend: String,
    pub wind_tunnel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundleOptions {
    pub spec: PathBuf,
    pub artifact: Option<PathBuf>,
    pub trace: Option<PathBuf>,
    pub trace_dir: Option<PathBuf>,
    pub lite_config: PathBuf,
    pub lite_config_dir: Option<PathBuf>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundleReport {
    pub kind: &'static str,
    pub output_dir: String,
    pub readme_path: String,
    pub evidence_manifest_path: String,
    pub verify_path: String,
    pub dist_sim_path: String,
    pub artifact_lab_path: String,
    pub modelgolf_path: String,
    pub wind_tunnel_path: String,
    pub trace_corpus_path: String,
    pub calibration_model_path: String,
    pub scenario_report_path: String,
    pub lite_report_path: String,
    pub lite_benchmark_summary_path: String,
    pub lite_suite_summary_path: String,
    pub wind_tunnel_reviewer_report_path: String,
    pub backend_cpu_check_path: String,
    pub backend_metal_check_path: String,
    pub proposal_feature_path: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceManifestReport {
    pub kind: &'static str,
    pub package_kind: &'static str,
    pub output_dir: String,
    pub record_claim: bool,
    pub leaderboard_claim: bool,
    pub proof_bundle_status: String,
    pub component_statuses: Vec<EvidenceComponentStatus>,
    pub generated_files: Vec<EvidenceFileReport>,
    pub local_evidence: Vec<EvidenceClaimReport>,
    pub requires_remote_validation: Vec<EvidenceClaimReport>,
    pub blocking_reasons: Vec<String>,
    pub caveats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceComponentStatus {
    pub name: String,
    pub status: String,
    pub path: String,
    pub decisive_for_local_bundle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceFileReport {
    pub path: String,
    pub kind: String,
    pub required: bool,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceClaimReport {
    pub name: String,
    pub status: String,
    pub evidence_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistSimOptions {
    pub spec: PathBuf,
    pub world_size: usize,
    pub steps: usize,
    pub seed: u64,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistSimReport {
    pub kind: &'static str,
    pub spec_fingerprint: String,
    pub world_size: usize,
    pub steps: usize,
    pub parameter_shapes: Vec<[usize; 3]>,
    pub bank_reports: Vec<DistSimBankReport>,
    pub reduce_scatter_equivalent: bool,
    pub all_gather_equivalent: bool,
    pub optimizer_parity_max_abs_diff: f32,
    pub tolerance: f32,
    pub estimated_reduce_scatter_bytes_per_step: usize,
    pub estimated_all_gather_bytes_per_step: usize,
    pub bf16_shadow_freshness: &'static str,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistSimBankReport {
    pub name: String,
    pub shape: [usize; 3],
    pub simulated_shape: [usize; 3],
    pub reduce_scatter_bytes_per_step: usize,
    pub all_gather_bytes_per_step: usize,
    pub optimizer_parity_max_abs_diff: f32,
    pub equivalent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLabOptions {
    pub spec: PathBuf,
    pub artifact: Option<PathBuf>,
    pub sweep: Vec<String>,
    pub mini_train: Option<PathBuf>,
    pub mini_val: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLabReport {
    pub kind: &'static str,
    pub spec_fingerprint: String,
    pub artifact_path: Option<String>,
    pub artifact_kind: String,
    pub requested_sweeps: Vec<String>,
    pub artifact_manifest_ok: bool,
    pub artifact_budget_known: bool,
    pub artifact_budget_ok: bool,
    pub model_bytes: Option<usize>,
    pub code_bytes_estimate: usize,
    pub total_bytes_estimate: Option<usize>,
    pub target_artifact_bytes: usize,
    pub layout_manifest_crc32: String,
    pub quant_kernel_ids: Vec<String>,
    pub groups: Vec<ArtifactGroupReport>,
    pub mixed_bit_allocations: Vec<ArtifactMixedBitAllocationReport>,
    pub sweeps: Vec<ArtifactSweepReport>,
    pub mini_bpb: Option<ArtifactMiniBpbReport>,
    pub pg_lite_artifact: Option<ArtifactLabLiteArtifactReport>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLabLiteArtifactReport {
    pub kind: &'static str,
    pub source_config_fingerprint: String,
    pub model_fingerprint: String,
    pub model_family: LiteModelFamily,
    pub context: usize,
    pub residual_weight: f64,
    pub bigram_rows: usize,
    pub residual_buckets: usize,
    pub bigram_entries: usize,
    pub residual_entries: usize,
    pub local_proxy_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactGroupReport {
    pub name: String,
    pub bits: u8,
    pub rows: Option<usize>,
    pub cols: Option<usize>,
    pub packed_weight_bytes: Option<usize>,
    pub scale_bytes: Option<usize>,
    pub lqer_bytes: Option<usize>,
    pub pack_kernel: String,
    pub dequant_kernel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSweepReport {
    pub name: String,
    pub sweep_kind: String,
    pub bits: Option<u8>,
    pub bytes: usize,
    pub reconstruction_mse: f64,
    pub reconstruction_max_abs: f32,
    pub proxy_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMixedBitAllocationReport {
    pub name: String,
    pub matrix_bits: u8,
    pub mlp_bits: u8,
    pub embed_bits: u8,
    pub attn_gate_bits: u8,
    pub estimated_weight_bytes: usize,
    pub scale_bytes: usize,
    pub lqer_bytes: usize,
    pub total_bytes_estimate: usize,
    pub target_artifact_bytes: usize,
    pub budget_ok: bool,
    pub delta_vs_current_bytes: Option<isize>,
    pub proxy_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMiniBpbReport {
    pub train_path: String,
    pub val_path: String,
    pub train_bytes: usize,
    pub val_bytes: usize,
    pub model_family: LiteModelFamily,
    pub validation_bpb: f64,
    pub score_scope: &'static str,
    pub proxy_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCheckOptions {
    pub backend: LiteBackendKind,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCheckReport {
    pub kind: &'static str,
    pub backend: LiteBackendKind,
    pub host_os: String,
    pub cargo_feature_enabled: bool,
    pub kernel_source_path: Option<String>,
    pub kernel_source_present: bool,
    pub kernel_source_crc32: Option<String>,
    pub expected_kernel_symbols: Vec<String>,
    pub kernel_symbols_found: Vec<String>,
    pub kernel_contracts: Vec<BackendKernelContractReport>,
    pub kernel_contract_ok: bool,
    pub metal_compiler_path: Option<String>,
    pub metal_compiler_available: bool,
    pub metal_compiler_error: Option<String>,
    pub metal_compile_smoke_attempted: bool,
    pub metal_compile_smoke_ok: bool,
    pub metal_compile_stderr: Option<String>,
    pub rust_runtime_linked: bool,
    pub executable_backend_available: bool,
    pub notes: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendKernelContractReport {
    pub symbol: String,
    pub role: String,
    pub optimization_note: String,
    pub executable_required_for_acceleration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelOptions {
    pub spec: PathBuf,
    pub trace: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelReport {
    pub kind: &'static str,
    pub spec_fingerprint: String,
    pub estimate_only: bool,
    pub prediction_source: String,
    pub trace_path: Option<String>,
    pub trace_total_ms: Option<f64>,
    pub trace_coverage: TraceCoverageReport,
    pub recurrent_backward_profile: String,
    pub exact_recurrent_boundary_fusion: String,
    pub exact_recurrent_boundary_layers: Vec<usize>,
    pub persistent_cta_block_backward: String,
    pub train_step_ms_estimate: f64,
    pub expected_train_wall_seconds: f64,
    pub active_recurrent_step_ms_estimate: f64,
    pub inactive_recurrent_step_ms_estimate: f64,
    pub active_recurrent_trace_ms_per_step: Option<f64>,
    pub inactive_recurrent_trace_ms_per_step: Option<f64>,
    pub recurrent_split: RecurrentSplitReport,
    pub optimizer_update_ms_estimate: f64,
    pub output_ce_ms_estimate: f64,
    pub expected_train_steps_in_600s: usize,
    pub eval_time_ms_estimate: f64,
    pub artifact_bytes_estimate: Option<usize>,
    pub budget_status: Vec<BudgetStatusReport>,
    pub top_bottleneck: String,
    pub top_bottlenecks: Vec<BottleneckReport>,
    pub next_recommended_experiment: String,
    pub recommendations: Vec<String>,
    pub risk_flags: Vec<String>,
    pub what_if_scenarios: Vec<WindTunnelWhatIfReport>,
    pub experiment_rankings: Vec<WindTunnelExperimentRankReport>,
    pub stage_estimates: Vec<StageEstimate>,
    pub operation_dag: Vec<OperationDagNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelWhatIfReport {
    pub name: String,
    pub description: String,
    pub status: String,
    pub delta_ms_per_step: f64,
    pub train_step_ms_estimate: f64,
    pub active_recurrent_step_ms_estimate: Option<f64>,
    pub expected_train_steps_in_600s: usize,
    pub expected_train_wall_seconds: f64,
    pub train_budget_status: String,
    pub rationale: String,
    pub estimate_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelExperimentRankReport {
    pub rank: usize,
    pub scenario: String,
    pub priority_score: f64,
    pub delta_ms_per_step: f64,
    pub train_step_ms_estimate: f64,
    pub train_budget_status: String,
    pub trace_confidence: String,
    pub rationale: String,
    pub recommended_validation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEstimate {
    pub name: String,
    pub ms: f64,
    pub pct_of_step: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCoverageReport {
    pub trace_supplied: bool,
    pub has_total_timing: bool,
    pub expected_fields: Vec<String>,
    pub present_fields: Vec<String>,
    pub missing_fields: Vec<String>,
    pub mapped_stage_count: usize,
    pub nonzero_mapped_stage_count: usize,
    pub defaulted_stage_count: usize,
    pub coverage_ratio: f64,
    pub useful_stage_coverage_ratio: f64,
    pub nonzero_stage_fields: Vec<String>,
    pub zero_stage_fields: Vec<String>,
    pub traced_stage_ms: f64,
    pub traced_stage_ms_ratio: f64,
    pub signal_quality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentSplitReport {
    pub recurrent_enabled: bool,
    pub active_step_ms: f64,
    pub inactive_step_ms: f64,
    pub active_minus_inactive_ms: f64,
    pub active_source: String,
    pub inactive_source: String,
    pub exact_boundary_fusion: String,
    pub exact_boundary_layers: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatusReport {
    pub name: String,
    pub status: String,
    pub estimate: Option<f64>,
    pub budget: Option<f64>,
    pub unit: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckReport {
    pub rank: usize,
    pub stage: String,
    pub ms: f64,
    pub pct_of_step: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationDagNode {
    pub name: String,
    pub op_kind: String,
    pub status: String,
    pub layer: Option<usize>,
    pub flops_estimate: u64,
    pub hbm_bytes_estimate: u64,
    pub record_relevant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteSpec {
    pub track: LiteTrack,
    pub artifact_budget_bytes: usize,
    pub train_time_seconds: f64,
    pub eval_time_seconds: f64,
    pub memory_budget_bytes: Option<usize>,
    pub score: String,
    pub data: LiteDataSpec,
    pub model: LiteModelSpec,
    pub backend: LiteBackendSpec,
}

impl Default for LiteSpec {
    fn default() -> Self {
        Self {
            track: LiteTrack::ByteGolf,
            artifact_budget_bytes: 1_000_000,
            train_time_seconds: 300.0,
            eval_time_seconds: 60.0,
            memory_budget_bytes: Some(4_000_000_000),
            score: "bpb".to_string(),
            data: LiteDataSpec::default(),
            model: LiteModelSpec::default(),
            backend: LiteBackendSpec::default(),
        }
    }
}

impl LiteSpec {
    fn config_fingerprint(&self) -> PgResult<String> {
        let canonical = serde_json::to_vec(self).map_err(|err| {
            PgError::DataFormat(format!("PG-Lite config fingerprint encode failed: {err}"))
        })?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&canonical);
        Ok(format!("lite-crc32:{:08x}", hasher.finalize()))
    }

    fn track_behavior(&self) -> &'static str {
        match self.track {
            LiteTrack::ByteGolf => "train_then_score_fixed_artifact",
            LiteTrack::ArtifactGolf => "score_prebuilt_artifact_only",
            LiteTrack::StreamGolf => "score_first_online_update",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteTrack {
    #[default]
    ByteGolf,
    ArtifactGolf,
    StreamGolf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteDataSpec {
    pub train_path: String,
    pub val_path: String,
    pub format: String,
}

impl Default for LiteDataSpec {
    fn default() -> Self {
        Self {
            train_path: "data/pg_lite/train.txt".to_string(),
            val_path: "data/pg_lite/val.txt".to_string(),
            format: "bytes".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteModelSpec {
    pub family: LiteModelFamily,
    pub vocab: String,
    pub context: usize,
    pub residual_buckets: usize,
    pub residual_weight: f64,
    pub artifact_path: Option<String>,
}

impl Default for LiteModelSpec {
    fn default() -> Self {
        Self {
            family: LiteModelFamily::NgramResidual,
            vocab: "byte".to_string(),
            context: 512,
            residual_buckets: 1024,
            residual_weight: 0.15,
            artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteModelFamily {
    ByteNgram,
    #[default]
    NgramResidual,
    ArtifactOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteBackendSpec {
    pub kind: LiteBackendKind,
    pub allow_cpu_fallback: bool,
}

impl Default for LiteBackendSpec {
    fn default() -> Self {
        Self {
            kind: LiteBackendKind::CpuReference,
            allow_cpu_fallback: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteBackendKind {
    #[default]
    CpuReference,
    MetalApple,
    MlxPrototype,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteRunOptions {
    pub config: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteInitOptions {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    pub artifact_budget_bytes: usize,
    pub train_time_seconds: f64,
    pub eval_time_seconds: f64,
    pub memory_budget_bytes: Option<usize>,
    pub val_bytes: Option<usize>,
    pub val_fraction: f64,
    pub context: usize,
    pub residual_buckets: usize,
    pub residual_weight: f64,
    pub backend: LiteBackendKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteInitReport {
    pub kind: &'static str,
    pub input_path: String,
    pub output_dir: String,
    pub input_bytes: usize,
    pub input_crc32: String,
    pub train_path: String,
    pub val_path: String,
    pub train_bytes: usize,
    pub train_crc32: String,
    pub val_bytes: usize,
    pub val_crc32: String,
    pub val_fraction_effective: f64,
    pub configs: Vec<LiteInitConfigReport>,
    pub readme_path: String,
    pub report_path: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteInitConfigReport {
    pub name: String,
    pub path: String,
    pub track: LiteTrack,
    pub model_family: LiteModelFamily,
    pub backend: LiteBackendKind,
    pub allow_cpu_fallback: bool,
    pub artifact_budget_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteVerifyArtifactOptions {
    pub artifact: PathBuf,
    pub config: Option<PathBuf>,
    pub val: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteVerifyArtifactReport {
    pub kind: &'static str,
    pub artifact_path: String,
    pub artifact_bytes: usize,
    pub artifact_crc32: String,
    pub artifact_kind: String,
    pub decoded_ok: bool,
    pub model_reconstruct_ok: bool,
    pub source_config_fingerprint: String,
    pub model_fingerprint: String,
    pub expected_config_fingerprint: Option<String>,
    pub source_config_matches_expected: Option<bool>,
    pub model_family: LiteModelFamily,
    pub context: usize,
    pub residual_weight: f64,
    pub bigram_rows: usize,
    pub residual_buckets: usize,
    pub bigram_entries: usize,
    pub residual_entries: usize,
    pub artifact_budget_bytes: Option<usize>,
    pub artifact_budget_ok: Option<bool>,
    pub val_path: Option<String>,
    pub val_bytes: Option<usize>,
    pub validation_bpb: Option<f64>,
    pub validation_bpb_scope: &'static str,
    pub notes: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteBenchmarkOptions {
    pub config: PathBuf,
    pub output_dir: PathBuf,
    pub repeats: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteSuiteOptions {
    pub config_dir: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteSuiteReport {
    pub kind: &'static str,
    pub config_dir: String,
    pub output_dir: String,
    pub configs_found: usize,
    pub runs: Vec<LiteSuiteRunSummary>,
    pub best_bpb: Option<f64>,
    pub best_config: Option<String>,
    pub summary_json_path: String,
    pub summary_markdown_path: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteBenchmarkReport {
    pub kind: &'static str,
    pub config_path: String,
    pub output_dir: String,
    pub repeats: usize,
    pub backends: Vec<LiteBenchmarkBackendReport>,
    pub cpu_vs_metal_speedup_total: Option<f64>,
    pub cpu_vs_metal_speedup_train: Option<f64>,
    pub cpu_vs_metal_bpb_delta: Option<f64>,
    pub best_total_wall_backend: Option<LiteBackendKind>,
    pub best_bpb_backend: Option<LiteBackendKind>,
    pub summary_json_path: String,
    pub summary_markdown_path: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteBenchmarkBackendReport {
    pub requested_backend: LiteBackendKind,
    pub status: String,
    pub error: Option<String>,
    pub runs: Vec<LiteBenchmarkRunReport>,
    pub run_count: usize,
    pub accelerated_run_count: usize,
    pub execution_backends: Vec<LiteBackendKind>,
    pub median_total_wall_seconds: Option<f64>,
    pub median_train_wall_seconds: Option<f64>,
    pub median_eval_wall_seconds: Option<f64>,
    pub median_validation_bpb: Option<f64>,
    pub artifact_actual_bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteBenchmarkRunReport {
    pub run_index: usize,
    pub output_dir: String,
    pub report_path: String,
    pub status: String,
    pub execution_backend: LiteBackendKind,
    pub backend_accelerated: bool,
    pub validation_bpb: f64,
    pub total_wall_seconds: f64,
    pub train_wall_seconds: f64,
    pub eval_wall_seconds: f64,
    pub artifact_actual_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteSuiteRunSummary {
    pub config_path: String,
    pub output_dir: String,
    pub track: LiteTrack,
    pub model_family: LiteModelFamily,
    pub requested_backend: LiteBackendKind,
    pub execution_backend: LiteBackendKind,
    pub backend_status: String,
    pub backend_accelerated: bool,
    pub status: String,
    pub validation_bpb: f64,
    pub artifact_bytes_estimate: usize,
    pub artifact_actual_bytes: usize,
    pub artifact_budget_ok: bool,
    pub memory_budget_status: String,
    pub score_first_tokens_scored: usize,
    pub score_first_update_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteRunReport {
    pub kind: &'static str,
    pub config_path: String,
    pub config_fingerprint: String,
    pub track: LiteTrack,
    pub track_behavior: &'static str,
    pub backend: LiteBackendKind,
    pub execution_backend: LiteBackendKind,
    pub backend_status: String,
    pub backend_accelerated: bool,
    pub backend_fallback_reason: Option<String>,
    pub model_family: LiteModelFamily,
    pub train_bytes: usize,
    pub train_bytes_used: usize,
    pub val_bytes: usize,
    pub train_time_seconds_budget: f64,
    pub eval_time_seconds_budget: f64,
    pub train_wall_seconds: f64,
    pub eval_wall_seconds: f64,
    pub timing: LiteTimingReport,
    pub artifact_budget_bytes: usize,
    pub artifact_bytes_estimate: usize,
    pub artifact_actual_bytes: usize,
    pub artifact_path: String,
    pub artifact_budget_ok: bool,
    pub artifact_manifest_path: String,
    pub memory_budget_bytes: Option<usize>,
    pub memory_bytes_estimate: usize,
    pub memory_budget_ok: Option<bool>,
    pub memory_budget_status: String,
    pub train_time_budget_ok: bool,
    pub eval_time_budget_ok: bool,
    pub validation_bpb: f64,
    pub validation_bpb_scope: &'static str,
    pub score_first_legal: bool,
    pub score_first_tokens_scored: usize,
    pub score_first_update_count: usize,
    pub proxy_only: bool,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteTimingReport {
    pub config_load_wall_seconds: f64,
    pub backend_select_wall_seconds: f64,
    pub data_load_wall_seconds: f64,
    pub train_wall_seconds: f64,
    pub eval_wall_seconds: f64,
    pub artifact_json_write_wall_seconds: f64,
    pub artifact_binary_write_wall_seconds: f64,
    pub manifest_write_wall_seconds: f64,
    pub total_wall_seconds_excluding_report_write: f64,
    pub report_write_excluded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteArtifactManifest {
    pub kind: &'static str,
    pub config_fingerprint: String,
    pub model_fingerprint: String,
    pub track: LiteTrack,
    pub track_behavior: &'static str,
    pub backend: LiteBackendKind,
    pub execution_backend: LiteBackendKind,
    pub backend_status: String,
    pub backend_accelerated: bool,
    pub backend_fallback_reason: Option<String>,
    pub model_family: LiteModelFamily,
    pub score: String,
    pub score_scope: &'static str,
    pub train_bytes: usize,
    pub train_bytes_used: usize,
    pub val_bytes: usize,
    pub artifact_budget_bytes: usize,
    pub artifact_bytes_estimate: usize,
    pub artifact_actual_bytes: usize,
    pub artifact_path: String,
    pub artifact_budget_ok: bool,
    pub memory_budget_bytes: Option<usize>,
    pub memory_bytes_estimate: usize,
    pub memory_budget_status: String,
    pub score_first_tokens_scored: usize,
    pub score_first_update_count: usize,
    pub outputs: Vec<LiteOutputArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteOutputArtifact {
    pub path: String,
    pub kind: &'static str,
    pub stable_json: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteStoredArtifact {
    pub kind: String,
    pub version: u32,
    pub source_config_fingerprint: String,
    pub model_fingerprint: String,
    pub model_family: LiteModelFamily,
    pub context: usize,
    pub residual_weight: f64,
    pub bigram_rows: usize,
    pub residual_buckets: usize,
    pub bigram_entries: Vec<[u32; 3]>,
    pub residual_entries: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteBackendSelection {
    pub requested_backend: LiteBackendKind,
    pub execution_backend: LiteBackendKind,
    pub status: String,
    pub accelerated: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelSuiteOptions {
    pub spec: PathBuf,
    pub trace_dir: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelReviewerOptions {
    pub spec: PathBuf,
    pub trace_dir: PathBuf,
    pub output_dir: PathBuf,
    pub target_step_ms: f64,
    pub active_recurrent_replay_cut_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelSuiteReport {
    pub kind: &'static str,
    pub spec_path: String,
    pub trace_dir: String,
    pub output_dir: String,
    pub traces_found: usize,
    pub traces_with_step_timing: usize,
    pub traces_with_stage_attribution: usize,
    pub traces_total_only: usize,
    pub traces_shape_only: usize,
    pub baseline_report_path: String,
    pub baseline_train_step_ms_estimate: f64,
    pub reports: Vec<WindTunnelSuiteTraceSummary>,
    pub mean_abs_baseline_step_error_ms: Option<f64>,
    pub mean_abs_baseline_pct_error: Option<f64>,
    pub summary_json_path: String,
    pub summary_markdown_path: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelReviewerReport {
    pub kind: &'static str,
    pub spec_path: String,
    pub trace_dir: String,
    pub output_dir: String,
    pub report_json_path: String,
    pub report_markdown_path: String,
    pub trace_index_path: String,
    pub corpus_path: String,
    pub calibration_path: String,
    pub scenario_path: String,
    pub suite_summary_path: String,
    pub traces_found: usize,
    pub parseable_traces: usize,
    pub deduped_runs: usize,
    pub timed_runs: usize,
    pub exact_2135_timed_runs: usize,
    pub stage_calibration_runs: usize,
    pub canonical_caseops_runs: usize,
    pub best_exact_2135_ms: Option<f64>,
    pub best_active_recurrent_ms: Option<f64>,
    pub calibration_confidence_summary: String,
    pub holdout_validation: WindTunnelHoldoutReport,
    pub scenario_clears_target: bool,
    pub scenario_predicted_step_ms: f64,
    pub recommended_next_experiment: String,
    pub top_stage_metrics: Vec<CalibratedStageMetricReport>,
    pub warnings: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelHoldoutReport {
    pub kind: &'static str,
    pub estimate_only: bool,
    pub predictor_kind: &'static str,
    pub train_sample_count: usize,
    pub holdout_sample_count: usize,
    pub baseline_ms: Option<f64>,
    pub mean_predicted_ms: Option<f64>,
    pub mean_abs_error_ms: Option<f64>,
    pub mean_abs_pct_error: Option<f64>,
    pub spearman_rank_correlation: Option<f64>,
    pub samples: Vec<WindTunnelHoldoutSampleReport>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelHoldoutSampleReport {
    pub run_key: String,
    pub selected_trace_path: String,
    pub class_label: String,
    pub prediction_source: String,
    pub observed_ms: f64,
    pub predicted_ms: f64,
    pub abs_error_ms: f64,
    pub pct_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelSuiteTraceSummary {
    pub trace_path: String,
    pub report_path: String,
    pub prediction_source: String,
    pub trace_signal_quality: String,
    pub trace_total_ms: Option<f64>,
    pub trace_coverage_ratio: f64,
    pub useful_stage_coverage_ratio: f64,
    pub nonzero_stage_field_count: usize,
    pub calibration_role: String,
    pub train_step_ms_estimate: f64,
    pub baseline_abs_step_error_ms: Option<f64>,
    pub baseline_pct_error: Option<f64>,
    pub top_bottleneck: String,
    pub risk_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceIndexOptions {
    pub spec: PathBuf,
    pub trace_dir: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelCorpusOptions {
    pub spec: PathBuf,
    pub trace_dir: PathBuf,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelCalibrationOptions {
    pub spec: PathBuf,
    pub trace_dir: Option<PathBuf>,
    pub corpus: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelScenarioOptions {
    pub spec: PathBuf,
    pub calibration: Option<PathBuf>,
    pub trace: Option<PathBuf>,
    pub active_recurrent_replay_cut_ms: f64,
    pub bank_update_cut_ms: f64,
    pub graph_overhead_cut_ms: f64,
    pub target_step_ms: f64,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceIndexReport {
    pub kind: &'static str,
    pub spec_path: String,
    pub trace_dir: String,
    pub traces_found: usize,
    pub parseable_traces: usize,
    pub traces_with_step_timing: usize,
    pub stage_calibration_traces: usize,
    pub partial_stage_calibration_traces: usize,
    pub total_calibration_traces: usize,
    pub shape_model_only_traces: usize,
    pub canonical_caseops_traces: usize,
    pub best_step_ms: Option<f64>,
    pub best_step_trace: Option<String>,
    pub fastest_exact_2135_ms: Option<f64>,
    pub fastest_exact_2135_trace: Option<String>,
    pub active_recurrent_best_ms: Option<f64>,
    pub active_recurrent_best_trace: Option<String>,
    pub entries: Vec<TraceIndexEntry>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceIndexEntry {
    pub trace_path: String,
    pub parsed_trace: bool,
    pub calibration_role: String,
    pub signal_quality: String,
    pub trace_total_ms: Option<f64>,
    pub active_recurrent_ms_per_step: Option<f64>,
    pub inactive_recurrent_ms_per_step: Option<f64>,
    pub active_recurrent_steps: Option<usize>,
    pub timing_steps: Option<usize>,
    pub steps_completed: Option<usize>,
    pub run_name: Option<String>,
    pub mode: Option<String>,
    pub record_profile: Option<String>,
    pub recurrent_backward_profile: Option<String>,
    pub cuda_graph_profile: Option<String>,
    pub world_size: Option<usize>,
    pub seq_len: Option<usize>,
    pub canonical_caseops_dataset: Option<bool>,
    pub train_shards: Option<usize>,
    pub val_tokens: Option<usize>,
    pub f32_to_bf16_bridge_launches: Option<usize>,
    pub bf16_to_f32_bridge_launches: Option<usize>,
    pub host_batch_flatten_calls: Option<usize>,
    pub host_to_device_batch_bytes: Option<usize>,
    pub useful_stage_coverage_ratio: f64,
    pub traced_stage_ms_ratio: f64,
    pub nonzero_stage_fields: Vec<String>,
    pub zero_stage_fields: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelCorpusReport {
    pub kind: String,
    pub spec_path: String,
    pub trace_dir: String,
    pub traces_found: usize,
    pub parseable_traces: usize,
    pub deduped_runs: usize,
    pub duplicate_source_files: usize,
    pub timed_runs: usize,
    pub stage_calibration_runs: usize,
    pub partial_stage_calibration_runs: usize,
    pub total_calibration_runs: usize,
    pub exact_2135_runs: usize,
    pub canonical_caseops_runs: usize,
    pub allst_diagnostic_runs: usize,
    pub best_exact_2135_ms: Option<f64>,
    pub best_exact_2135_run_key: Option<String>,
    pub best_active_recurrent_ms: Option<f64>,
    pub best_active_recurrent_run_key: Option<String>,
    pub runs: Vec<WindTunnelCorpusRun>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelCorpusRun {
    pub run_key: String,
    pub selected_trace_path: String,
    pub source_paths: Vec<String>,
    pub selection_score: i32,
    pub selection_reason: String,
    pub calibration_role: String,
    pub signal_quality: String,
    pub trace_total_ms: Option<f64>,
    pub active_recurrent_ms_per_step: Option<f64>,
    pub inactive_recurrent_ms_per_step: Option<f64>,
    pub active_recurrent_steps: Option<usize>,
    pub timing_steps: Option<usize>,
    pub steps_completed: Option<usize>,
    pub run_name: Option<String>,
    pub mode: Option<String>,
    pub record_profile: Option<String>,
    pub recurrent_backward_profile: Option<String>,
    pub cuda_graph_profile: Option<String>,
    pub world_size: Option<usize>,
    pub seq_len: Option<usize>,
    pub canonical_caseops_dataset: Option<bool>,
    pub train_shards: Option<usize>,
    pub val_tokens: Option<usize>,
    pub f32_to_bf16_bridge_launches: Option<usize>,
    pub bf16_to_f32_bridge_launches: Option<usize>,
    pub host_batch_flatten_calls: Option<usize>,
    pub host_to_device_batch_bytes: Option<usize>,
    pub useful_stage_coverage_ratio: f64,
    pub traced_stage_ms_ratio: f64,
    pub nonzero_stage_fields: Vec<String>,
    pub zero_stage_fields: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelCalibrationReport {
    pub kind: String,
    pub spec_path: String,
    pub source_kind: String,
    pub corpus_path: Option<String>,
    pub trace_dir: Option<String>,
    pub runs_considered: usize,
    pub timed_runs: usize,
    pub exact_2135_timed_runs: usize,
    pub stage_calibration_runs: usize,
    pub total_step_ms: Option<CalibratedMetricReport>,
    pub exact_2135_total_step_ms: Option<CalibratedMetricReport>,
    pub active_recurrent_ms: Option<CalibratedMetricReport>,
    pub inactive_recurrent_ms: Option<CalibratedMetricReport>,
    pub recurrent_active_minus_inactive_ms: Option<CalibratedMetricReport>,
    pub stage_metrics: Vec<CalibratedStageMetricReport>,
    pub recommended_trace_to_collect: String,
    pub confidence_summary: String,
    pub warnings: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedMetricReport {
    pub name: String,
    pub estimate_ms: f64,
    pub best_ms: f64,
    pub median_ms: f64,
    pub p90_ms: f64,
    pub sample_count: usize,
    pub confidence: String,
    pub source_run_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedStageMetricReport {
    pub stage: String,
    pub estimate_ms: f64,
    pub sample_count: usize,
    pub confidence: String,
    pub source_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTunnelScenarioReport {
    pub kind: String,
    pub spec_fingerprint: String,
    pub estimate_only: bool,
    pub source_kind: String,
    pub calibration_path: Option<String>,
    pub trace_path: Option<String>,
    pub base_step_ms: f64,
    pub base_step_source: String,
    pub target_step_ms: f64,
    pub active_recurrent_replay_cut_ms: f64,
    pub bank_update_cut_ms: f64,
    pub graph_overhead_cut_ms: f64,
    pub total_requested_cut_ms: f64,
    pub predicted_step_ms: f64,
    pub predicted_active_recurrent_ms: Option<f64>,
    pub expected_train_steps_in_600s: usize,
    pub expected_train_wall_seconds: f64,
    pub clears_target: bool,
    pub remaining_gap_ms: f64,
    pub risk_level: String,
    pub recommendation: String,
    pub warnings: Vec<String>,
    pub status: String,
}

pub fn run_verify(options: VerifyOptions) -> PgResult<VerifyReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let fingerprint = plan.variant_fingerprint.clone();
    let mut notes = Vec::new();

    let artifact_budget_known = options.artifact.is_some();
    let mut artifact_manifest_ok = options.artifact.is_none();
    let mut artifact_budget_ok = options.artifact.is_none();
    if let Some(path) = &options.artifact {
        let size = fs::metadata(path)?.len() as usize;
        artifact_budget_ok = size <= spec.quant.target_artifact_bytes;
        let mut model = GptModel::new(spec.model.to_model_config());
        match pg_quant::export::load_artifact_with_spec(path, &mut model, &spec.quant, true) {
            Ok(()) => artifact_manifest_ok = true,
            Err(err) => {
                artifact_manifest_ok = false;
                notes.push(format!("artifact strict manifest validation failed: {err}"));
            }
        }
    } else {
        notes.push("artifact not supplied; artifact checks are skipped".to_string());
    }

    let data_preflight = data_preflight_report(&spec)?;
    let caseops_sidecar_ok = data_preflight.sidecar_structural_ok;
    if !caseops_sidecar_ok {
        notes.push(
            "CaseOps byte sidecar is required or configured but not locally valid".to_string(),
        );
    }

    let eval_legality = local_eval_legality_report()?;
    if eval_legality.status != "pass" {
        notes.push("local eval legality smoke failed".to_string());
    }
    let bpb_smoke_ok = eval_legality.bpb_byte_accounting_ok;
    let score_first_legal = !spec.eval.qttt || spec.eval.legal_score_first;
    let artifact_audit_ok = match &options.artifact_audit {
        Some(path) => artifact_audit_file_ok(path, spec.quant.target_artifact_bytes)?,
        None => true,
    };
    if !artifact_audit_ok {
        notes.push("artifact audit JSON is missing required byte/hash/budget fields".to_string());
    }
    let score_first_log_ok = match &options.score_first_log {
        Some(path) => score_first_log_file_ok(path)?,
        None => true,
    };
    if !score_first_log_ok {
        notes.push("score-first log does not prove score-before-update ordering".to_string());
    }
    let proposal_features = proposal_feature_report();
    let proposal_claims_ok = proposal_features.full_train_step_graph == "not_implemented"
        && proposal_features.persistent_cta_block_backward == "not_implemented";

    let status = if artifact_manifest_ok
        && artifact_budget_ok
        && caseops_sidecar_ok
        && (!data_preflight.canonical_target || data_preflight.ready)
        && bpb_smoke_ok
        && eval_legality.status == "pass"
        && score_first_legal
        && artifact_audit_ok
        && score_first_log_ok
        && proposal_claims_ok
    {
        "pass"
    } else {
        "fail"
    }
    .to_string();

    let report = VerifyReport {
        kind: "pg_local_verify",
        spec_path: options.spec.display().to_string(),
        spec_name: spec.name,
        spec_fingerprint: fingerprint,
        artifact_manifest_ok,
        artifact_budget_known,
        artifact_budget_ok,
        caseops_sidecar_ok,
        data_preflight,
        eval_legality,
        bpb_smoke_ok,
        score_first_legal,
        artifact_audit_ok,
        score_first_log_ok,
        proposal_claims_ok,
        proposal_features,
        notes,
        status,
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_proof_bundle(options: ProofBundleOptions) -> PgResult<ProofBundleReport> {
    fs::create_dir_all(&options.output_dir)?;

    let verify_path = options.output_dir.join("verify.json");
    let dist_sim_path = options.output_dir.join("dist_sim.json");
    let artifact_lab_path = options.output_dir.join("artifact_lab.json");
    let modelgolf_path = options.output_dir.join("modelgolf.json");
    let wind_tunnel_path = options.output_dir.join("wind_tunnel.json");
    let trace_corpus_path = options.output_dir.join("trace_corpus.json");
    let calibration_model_path = options.output_dir.join("calibration_model.json");
    let scenario_report_path = options
        .output_dir
        .join("scenario_exact_recurrent_replay.json");
    let lite_output_dir = options.output_dir.join("pg_lite");
    let lite_report_path = lite_output_dir.join("report.json");
    let lite_benchmark_output_dir = options.output_dir.join("pg_lite_benchmark");
    let lite_benchmark_summary_path = lite_benchmark_output_dir.join("summary.json");
    let lite_suite_output_dir = options.output_dir.join("pg_lite_suite");
    let lite_suite_summary_path = lite_suite_output_dir.join("summary.json");
    let wind_tunnel_reviewer_output_dir = options.output_dir.join("wind_tunnel_reviewer");
    let wind_tunnel_reviewer_report_path = wind_tunnel_reviewer_output_dir.join("report.json");
    let backend_cpu_check_path = options.output_dir.join("backend_cpu_reference.json");
    let backend_metal_check_path = options.output_dir.join("backend_metal_apple.json");
    let proposal_feature_path = options.output_dir.join("proposal_features.json");
    let proof_bundle_path = options.output_dir.join("proof_bundle.json");
    let evidence_manifest_path = options.output_dir.join("evidence_manifest.json");
    let readme_path = options.output_dir.join("README.md");

    let verify = run_verify(VerifyOptions {
        spec: options.spec.clone(),
        artifact: options.artifact.clone(),
        artifact_audit: None,
        score_first_log: None,
        output: Some(verify_path.clone()),
    })?;
    let dist = run_dist_sim(DistSimOptions {
        spec: options.spec.clone(),
        world_size: 8,
        steps: 3,
        seed: 1,
        output: Some(dist_sim_path.clone()),
    })?;
    let artifact = run_artifact_lab(ArtifactLabOptions {
        spec: options.spec.clone(),
        artifact: options.artifact.clone(),
        sweep: vec![
            "quant".to_string(),
            "lqer".to_string(),
            "compression".to_string(),
        ],
        mini_train: None,
        mini_val: None,
        output: Some(artifact_lab_path.clone()),
    })?;
    let modelgolf = run_modelgolf_plan(ModelGolfOptions {
        section: modelgolf::ModelGolfSection::Full,
        spec: options.spec.clone(),
        output: Some(modelgolf_path.clone()),
        hardware: "proof_bundle_local".to_string(),
        runtime: "pg-local".to_string(),
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
    })?;
    let wind = run_wind_tunnel(WindTunnelOptions {
        spec: options.spec.clone(),
        trace: options.trace.clone(),
        output: Some(wind_tunnel_path.clone()),
    })?;
    let trace_dir = options
        .trace_dir
        .clone()
        .or_else(|| {
            options
                .trace
                .as_ref()
                .and_then(|trace| trace.parent().map(Path::to_path_buf))
        })
        .unwrap_or_else(|| options.output_dir.join("trace_corpus_empty"));
    fs::create_dir_all(&trace_dir)?;
    let _corpus = run_wind_tunnel_corpus(WindTunnelCorpusOptions {
        spec: options.spec.clone(),
        trace_dir: trace_dir.clone(),
        output: Some(trace_corpus_path.clone()),
    })?;
    let calibration = run_wind_tunnel_calibrate(WindTunnelCalibrationOptions {
        spec: options.spec.clone(),
        trace_dir: None,
        corpus: Some(trace_corpus_path.clone()),
        output: Some(calibration_model_path.clone()),
    })?;
    let scenario = run_wind_tunnel_scenario(WindTunnelScenarioOptions {
        spec: options.spec.clone(),
        calibration: Some(calibration_model_path.clone()),
        trace: options.trace.clone(),
        active_recurrent_replay_cut_ms: 10.0,
        bank_update_cut_ms: 0.0,
        graph_overhead_cut_ms: 0.0,
        target_step_ms: 120.0,
        output: Some(scenario_report_path.clone()),
    })?;
    let lite = run_lite(LiteRunOptions {
        config: options.lite_config.clone(),
        output: lite_output_dir.clone(),
    })?;
    let lite_benchmark = run_lite_benchmark(LiteBenchmarkOptions {
        config: options.lite_config.clone(),
        output_dir: lite_benchmark_output_dir.clone(),
        repeats: 3,
    })?;
    let lite_suite = if let Some(config_dir) = options.lite_config_dir.clone() {
        run_lite_suite(LiteSuiteOptions {
            config_dir,
            output_dir: lite_suite_output_dir.clone(),
        })?
    } else {
        run_lite_suite_for_configs(
            options.lite_config.display().to_string(),
            lite_suite_output_dir.clone(),
            vec![options.lite_config.clone()],
        )?
    };
    let wind_tunnel_reviewer = run_wind_tunnel_report(WindTunnelReviewerOptions {
        spec: options.spec.clone(),
        trace_dir: trace_dir.clone(),
        output_dir: wind_tunnel_reviewer_output_dir.clone(),
        target_step_ms: 120.0,
        active_recurrent_replay_cut_ms: 10.0,
    })?;
    let backend_cpu = run_backend_check(BackendCheckOptions {
        backend: LiteBackendKind::CpuReference,
        output: Some(backend_cpu_check_path.clone()),
    })?;
    let backend_metal = run_backend_check(BackendCheckOptions {
        backend: LiteBackendKind::MetalApple,
        output: Some(backend_metal_check_path.clone()),
    })?;
    let proposal_features = proposal_feature_report();
    write_json_pretty(&proposal_feature_path, &proposal_features)?;

    let pass = verify.status == "pass"
        && dist.status == "pass"
        && artifact.status == "pass"
        && lite.status == "pass"
        && lite_benchmark.status == "pass"
        && lite_suite.status == "pass"
        && backend_cpu.status == "pass"
        && backend_metal.kernel_source_present
        && backend_metal.kernel_contract_ok
        && wind.train_step_ms_estimate.is_finite();
    let status = if pass { "pass" } else { "fail" }.to_string();
    let report = ProofBundleReport {
        kind: "pg_local_proof_bundle",
        output_dir: options.output_dir.display().to_string(),
        readme_path: readme_path.display().to_string(),
        evidence_manifest_path: evidence_manifest_path.display().to_string(),
        verify_path: verify_path.display().to_string(),
        dist_sim_path: dist_sim_path.display().to_string(),
        artifact_lab_path: artifact_lab_path.display().to_string(),
        modelgolf_path: modelgolf_path.display().to_string(),
        wind_tunnel_path: wind_tunnel_path.display().to_string(),
        trace_corpus_path: trace_corpus_path.display().to_string(),
        calibration_model_path: calibration_model_path.display().to_string(),
        scenario_report_path: scenario_report_path.display().to_string(),
        lite_report_path: lite_report_path.display().to_string(),
        lite_benchmark_summary_path: lite_benchmark_summary_path.display().to_string(),
        lite_suite_summary_path: lite_suite_summary_path.display().to_string(),
        wind_tunnel_reviewer_report_path: wind_tunnel_reviewer_report_path.display().to_string(),
        backend_cpu_check_path: backend_cpu_check_path.display().to_string(),
        backend_metal_check_path: backend_metal_check_path.display().to_string(),
        proposal_feature_path: proposal_feature_path.display().to_string(),
        status: status.clone(),
    };
    write_json_pretty(&proof_bundle_path, &report)?;
    let manifest = evidence_manifest_for_proof_bundle(
        &options.output_dir,
        &report,
        &verify,
        &dist,
        &artifact,
        &modelgolf,
        &wind,
        &calibration,
        &scenario,
        &lite,
        &lite_benchmark,
        &lite_suite,
        &wind_tunnel_reviewer,
        &backend_cpu,
        &backend_metal,
        &proposal_features,
    );
    write_json_pretty(&evidence_manifest_path, &manifest)?;
    fs::write(&readme_path, proof_bundle_readme(&report, &manifest))?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn evidence_manifest_for_proof_bundle(
    output_dir: &Path,
    report: &ProofBundleReport,
    verify: &VerifyReport,
    dist: &DistSimReport,
    artifact: &ArtifactLabReport,
    modelgolf: &ModelGolfReport,
    wind: &WindTunnelReport,
    calibration: &WindTunnelCalibrationReport,
    scenario: &WindTunnelScenarioReport,
    lite: &LiteRunReport,
    lite_benchmark: &LiteBenchmarkReport,
    lite_suite: &LiteSuiteReport,
    wind_tunnel_reviewer: &WindTunnelReviewerReport,
    backend_cpu: &BackendCheckReport,
    backend_metal: &BackendCheckReport,
    proposal_features: &ProposalFeatureReport,
) -> EvidenceManifestReport {
    let mut blocking_reasons = Vec::new();
    if verify.status != "pass" {
        blocking_reasons.push(format!(
            "verify status is {}; notes: {}",
            verify.status,
            verify.notes.join("; ")
        ));
    }
    if !verify.artifact_budget_known {
        blocking_reasons.push("final artifact bytes were not supplied to verify".to_string());
    }
    if !verify.caseops_sidecar_ok {
        blocking_reasons.push("canonical CaseOps byte sidecar is not locally valid".to_string());
    }
    for note in &verify.data_preflight.notes {
        blocking_reasons.push(format!("data preflight: {note}"));
    }
    if artifact.artifact_budget_known && !artifact.artifact_budget_ok {
        blocking_reasons.push("supplied artifact exceeds configured byte budget".to_string());
    }
    if !backend_metal.executable_backend_available {
        blocking_reasons.push(format!(
            "PG-Lite Metal backend is not executable in this local proof build; feature_enabled={}, runtime_linked={}, notes: {}",
            backend_metal.cargo_feature_enabled,
            backend_metal.rust_runtime_linked,
            backend_metal.notes.join("; ")
        ));
    }

    EvidenceManifestReport {
        kind: "pg_local_evidence_manifest",
        package_kind: "local_non_record_proof",
        output_dir: output_dir.display().to_string(),
        record_claim: false,
        leaderboard_claim: false,
        proof_bundle_status: report.status.clone(),
        component_statuses: vec![
            EvidenceComponentStatus {
                name: "verify".to_string(),
                status: verify.status.clone(),
                path: report.verify_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "dist_sim".to_string(),
                status: dist.status.clone(),
                path: report.dist_sim_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "artifact_lab".to_string(),
                status: artifact.status.clone(),
                path: report.artifact_lab_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "modelgolf".to_string(),
                status: modelgolf.status.clone(),
                path: report.modelgolf_path.clone(),
                decisive_for_local_bundle: false,
            },
            EvidenceComponentStatus {
                name: "wind_tunnel".to_string(),
                status: if wind.train_step_ms_estimate.is_finite() {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
                path: report.wind_tunnel_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "wind_tunnel_calibration".to_string(),
                status: calibration.status.clone(),
                path: report.calibration_model_path.clone(),
                decisive_for_local_bundle: false,
            },
            EvidenceComponentStatus {
                name: "wind_tunnel_scenario".to_string(),
                status: scenario.status.clone(),
                path: report.scenario_report_path.clone(),
                decisive_for_local_bundle: false,
            },
            EvidenceComponentStatus {
                name: "wind_tunnel_reviewer".to_string(),
                status: wind_tunnel_reviewer.status.clone(),
                path: report.wind_tunnel_reviewer_report_path.clone(),
                decisive_for_local_bundle: false,
            },
            EvidenceComponentStatus {
                name: "pg_lite".to_string(),
                status: lite.status.clone(),
                path: report.lite_report_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "pg_lite_benchmark".to_string(),
                status: lite_benchmark.status.clone(),
                path: report.lite_benchmark_summary_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "pg_lite_suite".to_string(),
                status: lite_suite.status.clone(),
                path: report.lite_suite_summary_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "backend_cpu_reference".to_string(),
                status: backend_cpu.status.clone(),
                path: report.backend_cpu_check_path.clone(),
                decisive_for_local_bundle: true,
            },
            EvidenceComponentStatus {
                name: "backend_metal_apple".to_string(),
                status: if backend_metal.executable_backend_available {
                    "runtime_available".to_string()
                } else if backend_metal.rust_runtime_linked {
                    "runtime_linked_not_executable".to_string()
                } else if backend_metal.kernel_source_present && backend_metal.kernel_contract_ok {
                    "source_contract_feature_gated".to_string()
                } else {
                    backend_metal.status.clone()
                },
                path: report.backend_metal_check_path.clone(),
                decisive_for_local_bundle: true,
            },
        ],
        generated_files: vec![
            evidence_file(&report.verify_path, "verify_report", true),
            evidence_file(&report.dist_sim_path, "distributed_sim_report", true),
            evidence_file(&report.artifact_lab_path, "artifact_lab_report", true),
            evidence_file(&report.modelgolf_path, "modelgolf_planner_report", true),
            evidence_file(&report.wind_tunnel_path, "wind_tunnel_report", true),
            evidence_file(&report.trace_corpus_path, "wind_tunnel_trace_corpus", true),
            evidence_file(
                &report.calibration_model_path,
                "wind_tunnel_calibration_model",
                true,
            ),
            evidence_file(
                &report.scenario_report_path,
                "wind_tunnel_scenario_report",
                true,
            ),
            evidence_file(
                &report.wind_tunnel_reviewer_report_path,
                "wind_tunnel_reviewer_report",
                true,
            ),
            evidence_file(&report.lite_report_path, "pg_lite_run_report", true),
            evidence_file(
                &report.lite_benchmark_summary_path,
                "pg_lite_backend_benchmark",
                true,
            ),
            evidence_file(
                &output_dir.join("pg_lite/model.pglite.bin").display().to_string(),
                "pg_lite_model_artifact",
                true,
            ),
            evidence_file(
                &report.lite_suite_summary_path,
                "pg_lite_suite_summary",
                true,
            ),
            evidence_file(&report.backend_cpu_check_path, "backend_check", true),
            evidence_file(&report.backend_metal_check_path, "backend_check", true),
            evidence_file(&report.proposal_feature_path, "proposal_feature_report", true),
            evidence_file(
                &output_dir.join("proof_bundle.json").display().to_string(),
                "proof_bundle_report",
                true,
            ),
            EvidenceFileReport {
                path: report.readme_path.clone(),
                kind: "human_readme".to_string(),
                required: true,
                exists: true,
            },
            EvidenceFileReport {
                path: report.evidence_manifest_path.clone(),
                kind: "evidence_manifest".to_string(),
                required: true,
                exists: true,
            },
        ],
        local_evidence: vec![
            EvidenceClaimReport {
                name: "proposal_claim_honesty".to_string(),
                status: if verify.proposal_claims_ok {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
                evidence_path: Some(report.verify_path.clone()),
            },
            EvidenceClaimReport {
                name: "record_data_preflight".to_string(),
                status: if verify.data_preflight.ready {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
                evidence_path: Some(report.verify_path.clone()),
            },
            EvidenceClaimReport {
                name: "local_eval_legality".to_string(),
                status: verify.eval_legality.status.clone(),
                evidence_path: Some(report.verify_path.clone()),
            },
            EvidenceClaimReport {
                name: "distributed_optimizer_math".to_string(),
                status: dist.status.clone(),
                evidence_path: Some(report.dist_sim_path.clone()),
            },
            EvidenceClaimReport {
                name: "quant_layout_compiler".to_string(),
                status: proposal_features.quant_layout_compiler.clone(),
                evidence_path: Some(report.artifact_lab_path.clone()),
            },
            EvidenceClaimReport {
                name: "modelgolf_constraint_compiler".to_string(),
                status: modelgolf.status.clone(),
                evidence_path: Some(report.modelgolf_path.clone()),
            },
            EvidenceClaimReport {
                name: "pg_lite_cpu_reference".to_string(),
                status: lite_suite.status.clone(),
                evidence_path: Some(report.lite_suite_summary_path.clone()),
            },
            EvidenceClaimReport {
                name: "pg_lite_metal_backend".to_string(),
                status: if backend_metal.executable_backend_available {
                    "local_validated"
                } else {
                    proposal_features.pg_lite_metal_backend.as_str()
                }
                .to_string(),
                evidence_path: Some(report.lite_benchmark_summary_path.clone()),
            },
            EvidenceClaimReport {
                name: "wind_tunnel_estimate".to_string(),
                status: "estimate_only".to_string(),
                evidence_path: Some(report.wind_tunnel_path.clone()),
            },
            EvidenceClaimReport {
                name: "wind_tunnel_reviewer_packet".to_string(),
                status: wind_tunnel_reviewer.status.clone(),
                evidence_path: Some(report.wind_tunnel_reviewer_report_path.clone()),
            },
            EvidenceClaimReport {
                name: "wind_tunnel_trace_corpus".to_string(),
                status: calibration.status.clone(),
                evidence_path: Some(report.trace_corpus_path.clone()),
            },
            EvidenceClaimReport {
                name: "wind_tunnel_next_cut_scenario".to_string(),
                status: if scenario.clears_target {
                    "scenario_clears_target_estimate_only"
                } else {
                    "scenario_misses_target_estimate_only"
                }
                .to_string(),
                evidence_path: Some(report.scenario_report_path.clone()),
            },
        ],
        requires_remote_validation: vec![
            EvidenceClaimReport {
                name: "full_record_train_eval_export".to_string(),
                status: "missing".to_string(),
                evidence_path: None,
            },
            EvidenceClaimReport {
                name: "post_export_full_validation_bpb".to_string(),
                status: "missing".to_string(),
                evidence_path: None,
            },
            EvidenceClaimReport {
                name: "artifact_budget_from_final_record_run".to_string(),
                status: if verify.artifact_budget_known && verify.artifact_budget_ok {
                    "local_artifact_supplied_not_record_run"
                } else {
                    "missing"
                }
                .to_string(),
                evidence_path: Some(report.verify_path.clone()),
            },
            EvidenceClaimReport {
                name: "exact_2135_h100_step_time".to_string(),
                status: "missing".to_string(),
                evidence_path: None,
            },
            EvidenceClaimReport {
                name: "persistent_cta_block_backward".to_string(),
                status: proposal_features.persistent_cta_block_backward.clone(),
                evidence_path: Some(report.proposal_feature_path.clone()),
            },
            EvidenceClaimReport {
                name: "record_active_xsa_inside_sdpa".to_string(),
                status: proposal_features.xsa_inside_sdpa.clone(),
                evidence_path: Some(report.proposal_feature_path.clone()),
            },
        ],
        blocking_reasons,
        caveats: vec![
            "This package is local evidence only; it is not a Parameter Golf leaderboard claim."
                .to_string(),
            "Wind Tunnel values are estimates unless backed by supplied H100 timing traces."
                .to_string(),
            "PG-Lite BPB and artifact-lab mini-BPB are local proxy scores, not FineWeb validation BPB."
                .to_string(),
            "A PG-Lite Metal claim requires backend-check to report executable_backend_available=true; CPU fallback remains a contract smoke only.".to_string(),
        ],
    }
}

fn evidence_file(path: &str, kind: &str, required: bool) -> EvidenceFileReport {
    EvidenceFileReport {
        path: path.to_string(),
        kind: kind.to_string(),
        required,
        exists: Path::new(path).exists(),
    }
}

fn proof_bundle_readme(report: &ProofBundleReport, manifest: &EvidenceManifestReport) -> String {
    let mut body = String::new();
    body.push_str("# PG-Local Proof Bundle\n\n");
    body.push_str("This directory is a local, non-record evidence packet for the Rust/CUDA Parameter Golf project. It is designed for review and planning before an H100 record run.\n\n");
    body.push_str("## Status\n\n");
    body.push_str(&format!(
        "- Proof bundle status: `{}`\n",
        manifest.proof_bundle_status
    ));
    body.push_str(&format!("- Record claim: `{}`\n", manifest.record_claim));
    body.push_str(&format!(
        "- Leaderboard claim: `{}`\n\n",
        manifest.leaderboard_claim
    ));

    body.push_str("## Component Reports\n\n");
    body.push_str("| Component | Status | Report |\n");
    body.push_str("|---|---:|---|\n");
    for component in &manifest.component_statuses {
        body.push_str(&format!(
            "| `{}` | `{}` | `{}` |\n",
            component.name, component.status, component.path
        ));
    }

    body.push_str("\n## Local Evidence\n\n");
    body.push_str("| Claim | Status | Evidence |\n");
    body.push_str("|---|---:|---|\n");
    for claim in &manifest.local_evidence {
        body.push_str(&format!(
            "| `{}` | `{}` | {} |\n",
            claim.name,
            claim.status,
            claim
                .evidence_path
                .as_deref()
                .map(|path| format!("`{path}`"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }

    body.push_str("\n## Still Requires Remote Validation\n\n");
    body.push_str("| Requirement | Status | Evidence |\n");
    body.push_str("|---|---:|---|\n");
    for claim in &manifest.requires_remote_validation {
        body.push_str(&format!(
            "| `{}` | `{}` | {} |\n",
            claim.name,
            claim.status,
            claim
                .evidence_path
                .as_deref()
                .map(|path| format!("`{path}`"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }

    if !manifest.blocking_reasons.is_empty() {
        body.push_str("\n## Blocking Reasons\n\n");
        for reason in &manifest.blocking_reasons {
            body.push_str(&format!("- {reason}\n"));
        }
    }

    body.push_str("\n## Caveats\n\n");
    for caveat in &manifest.caveats {
        body.push_str(&format!("- {caveat}\n"));
    }

    body.push_str("\n## File Index\n\n");
    body.push_str("| Kind | Required | Exists | Path |\n");
    body.push_str("|---|---:|---:|---|\n");
    for file in &manifest.generated_files {
        body.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` |\n",
            file.kind, file.required, file.exists, file.path
        ));
    }

    body.push_str("\n## Entry Points\n\n");
    body.push_str(&format!("- Bundle directory: `{}`\n", report.output_dir));
    body.push_str(&format!(
        "- Proof bundle JSON: `{}/proof_bundle.json`\n",
        report.output_dir
    ));
    body.push_str(&format!(
        "- Evidence manifest: `{}`\n",
        report.evidence_manifest_path
    ));
    body.push_str(&format!("- Machine summary: `{}`\n", report.output_dir));
    body.push_str("\nThis README is generated by `pg-local proof-bundle`; edit the source command or inputs, then regenerate rather than hand-editing this file.\n");
    body
}

pub fn run_dist_sim(options: DistSimOptions) -> PgResult<DistSimReport> {
    if options.world_size == 0 {
        return Err(PgError::InvalidOp("world-size must be > 0".into()));
    }
    if options.steps == 0 {
        return Err(PgError::InvalidOp("steps must be > 0".into()));
    }
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let shapes = vec![
        plan.bank_layout.qo_bank,
        plan.bank_layout.kv_bank,
        plan.bank_layout.mlp_up_bank,
        plan.bank_layout.mlp_down_bank,
    ];
    let bank_names = ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"];

    let mut max_abs = 0.0f32;
    let mut bank_reports = Vec::new();
    for (bank_idx, (&name, shape)) in bank_names.iter().zip(shapes.iter()).enumerate() {
        let report = simulate_bank_dist(
            name,
            *shape,
            bank_idx,
            options.world_size,
            options.steps,
            options.seed,
        );
        max_abs = max_abs.max(report.optimizer_parity_max_abs_diff);
        bank_reports.push(report);
    }

    let total_bank_elems = plan.bank_layout.qo_bank_elems
        + plan.bank_layout.kv_bank_elems
        + plan.bank_layout.mlp_up_bank_elems
        + plan.bank_layout.mlp_down_bank_elems;
    let total_grad_bytes = total_bank_elems * std::mem::size_of::<f32>();
    let shard_bytes = total_grad_bytes / options.world_size.max(1);
    let tolerance = 1e-5;
    let pass = max_abs <= tolerance;
    let report = DistSimReport {
        kind: "pg_local_dist_sim",
        spec_fingerprint: plan.variant_fingerprint,
        world_size: options.world_size,
        steps: options.steps,
        parameter_shapes: shapes,
        bank_reports,
        reduce_scatter_equivalent: pass,
        all_gather_equivalent: pass,
        optimizer_parity_max_abs_diff: max_abs,
        tolerance,
        estimated_reduce_scatter_bytes_per_step: shard_bytes,
        estimated_all_gather_bytes_per_step: shard_bytes,
        bf16_shadow_freshness: "pass",
        status: if pass { "pass" } else { "fail" }.to_string(),
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_artifact_lab(options: ArtifactLabOptions) -> PgResult<ArtifactLabReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let model_config = spec.model.to_model_config();
    let manifest = compile_quant_layout_manifest(&spec.quant, Some(&model_config))?;
    let kernel_set = CompiledQuantKernelSet::for_manifest(&manifest);
    let kernel_ids = kernel_set
        .kernel_ids()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let mut artifact_manifest_ok = options.artifact.is_none();
    let mut artifact_kind = if options.artifact.is_some() {
        "unknown".to_string()
    } else {
        "none".to_string()
    };
    let mut model_bytes = None;
    let mut pg_lite_artifact = None;
    if let Some(path) = &options.artifact {
        let size = fs::metadata(path)?.len() as usize;
        model_bytes = Some(size);
        match load_lite_stored_artifact(path) {
            Ok(artifact) => {
                artifact_kind = "pg_lite".to_string();
                artifact_manifest_ok = true;
                pg_lite_artifact = Some(ArtifactLabLiteArtifactReport {
                    kind: "pg_lite_model_artifact",
                    source_config_fingerprint: artifact.source_config_fingerprint,
                    model_fingerprint: artifact.model_fingerprint,
                    model_family: artifact.model_family,
                    context: artifact.context,
                    residual_weight: artifact.residual_weight,
                    bigram_rows: artifact.bigram_rows,
                    residual_buckets: artifact.residual_buckets,
                    bigram_entries: artifact.bigram_entries.len(),
                    residual_entries: artifact.residual_entries.len(),
                    local_proxy_only: true,
                });
            }
            Err(_) => {
                let mut model = GptModel::new(model_config.clone());
                artifact_manifest_ok =
                    pg_quant::export::load_artifact_with_spec(path, &mut model, &spec.quant, true)
                        .is_ok();
                if artifact_manifest_ok {
                    artifact_kind = "parameter_golf_record".to_string();
                }
            }
        }
    }
    let code_bytes_estimate = 0usize;
    let total_bytes_estimate = model_bytes.map(|bytes| bytes + code_bytes_estimate);
    let artifact_budget_known = model_bytes.is_some();
    let artifact_budget_ok = total_bytes_estimate
        .map(|total| total <= spec.quant.target_artifact_bytes)
        .unwrap_or(false);

    let groups = manifest
        .groups
        .iter()
        .map(|group| ArtifactGroupReport {
            name: group.name.to_string(),
            bits: group.bits,
            rows: group.rows,
            cols: group.cols,
            packed_weight_bytes: group.packed_weight_bytes,
            scale_bytes: group.scale_bytes,
            lqer_bytes: group.lqer_bytes,
            pack_kernel: group.pack_kernel.to_string(),
            dequant_kernel: group.dequant_kernel.to_string(),
        })
        .collect::<Vec<_>>();
    let requested_sweeps = normalize_sweeps(&options.sweep);
    let sweeps = artifact_sweeps(&kernel_set, &requested_sweeps)?;
    let mixed_bit_allocations = mixed_bit_allocation_reports(
        &manifest.groups,
        spec.quant.target_artifact_bytes,
        total_bytes_estimate,
    );
    let mini_bpb = artifact_mini_bpb(options.mini_train.as_deref(), options.mini_val.as_deref())?;

    let report = ArtifactLabReport {
        kind: "pg_local_artifact_lab",
        spec_fingerprint: plan.variant_fingerprint,
        artifact_path: options
            .artifact
            .as_ref()
            .map(|path| path.display().to_string()),
        artifact_kind,
        requested_sweeps,
        artifact_manifest_ok,
        artifact_budget_known,
        artifact_budget_ok,
        model_bytes,
        code_bytes_estimate,
        total_bytes_estimate,
        target_artifact_bytes: spec.quant.target_artifact_bytes,
        layout_manifest_crc32: manifest.fingerprint_crc32,
        quant_kernel_ids: kernel_ids,
        groups,
        mixed_bit_allocations,
        sweeps,
        mini_bpb,
        pg_lite_artifact,
        status: if artifact_manifest_ok && (!artifact_budget_known || artifact_budget_ok) {
            "pass"
        } else {
            "fail"
        }
        .to_string(),
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_backend_check(options: BackendCheckOptions) -> PgResult<BackendCheckReport> {
    let host_os = std::env::consts::OS.to_string();
    let mut notes = Vec::new();
    let kernel_contracts = match options.backend {
        LiteBackendKind::MetalApple => pg_lite_metal_kernel_contracts(),
        LiteBackendKind::CpuReference | LiteBackendKind::MlxPrototype => Vec::new(),
    };
    let expected_kernel_symbols = kernel_contracts
        .iter()
        .map(|contract| contract.symbol.clone())
        .collect::<Vec<_>>();
    let (
        kernel_source_path,
        kernel_source_present,
        metal_compiler_path,
        metal_compiler_available,
        metal_compiler_error,
    ) = match options.backend {
        LiteBackendKind::CpuReference => {
            notes.push("CPU reference backend is always available in pg-local.".to_string());
            (None, false, None, false, None)
        }
        LiteBackendKind::MetalApple => {
            let path = pg_lite_metal_kernel_path();
            let source_present = path.exists();
            if !source_present {
                notes.push("PG-Lite Metal kernel source is missing.".to_string());
            }
            let compiler = find_metal_compiler();
            if compiler.path.is_none() {
                if let Some(error) = compiler.error.as_deref() {
                    notes.push(format!(
                        "Apple Metal compiler probe failed via `xcrun --find metal`: {error}"
                    ));
                } else {
                    notes.push(
                        "Apple Metal compiler was not found via `xcrun --find metal`.".to_string(),
                    );
                }
            }
            (
                Some(path.display().to_string()),
                source_present,
                compiler.path.clone(),
                compiler.path.is_some(),
                compiler.error.clone(),
            )
        }
        LiteBackendKind::MlxPrototype => {
            notes.push(
                "MLX prototype interop is intentionally not linked into the Rust pg-local CLI."
                    .to_string(),
            );
            (None, false, None, false, None)
        }
    };
    let (kernel_source_crc32, kernel_symbols_found) =
        if options.backend == LiteBackendKind::MetalApple {
            match kernel_source_path.as_deref() {
                Some(path) if kernel_source_present => {
                    let source = fs::read_to_string(path)?;
                    let mut hasher = crc32fast::Hasher::new();
                    hasher.update(source.as_bytes());
                    let found = expected_kernel_symbols
                        .iter()
                        .filter(|symbol| metal_source_contains_kernel(&source, symbol))
                        .cloned()
                        .collect::<Vec<_>>();
                    (Some(format!("{:08x}", hasher.finalize())), found)
                }
                _ => (None, Vec::new()),
            }
        } else {
            (None, Vec::new())
        };
    let kernel_contract_ok = expected_kernel_symbols
        .iter()
        .all(|symbol| kernel_symbols_found.iter().any(|found| found == symbol));
    if options.backend == LiteBackendKind::MetalApple && !kernel_contract_ok {
        notes.push(format!(
            "PG-Lite Metal source is missing expected kernel symbols: {}",
            expected_kernel_symbols
                .iter()
                .filter(|symbol| !kernel_symbols_found.iter().any(|found| found == *symbol))
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let (metal_compile_smoke_attempted, metal_compile_smoke_ok, metal_compile_stderr) =
        if options.backend == LiteBackendKind::MetalApple
            && kernel_source_present
            && metal_compiler_available
        {
            let source = kernel_source_path
                .as_deref()
                .map(PathBuf::from)
                .expect("Metal source path must be present when source is present");
            match compile_metal_kernel_smoke(&source) {
                Ok(()) => {
                    notes.push(
                        "PG-Lite Metal kernel source compiled to a temporary metallib.".to_string(),
                    );
                    (true, true, None)
                }
                Err(err) => {
                    notes.push("PG-Lite Metal compile smoke failed.".to_string());
                    (true, false, Some(err))
                }
            }
        } else {
            (false, false, None)
        };
    let cargo_feature_enabled = match options.backend {
        LiteBackendKind::CpuReference => true,
        LiteBackendKind::MetalApple => cfg!(feature = "metal_apple"),
        LiteBackendKind::MlxPrototype => false,
    };
    let rust_runtime_linked = match options.backend {
        LiteBackendKind::CpuReference => true,
        LiteBackendKind::MetalApple => cfg!(feature = "metal_apple"),
        LiteBackendKind::MlxPrototype => false,
    };
    let mut metal_runtime_ok = false;
    if options.backend == LiteBackendKind::MetalApple {
        if !cargo_feature_enabled {
            notes.push(
                "`metal_apple` Cargo feature is disabled; runtime execution is unavailable."
                    .to_string(),
            );
        } else {
            match metal_lite_runtime_probe() {
                Ok(message) => {
                    metal_runtime_ok = true;
                    notes.push(format!(
                        "PG-Lite Rust Metal runtime probe passed: {message}"
                    ));
                }
                Err(err) => {
                    notes.push(format!("PG-Lite Rust Metal runtime probe failed: {err}"));
                }
            }
        }
    }
    let executable_backend_available = match options.backend {
        LiteBackendKind::CpuReference => true,
        LiteBackendKind::MetalApple => metal_runtime_ok,
        LiteBackendKind::MlxPrototype => false,
    };
    let status = if executable_backend_available {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = BackendCheckReport {
        kind: "pg_local_backend_check",
        backend: options.backend,
        host_os,
        cargo_feature_enabled,
        kernel_source_path,
        kernel_source_present,
        kernel_source_crc32,
        expected_kernel_symbols,
        kernel_symbols_found,
        kernel_contracts,
        kernel_contract_ok,
        metal_compiler_path,
        metal_compiler_available,
        metal_compiler_error,
        metal_compile_smoke_attempted,
        metal_compile_smoke_ok,
        metal_compile_stderr,
        rust_runtime_linked,
        executable_backend_available,
        notes,
        status,
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_wind_tunnel(options: WindTunnelOptions) -> PgResult<WindTunnelReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let trace = options
        .trace
        .as_ref()
        .and_then(|path| fs::read_to_string(path).ok());
    let trace_json = trace.as_deref().and_then(parse_trace_body);

    let mut stages = default_stage_estimates(&plan);
    if let Some(value) = trace_json.as_ref() {
        apply_trace_to_stages(value, &mut stages);
    }
    let staged_total: f64 = stages.iter().map(|stage| stage.ms).sum();
    let trace_total_ms = trace_json.as_ref().and_then(|value| {
        trace_number_for_aliases(
            value,
            &[
                "timing_measured_ms_per_step",
                "timing_train_step_ms_per_step",
            ],
        )
    });
    let total = trace_total_ms.unwrap_or(staged_total);
    if let Some(trace_total) = trace_total_ms {
        let unattributed = (trace_total - staged_total).max(0.0);
        if unattributed > 0.001 {
            stages.push(StageEstimate {
                name: "unattributed_trace_overhead".to_string(),
                ms: unattributed,
                pct_of_step: 0.0,
                source: "trace_total_minus_known_stages".to_string(),
            });
        }
    }
    annotate_stage_percentages(&mut stages, total);
    let trace_coverage = trace_coverage_report(trace_json.as_ref(), &stages, total);
    let prediction_source = prediction_source(trace_total_ms, &trace_coverage);
    let optimizer = stage_ms(&stages, "optimizer_update");
    let ce = stage_ms(&stages, "output_ce");
    let active_trace = trace_json
        .as_ref()
        .and_then(|value| trace_number(value, "timing_recurrent_active_ms_per_step"));
    let inactive_trace = trace_json
        .as_ref()
        .and_then(|value| trace_number(value, "timing_recurrent_inactive_ms_per_step"));
    let recurrent_penalty = recurrent_default_active_penalty_ms(&spec);
    let active = active_trace.unwrap_or(total + recurrent_penalty);
    let inactive = inactive_trace.unwrap_or((total - recurrent_penalty * 0.25).max(1.0));
    let active_source = if active_trace.is_some() {
        "trace_field:timing_recurrent_active_ms_per_step"
    } else {
        "shape_model_default_recurrent_penalty"
    }
    .to_string();
    let inactive_source = if inactive_trace.is_some() {
        "trace_field:timing_recurrent_inactive_ms_per_step"
    } else {
        "shape_model_default_recurrent_penalty"
    }
    .to_string();
    let expected_steps = (600_000.0 / total.max(1.0)).floor() as usize;
    let expected_train_wall_seconds = spec.train.total_iterations as f64 * total.max(0.0) / 1_000.0;
    let artifact_bytes =
        compile_quant_layout_manifest(&spec.quant, Some(&spec.model.to_model_config()))?
            .estimated_raw_weight_bytes;
    let top_bottlenecks = top_bottlenecks(&stages, 3);
    let top = top_bottlenecks
        .first()
        .map(|stage| stage.stage.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let exact_boundary_fusion = exact_recurrent_boundary_fusion_status(&spec);
    let exact_boundary_layers = exact_recurrent_boundary_layers(&plan);
    let operation_dag = build_operation_dag(&plan, &exact_boundary_fusion);
    let recurrent_split = RecurrentSplitReport {
        recurrent_enabled: spec.model.recurrence.enabled,
        active_step_ms: active,
        inactive_step_ms: inactive,
        active_minus_inactive_ms: active - inactive,
        active_source,
        inactive_source,
        exact_boundary_fusion: exact_boundary_fusion.clone(),
        exact_boundary_layers: exact_boundary_layers.clone(),
    };
    let budget_status =
        wind_tunnel_budget_status(&spec, expected_train_wall_seconds, total, artifact_bytes);
    let risk_flags = wind_tunnel_risk_flags(
        &spec,
        &trace_coverage,
        &budget_status,
        active_trace,
        inactive_trace,
        &top,
    );
    let what_if_scenarios = wind_tunnel_what_if_scenarios(&spec, total, active, optimizer, ce);
    let experiment_rankings = wind_tunnel_experiment_rankings(
        &what_if_scenarios,
        &trace_coverage,
        active_trace,
        inactive_trace,
    );
    let recommendations = recommendations_for_wind_tunnel(&top_bottlenecks, &risk_flags);
    let recommendation = experiment_rankings
        .first()
        .map(|ranked| {
            format!(
                "try {} (estimated -{:.3} ms/step; train budget {})",
                ranked.scenario, ranked.delta_ms_per_step, ranked.train_budget_status
            )
        })
        .or_else(|| recommendations.first().cloned())
        .unwrap_or_else(|| {
            "collect a stage-timing trace and update the local calibration".to_string()
        });

    let report = WindTunnelReport {
        kind: "pg_local_wind_tunnel",
        spec_fingerprint: plan.variant_fingerprint,
        estimate_only: true,
        prediction_source,
        trace_path: options
            .trace
            .as_ref()
            .map(|path| path.display().to_string()),
        trace_total_ms,
        trace_coverage,
        recurrent_backward_profile: recurrent_profile_label(
            spec.runtime.recurrent_backward_profile,
        )
        .to_string(),
        exact_recurrent_boundary_fusion: exact_boundary_fusion,
        exact_recurrent_boundary_layers: exact_boundary_layers,
        persistent_cta_block_backward: "not_implemented_unclaimed".to_string(),
        train_step_ms_estimate: total,
        expected_train_wall_seconds,
        active_recurrent_step_ms_estimate: active,
        inactive_recurrent_step_ms_estimate: inactive,
        active_recurrent_trace_ms_per_step: active_trace,
        inactive_recurrent_trace_ms_per_step: inactive_trace,
        recurrent_split,
        optimizer_update_ms_estimate: optimizer,
        output_ce_ms_estimate: ce,
        expected_train_steps_in_600s: expected_steps,
        eval_time_ms_estimate: spec.eval.chunk_tokens as f64 / 64.0,
        artifact_bytes_estimate: artifact_bytes,
        budget_status,
        top_bottleneck: top.clone(),
        top_bottlenecks,
        next_recommended_experiment: recommendation,
        recommendations,
        risk_flags,
        what_if_scenarios,
        experiment_rankings,
        stage_estimates: stages,
        operation_dag,
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_lite(options: LiteRunOptions) -> PgResult<LiteRunReport> {
    let total_start = Instant::now();
    let config_start = Instant::now();
    let body = fs::read_to_string(&options.config)?;
    let spec: LiteSpec = toml::from_str(&body)
        .map_err(|err| PgError::DataFormat(format!("invalid PG-Lite TOML: {err}")))?;
    let config_load_wall_seconds = config_start.elapsed().as_secs_f64();
    let config_fingerprint = spec.config_fingerprint()?;
    if spec.score != "bpb" {
        return Err(PgError::InvalidOp(format!(
            "unsupported PG-Lite score {}; expected bpb",
            spec.score
        )));
    }
    let backend_start = Instant::now();
    let backend = select_lite_backend(&spec)?;
    let backend_select_wall_seconds = backend_start.elapsed().as_secs_f64();
    let data_start = Instant::now();
    let train = fs::read(&spec.data.train_path)?;
    let val = fs::read(&spec.data.val_path)?;
    let data_load_wall_seconds = data_start.elapsed().as_secs_f64();
    if val.is_empty() {
        return Err(PgError::DataFormat(
            "PG-Lite validation data is empty".into(),
        ));
    }

    validate_lite_spec(&spec)?;
    let train_start = Instant::now();
    let mut metal_eval: Option<LiteEvalStats> = None;
    let model = if backend.execution_backend == LiteBackendKind::MetalApple
        && spec.track == LiteTrack::ByteGolf
    {
        let metal = run_metal_lite_train_eval(&train, &val, &spec)?;
        metal_eval = Some(metal.eval);
        metal.model
    } else if spec.track == LiteTrack::ArtifactGolf {
        if let Some(path) = spec.model.artifact_path.as_deref() {
            let stored = load_lite_stored_artifact(Path::new(path))?;
            LiteNgramResidual::from_stored_artifact(&stored)?
        } else {
            LiteNgramResidual::train(&train, &spec)
        }
    } else {
        LiteNgramResidual::train(&train, &spec)
    };
    let measured_train_wall = train_start.elapsed().as_secs_f64();
    let train_wall = if spec.track == LiteTrack::ArtifactGolf {
        0.0
    } else {
        measured_train_wall
    };
    let eval_start = Instant::now();
    let eval = if let Some(eval) = metal_eval {
        eval
    } else if backend.execution_backend == LiteBackendKind::MetalApple {
        match spec.track {
            LiteTrack::ByteGolf | LiteTrack::ArtifactGolf => {
                run_metal_lite_fixed_eval(&model, &val)?
            }
            LiteTrack::StreamGolf => run_metal_lite_score_first_eval(&model, &val)?,
        }
    } else {
        match spec.track {
            LiteTrack::ByteGolf | LiteTrack::ArtifactGolf => model.loss_on_bytes(&val, false),
            LiteTrack::StreamGolf => model.loss_on_bytes_score_first(&val),
        }
    };
    let eval_wall = eval_start.elapsed().as_secs_f64();
    let avg_loss = if eval.tokens == 0 {
        f64::INFINITY
    } else {
        eval.loss / eval.tokens as f64
    };
    let bpb = compute_bpb(avg_loss, eval.tokens as f64, eval.tokens as f64);
    let artifact_bytes_estimate = model.artifact_bytes_estimate();
    let memory_bytes = model.memory_bytes_estimate();
    let memory_budget_ok = spec
        .memory_budget_bytes
        .map(|budget| memory_bytes <= budget);
    let memory_budget_status = match memory_budget_ok {
        Some(true) => "pass",
        Some(false) => "fail",
        None => "not_declared",
    }
    .to_string();
    let train_time_budget_ok = train_wall <= spec.train_time_seconds.max(0.0);
    let eval_time_budget_ok = eval_wall <= spec.eval_time_seconds.max(0.0);
    fs::create_dir_all(&options.output)?;
    let report_path = options.output.join("report.json");
    let artifact_manifest_path = options.output.join("artifact_manifest.json");
    let model_artifact_path = options.output.join("model.pglite.bin");
    let model_artifact_json_path = options.output.join("model.pglite.json");
    let stored_artifact = model.to_stored_artifact(config_fingerprint.clone());
    let artifact_json_start = Instant::now();
    let artifact_debug_json_bytes =
        write_lite_stored_artifact_json(&model_artifact_json_path, &stored_artifact)?;
    let artifact_json_write_wall_seconds = artifact_json_start.elapsed().as_secs_f64();
    let artifact_binary_start = Instant::now();
    let artifact_actual_bytes =
        write_lite_stored_artifact_binary(&model_artifact_path, &stored_artifact)?;
    let artifact_binary_write_wall_seconds = artifact_binary_start.elapsed().as_secs_f64();
    let artifact_budget_ok = artifact_actual_bytes <= spec.artifact_budget_bytes;
    let memory_budget_pass = memory_budget_ok.unwrap_or(true);
    let status = if bpb.is_finite()
        && artifact_budget_ok
        && memory_budget_pass
        && train_time_budget_ok
        && eval_time_budget_ok
    {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let train_bytes_used = if spec.track == LiteTrack::ArtifactGolf {
        0
    } else {
        train.len()
    };
    let track_behavior = spec.track_behavior();
    let score_scope = "local_proxy_bpb_not_leaderboard";
    let timing = LiteTimingReport {
        config_load_wall_seconds,
        backend_select_wall_seconds,
        data_load_wall_seconds,
        train_wall_seconds: train_wall,
        eval_wall_seconds: eval_wall,
        artifact_json_write_wall_seconds,
        artifact_binary_write_wall_seconds,
        manifest_write_wall_seconds: 0.0,
        total_wall_seconds_excluding_report_write: 0.0,
        report_write_excluded: true,
    };
    let manifest = LiteArtifactManifest {
        kind: "pg_lite_artifact_manifest",
        config_fingerprint: config_fingerprint.clone(),
        model_fingerprint: model.artifact_fingerprint(),
        track: spec.track,
        track_behavior,
        backend: backend.requested_backend,
        execution_backend: backend.execution_backend,
        backend_status: backend.status.clone(),
        backend_accelerated: backend.accelerated,
        backend_fallback_reason: backend.fallback_reason.clone(),
        model_family: model.family,
        score: spec.score.clone(),
        score_scope,
        train_bytes: train.len(),
        train_bytes_used,
        val_bytes: val.len(),
        artifact_budget_bytes: spec.artifact_budget_bytes,
        artifact_bytes_estimate,
        artifact_actual_bytes,
        artifact_path: "model.pglite.bin".to_string(),
        artifact_budget_ok,
        memory_budget_bytes: spec.memory_budget_bytes,
        memory_bytes_estimate: memory_bytes,
        memory_budget_status: memory_budget_status.clone(),
        score_first_tokens_scored: eval.score_first_tokens_scored,
        score_first_update_count: eval.score_first_update_count,
        outputs: vec![
            LiteOutputArtifact {
                path: "model.pglite.bin".to_string(),
                kind: "model_artifact_binary",
                stable_json: false,
            },
            LiteOutputArtifact {
                path: "model.pglite.json".to_string(),
                kind: "model_artifact_debug_json",
                stable_json: true,
            },
            LiteOutputArtifact {
                path: "report.json".to_string(),
                kind: "run_report",
                stable_json: false,
            },
            LiteOutputArtifact {
                path: "artifact_manifest.json".to_string(),
                kind: "artifact_manifest",
                stable_json: true,
            },
        ],
    };
    let _ = artifact_debug_json_bytes;
    let manifest_start = Instant::now();
    write_json_pretty(&artifact_manifest_path, &manifest)?;
    let manifest_write_wall_seconds = manifest_start.elapsed().as_secs_f64();
    let mut timing = timing;
    timing.manifest_write_wall_seconds = manifest_write_wall_seconds;
    timing.total_wall_seconds_excluding_report_write = total_start.elapsed().as_secs_f64();
    let report = LiteRunReport {
        kind: "pg_lite_run",
        config_path: options.config.display().to_string(),
        config_fingerprint,
        track: spec.track,
        track_behavior,
        backend: backend.requested_backend,
        execution_backend: backend.execution_backend,
        backend_status: backend.status,
        backend_accelerated: backend.accelerated,
        backend_fallback_reason: backend.fallback_reason,
        model_family: model.family,
        train_bytes: train.len(),
        train_bytes_used,
        val_bytes: val.len(),
        train_time_seconds_budget: spec.train_time_seconds,
        eval_time_seconds_budget: spec.eval_time_seconds,
        train_wall_seconds: train_wall,
        eval_wall_seconds: eval_wall,
        timing,
        artifact_budget_bytes: spec.artifact_budget_bytes,
        artifact_bytes_estimate,
        artifact_actual_bytes,
        artifact_path: model_artifact_path.display().to_string(),
        artifact_budget_ok,
        artifact_manifest_path: artifact_manifest_path.display().to_string(),
        memory_budget_bytes: spec.memory_budget_bytes,
        memory_bytes_estimate: memory_bytes,
        memory_budget_ok,
        memory_budget_status,
        train_time_budget_ok,
        eval_time_budget_ok,
        validation_bpb: bpb,
        validation_bpb_scope: score_scope,
        score_first_legal: spec.track != LiteTrack::StreamGolf
            || eval.score_first_update_count == eval.score_first_tokens_scored,
        score_first_tokens_scored: eval.score_first_tokens_scored,
        score_first_update_count: eval.score_first_update_count,
        proxy_only: true,
        status,
    };
    write_report_if_requested(&report, Some(&report_path))?;
    Ok(report)
}

pub fn run_lite_suite(options: LiteSuiteOptions) -> PgResult<LiteSuiteReport> {
    let mut configs = glob_paths(&options.config_dir, "*.toml")?;
    configs.sort();
    run_lite_suite_for_configs(
        options.config_dir.display().to_string(),
        options.output_dir,
        configs,
    )
}

pub fn run_lite_benchmark(options: LiteBenchmarkOptions) -> PgResult<LiteBenchmarkReport> {
    let repeats = options.repeats.max(1);
    fs::create_dir_all(&options.output_dir)?;
    let base_spec = load_lite_spec(&options.config)?;
    validate_lite_spec(&base_spec)?;
    let mut backends = Vec::new();
    for backend in [LiteBackendKind::CpuReference, LiteBackendKind::MetalApple] {
        backends.push(run_lite_benchmark_backend(
            &options.config,
            &base_spec,
            &options.output_dir,
            backend,
            repeats,
        )?);
    }
    let cpu = backends
        .iter()
        .find(|backend| backend.requested_backend == LiteBackendKind::CpuReference);
    let metal = backends
        .iter()
        .find(|backend| backend.requested_backend == LiteBackendKind::MetalApple);
    let cpu_vs_metal_speedup_total = cpu
        .and_then(|cpu| cpu.median_total_wall_seconds)
        .zip(metal.and_then(|metal| metal.median_total_wall_seconds))
        .and_then(|(cpu, metal)| (metal > 0.0).then_some(cpu / metal));
    let cpu_vs_metal_speedup_train = cpu
        .and_then(|cpu| cpu.median_train_wall_seconds)
        .zip(metal.and_then(|metal| metal.median_train_wall_seconds))
        .and_then(|(cpu, metal)| (metal > 0.0).then_some(cpu / metal));
    let cpu_vs_metal_bpb_delta = cpu
        .and_then(|cpu| cpu.median_validation_bpb)
        .zip(metal.and_then(|metal| metal.median_validation_bpb))
        .map(|(cpu, metal)| metal - cpu);
    let best_total_wall_backend = backends
        .iter()
        .filter_map(|backend| {
            backend
                .median_total_wall_seconds
                .map(|seconds| (backend.requested_backend, seconds))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(backend, _)| backend);
    let best_bpb_backend = backends
        .iter()
        .filter_map(|backend| {
            backend
                .median_validation_bpb
                .map(|bpb| (backend.requested_backend, bpb))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(backend, _)| backend);
    let summary_json_path = options.output_dir.join("summary.json");
    let summary_markdown_path = options.output_dir.join("summary.md");
    let status = if backends.iter().any(|backend| {
        backend.requested_backend == LiteBackendKind::CpuReference && !backend.runs.is_empty()
    }) {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = LiteBenchmarkReport {
        kind: "pg_lite_benchmark",
        config_path: options.config.display().to_string(),
        output_dir: options.output_dir.display().to_string(),
        repeats,
        backends,
        cpu_vs_metal_speedup_total,
        cpu_vs_metal_speedup_train,
        cpu_vs_metal_bpb_delta,
        best_total_wall_backend,
        best_bpb_backend,
        summary_json_path: summary_json_path.display().to_string(),
        summary_markdown_path: summary_markdown_path.display().to_string(),
        status,
    };
    write_json_pretty(&summary_json_path, &report)?;
    fs::write(&summary_markdown_path, lite_benchmark_markdown(&report))?;
    Ok(report)
}

fn run_lite_benchmark_backend(
    config_path: &Path,
    base_spec: &LiteSpec,
    output_dir: &Path,
    backend: LiteBackendKind,
    repeats: usize,
) -> PgResult<LiteBenchmarkBackendReport> {
    let backend_dir = output_dir.join(lite_backend_label(backend));
    fs::create_dir_all(&backend_dir)?;
    let mut spec = base_spec.clone();
    spec.backend.kind = backend;
    spec.backend.allow_cpu_fallback = false;
    let config_copy = backend_dir.join("config.toml");
    fs::write(
        &config_copy,
        toml::to_string_pretty(&spec).map_err(|err| {
            PgError::DataFormat(format!("PG-Lite benchmark TOML encode failed: {err}"))
        })?,
    )?;
    let mut runs = Vec::new();
    let mut first_error = None;
    for run_index in 0..repeats {
        let run_output = backend_dir.join(format!("run_{run_index:03}"));
        match run_lite(LiteRunOptions {
            config: config_copy.clone(),
            output: run_output.clone(),
        }) {
            Ok(report) => runs.push(LiteBenchmarkRunReport {
                run_index,
                output_dir: run_output.display().to_string(),
                report_path: run_output.join("report.json").display().to_string(),
                status: report.status,
                execution_backend: report.execution_backend,
                backend_accelerated: report.backend_accelerated,
                validation_bpb: report.validation_bpb,
                total_wall_seconds: report.timing.total_wall_seconds_excluding_report_write,
                train_wall_seconds: report.train_wall_seconds,
                eval_wall_seconds: report.eval_wall_seconds,
                artifact_actual_bytes: report.artifact_actual_bytes,
            }),
            Err(err) => {
                first_error = Some(format!("{err}"));
                break;
            }
        }
    }
    let total = runs
        .iter()
        .map(|run| run.total_wall_seconds)
        .collect::<Vec<_>>();
    let train = runs
        .iter()
        .map(|run| run.train_wall_seconds)
        .collect::<Vec<_>>();
    let eval = runs
        .iter()
        .map(|run| run.eval_wall_seconds)
        .collect::<Vec<_>>();
    let bpb = runs
        .iter()
        .map(|run| run.validation_bpb)
        .collect::<Vec<_>>();
    let mut execution_backends = runs
        .iter()
        .map(|run| run.execution_backend)
        .collect::<Vec<_>>();
    execution_backends.sort_by_key(|backend| lite_backend_label(*backend));
    execution_backends.dedup();
    let accelerated_run_count = runs.iter().filter(|run| run.backend_accelerated).count();
    let artifact_actual_bytes = runs.first().map(|run| run.artifact_actual_bytes);
    let status = if first_error.is_none() && runs.len() == repeats {
        "pass"
    } else if runs.is_empty() {
        "unavailable"
    } else {
        "partial"
    }
    .to_string();
    Ok(LiteBenchmarkBackendReport {
        requested_backend: backend,
        status,
        error: first_error.map(|err| {
            format!(
                "{} benchmark failed for {}: {err}",
                config_path.display(),
                lite_backend_label(backend)
            )
        }),
        runs,
        run_count: total.len(),
        accelerated_run_count,
        execution_backends,
        median_total_wall_seconds: median_f64(total),
        median_train_wall_seconds: median_f64(train),
        median_eval_wall_seconds: median_f64(eval),
        median_validation_bpb: median_f64(bpb),
        artifact_actual_bytes,
    })
}

fn run_lite_suite_for_configs(
    config_dir_label: String,
    output_dir: PathBuf,
    configs: Vec<PathBuf>,
) -> PgResult<LiteSuiteReport> {
    fs::create_dir_all(&output_dir)?;
    let mut runs = Vec::with_capacity(configs.len());
    for config in configs.iter() {
        let stem = config
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(sanitize_path_component)
            .unwrap_or_else(|| "config".to_string());
        let run_output_dir = output_dir.join(stem);
        let report = run_lite(LiteRunOptions {
            config: config.clone(),
            output: run_output_dir.clone(),
        })?;
        runs.push(LiteSuiteRunSummary {
            config_path: config.display().to_string(),
            output_dir: run_output_dir.display().to_string(),
            track: report.track,
            model_family: report.model_family,
            requested_backend: report.backend,
            execution_backend: report.execution_backend,
            backend_status: report.backend_status,
            backend_accelerated: report.backend_accelerated,
            status: report.status,
            validation_bpb: report.validation_bpb,
            artifact_bytes_estimate: report.artifact_bytes_estimate,
            artifact_actual_bytes: report.artifact_actual_bytes,
            artifact_budget_ok: report.artifact_budget_ok,
            memory_budget_status: report.memory_budget_status,
            score_first_tokens_scored: report.score_first_tokens_scored,
            score_first_update_count: report.score_first_update_count,
        });
    }

    let best = runs
        .iter()
        .filter(|run| run.status == "pass" && run.validation_bpb.is_finite())
        .min_by(|a, b| {
            a.validation_bpb
                .partial_cmp(&b.validation_bpb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let best_bpb = best.map(|run| run.validation_bpb);
    let best_config = best.map(|run| run.config_path.clone());
    let summary_json_path = output_dir.join("summary.json");
    let summary_markdown_path = output_dir.join("summary.md");
    let status = if !runs.is_empty() && runs.iter().all(|run| run.status == "pass") {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = LiteSuiteReport {
        kind: "pg_lite_suite",
        config_dir: config_dir_label,
        output_dir: output_dir.display().to_string(),
        configs_found: runs.len(),
        runs,
        best_bpb,
        best_config,
        summary_json_path: summary_json_path.display().to_string(),
        summary_markdown_path: summary_markdown_path.display().to_string(),
        status,
    };
    write_json_pretty(&summary_json_path, &report)?;
    fs::write(&summary_markdown_path, lite_suite_markdown(&report))?;
    Ok(report)
}

pub fn run_wind_tunnel_suite(options: WindTunnelSuiteOptions) -> PgResult<WindTunnelSuiteReport> {
    fs::create_dir_all(&options.output_dir)?;
    let mut traces = wind_tunnel_trace_paths(&options.trace_dir)?;
    traces.sort();

    let baseline_report_path = options.output_dir.join("baseline_shape_model.json");
    let baseline = run_wind_tunnel(WindTunnelOptions {
        spec: options.spec.clone(),
        trace: None,
        output: Some(baseline_report_path.clone()),
    })?;
    let baseline_step = baseline.train_step_ms_estimate;

    let mut reports = Vec::with_capacity(traces.len());
    for trace in traces.iter() {
        let stem = trace
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(sanitize_path_component)
            .unwrap_or_else(|| "trace".to_string());
        let report_path = options.output_dir.join(format!("{stem}.wind_tunnel.json"));
        let report = run_wind_tunnel(WindTunnelOptions {
            spec: options.spec.clone(),
            trace: Some(trace.clone()),
            output: Some(report_path.clone()),
        })?;
        let abs_error = report
            .trace_total_ms
            .map(|trace_ms| (baseline_step - trace_ms).abs());
        let pct_error = abs_error
            .zip(report.trace_total_ms)
            .and_then(|(err, trace_ms)| {
                if trace_ms.abs() > f64::EPSILON {
                    Some(err / trace_ms.abs() * 100.0)
                } else {
                    None
                }
            });
        reports.push(WindTunnelSuiteTraceSummary {
            trace_path: trace.display().to_string(),
            report_path: report_path.display().to_string(),
            prediction_source: report.prediction_source.clone(),
            trace_signal_quality: report.trace_coverage.signal_quality.clone(),
            trace_total_ms: report.trace_total_ms,
            trace_coverage_ratio: report.trace_coverage.coverage_ratio,
            useful_stage_coverage_ratio: report.trace_coverage.useful_stage_coverage_ratio,
            nonzero_stage_field_count: report.trace_coverage.nonzero_mapped_stage_count,
            calibration_role: trace_calibration_role(&report),
            train_step_ms_estimate: report.train_step_ms_estimate,
            baseline_abs_step_error_ms: abs_error,
            baseline_pct_error: pct_error,
            top_bottleneck: report.top_bottleneck,
            risk_flags: report.risk_flags,
        });
    }

    let errors = reports
        .iter()
        .filter_map(|report| report.baseline_abs_step_error_ms)
        .collect::<Vec<_>>();
    let pct_errors = reports
        .iter()
        .filter_map(|report| report.baseline_pct_error)
        .collect::<Vec<_>>();
    let traces_with_step_timing = reports
        .iter()
        .filter(|report| report.trace_total_ms.is_some())
        .count();
    let traces_with_stage_attribution = reports
        .iter()
        .filter(|report| {
            report.calibration_role == "stage_calibration"
                || report.calibration_role == "partial_stage_calibration"
        })
        .count();
    let traces_total_only = reports
        .iter()
        .filter(|report| report.calibration_role == "total_calibration")
        .count();
    let traces_shape_only = reports
        .iter()
        .filter(|report| report.calibration_role == "shape_model_only")
        .count();
    let summary_json_path = options.output_dir.join("summary.json");
    let summary_markdown_path = options.output_dir.join("summary.md");
    let status = if traces_with_step_timing > 0 {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = WindTunnelSuiteReport {
        kind: "pg_local_wind_tunnel_suite",
        spec_path: options.spec.display().to_string(),
        trace_dir: options.trace_dir.display().to_string(),
        output_dir: options.output_dir.display().to_string(),
        traces_found: reports.len(),
        traces_with_step_timing,
        traces_with_stage_attribution,
        traces_total_only,
        traces_shape_only,
        baseline_report_path: baseline_report_path.display().to_string(),
        baseline_train_step_ms_estimate: baseline_step,
        reports,
        mean_abs_baseline_step_error_ms: mean_f64(&errors),
        mean_abs_baseline_pct_error: mean_f64(&pct_errors),
        summary_json_path: summary_json_path.display().to_string(),
        summary_markdown_path: summary_markdown_path.display().to_string(),
        status,
    };
    write_json_pretty(&summary_json_path, &report)?;
    fs::write(&summary_markdown_path, wind_tunnel_suite_markdown(&report))?;
    Ok(report)
}

pub fn run_wind_tunnel_report(
    options: WindTunnelReviewerOptions,
) -> PgResult<WindTunnelReviewerReport> {
    fs::create_dir_all(&options.output_dir)?;

    let trace_index_path = options.output_dir.join("trace_index.json");
    let corpus_path = options.output_dir.join("trace_corpus.json");
    let calibration_path = options.output_dir.join("calibration_model.json");
    let scenario_path = options
        .output_dir
        .join("scenario_exact_recurrent_replay.json");
    let suite_dir = options.output_dir.join("suite");
    let report_json_path = options.output_dir.join("report.json");
    let report_markdown_path = options.output_dir.join("report.md");

    let index_scratch = scratch_report_path("pg_local_report_trace_index");
    let corpus_scratch = scratch_report_path("pg_local_report_trace_corpus");
    let calibration_scratch = scratch_report_path("pg_local_report_calibration");
    let scenario_scratch = scratch_report_path("pg_local_report_scenario");

    let index = run_trace_index(TraceIndexOptions {
        spec: options.spec.clone(),
        trace_dir: options.trace_dir.clone(),
        output: Some(index_scratch.clone()),
    })?;
    let corpus = run_wind_tunnel_corpus(WindTunnelCorpusOptions {
        spec: options.spec.clone(),
        trace_dir: options.trace_dir.clone(),
        output: Some(corpus_scratch.clone()),
    })?;
    let calibration = run_wind_tunnel_calibrate(WindTunnelCalibrationOptions {
        spec: options.spec.clone(),
        trace_dir: None,
        corpus: Some(corpus_scratch.clone()),
        output: Some(calibration_scratch.clone()),
    })?;
    let scenario = run_wind_tunnel_scenario(WindTunnelScenarioOptions {
        spec: options.spec.clone(),
        calibration: Some(calibration_scratch.clone()),
        trace: None,
        active_recurrent_replay_cut_ms: options.active_recurrent_replay_cut_ms,
        bank_update_cut_ms: 0.0,
        graph_overhead_cut_ms: 0.0,
        target_step_ms: options.target_step_ms,
        output: Some(scenario_scratch.clone()),
    })?;
    let suite = run_wind_tunnel_suite(WindTunnelSuiteOptions {
        spec: options.spec.clone(),
        trace_dir: options.trace_dir.clone(),
        output_dir: suite_dir.clone(),
    })?;

    let _ = fs::remove_file(index_scratch);
    let _ = fs::remove_file(corpus_scratch);
    let _ = fs::remove_file(calibration_scratch);
    let _ = fs::remove_file(scenario_scratch);

    write_json_pretty(&trace_index_path, &index)?;
    write_json_pretty(&corpus_path, &corpus)?;
    write_json_pretty(&calibration_path, &calibration)?;
    write_json_pretty(&scenario_path, &scenario)?;

    let mut top_stage_metrics = calibration.stage_metrics.clone();
    top_stage_metrics.sort_by(|a, b| {
        b.estimate_ms
            .partial_cmp(&a.estimate_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_stage_metrics.truncate(8);

    let holdout_validation = wind_tunnel_holdout_report(&corpus);
    let exact_2135_timed_runs = corpus
        .runs
        .iter()
        .filter(|run| corpus_run_is_exact_2135(run) && run.trace_total_ms.is_some())
        .count();
    let mut warnings = calibration.warnings.clone();
    if holdout_validation.status != "pass" {
        push_unique(
            &mut warnings,
            format!(
                "holdout validation status is {}; collect more timed traces for a stronger local predictor",
                holdout_validation.status
            ),
        );
    }
    if suite.status != "pass" {
        push_unique(
            &mut warnings,
            "Wind Tunnel suite did not find measured step timing; reviewer report is shape-model dominated"
                .to_string(),
        );
    }
    if exact_2135_timed_runs == 0 {
        push_unique(
            &mut warnings,
            "no timed exact #2135 traces found in the corpus".to_string(),
        );
    }
    if corpus.canonical_caseops_runs == 0 {
        push_unique(
            &mut warnings,
            "no canonical CaseOps trace found; this report cannot prove record data compliance"
                .to_string(),
        );
    }

    let mut recommended_next_experiment = scenario.recommendation.clone();
    if !calibration.recommended_trace_to_collect.is_empty() {
        recommended_next_experiment.push_str("; next trace to collect: ");
        recommended_next_experiment.push_str(&calibration.recommended_trace_to_collect);
    }
    let status = if corpus.timed_runs > 0 && calibration.total_step_ms.is_some() {
        "pass"
    } else if index.parseable_traces > 0 {
        "partial"
    } else {
        "fail"
    }
    .to_string();
    let report = WindTunnelReviewerReport {
        kind: "pg_local_wind_tunnel_reviewer_report",
        spec_path: options.spec.display().to_string(),
        trace_dir: options.trace_dir.display().to_string(),
        output_dir: options.output_dir.display().to_string(),
        report_json_path: report_json_path.display().to_string(),
        report_markdown_path: report_markdown_path.display().to_string(),
        trace_index_path: trace_index_path.display().to_string(),
        corpus_path: corpus_path.display().to_string(),
        calibration_path: calibration_path.display().to_string(),
        scenario_path: scenario_path.display().to_string(),
        suite_summary_path: suite.summary_json_path.clone(),
        traces_found: index.traces_found,
        parseable_traces: index.parseable_traces,
        deduped_runs: corpus.deduped_runs,
        timed_runs: corpus.timed_runs,
        exact_2135_timed_runs,
        stage_calibration_runs: corpus.stage_calibration_runs,
        canonical_caseops_runs: corpus.canonical_caseops_runs,
        best_exact_2135_ms: corpus.best_exact_2135_ms,
        best_active_recurrent_ms: corpus.best_active_recurrent_ms,
        calibration_confidence_summary: calibration.confidence_summary.clone(),
        holdout_validation,
        scenario_clears_target: scenario.clears_target,
        scenario_predicted_step_ms: scenario.predicted_step_ms,
        recommended_next_experiment,
        top_stage_metrics,
        warnings,
        status,
    };
    write_json_pretty(&report_json_path, &report)?;
    fs::write(
        &report_markdown_path,
        wind_tunnel_reviewer_markdown(&report, &index, &corpus, &calibration, &scenario, &suite),
    )?;
    Ok(report)
}

pub fn run_trace_index(options: TraceIndexOptions) -> PgResult<TraceIndexReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let mut base_stages = default_stage_estimates(&plan);
    annotate_stage_percentages(&mut base_stages, 1.0);
    let traces = wind_tunnel_trace_paths(&options.trace_dir)?;
    let mut entries = Vec::with_capacity(traces.len());
    for path in traces {
        let body = fs::read_to_string(&path).unwrap_or_default();
        let selected = parse_trace_body(&body);
        let audit = parse_trace_aux_body(&body);
        let Some(trace) = selected.as_ref() else {
            entries.push(TraceIndexEntry {
                trace_path: path.display().to_string(),
                parsed_trace: false,
                calibration_role: "unparseable".to_string(),
                signal_quality: "unparseable".to_string(),
                trace_total_ms: None,
                active_recurrent_ms_per_step: None,
                inactive_recurrent_ms_per_step: None,
                active_recurrent_steps: None,
                timing_steps: None,
                steps_completed: None,
                run_name: None,
                mode: None,
                record_profile: None,
                recurrent_backward_profile: None,
                cuda_graph_profile: None,
                world_size: None,
                seq_len: None,
                canonical_caseops_dataset: None,
                train_shards: None,
                val_tokens: None,
                f32_to_bf16_bridge_launches: None,
                bf16_to_f32_bridge_launches: None,
                host_batch_flatten_calls: None,
                host_to_device_batch_bytes: None,
                useful_stage_coverage_ratio: 0.0,
                traced_stage_ms_ratio: 0.0,
                nonzero_stage_fields: Vec::new(),
                zero_stage_fields: Vec::new(),
                tags: vec!["unparseable".to_string()],
            });
            continue;
        };
        let mut stages = base_stages.clone();
        apply_trace_to_stages(trace, &mut stages);
        let total = trace_number_for_aliases(
            trace,
            &[
                "timing_measured_ms_per_step",
                "timing_train_step_ms_per_step",
            ],
        )
        .unwrap_or_else(|| stages.iter().map(|stage| stage.ms).sum::<f64>().max(1.0));
        let coverage = trace_coverage_report(Some(trace), &stages, total);
        let mut report_stub = WindTunnelReport {
            kind: "pg_local_wind_tunnel",
            spec_fingerprint: String::new(),
            estimate_only: true,
            prediction_source: prediction_source(
                trace_number_for_aliases(
                    trace,
                    &[
                        "timing_measured_ms_per_step",
                        "timing_train_step_ms_per_step",
                    ],
                ),
                &coverage,
            ),
            trace_path: Some(path.display().to_string()),
            trace_total_ms: trace_number_for_aliases(
                trace,
                &[
                    "timing_measured_ms_per_step",
                    "timing_train_step_ms_per_step",
                ],
            ),
            trace_coverage: coverage.clone(),
            recurrent_backward_profile: String::new(),
            exact_recurrent_boundary_fusion: String::new(),
            exact_recurrent_boundary_layers: Vec::new(),
            persistent_cta_block_backward: String::new(),
            train_step_ms_estimate: total,
            expected_train_wall_seconds: 0.0,
            active_recurrent_step_ms_estimate: 0.0,
            inactive_recurrent_step_ms_estimate: 0.0,
            active_recurrent_trace_ms_per_step: None,
            inactive_recurrent_trace_ms_per_step: None,
            recurrent_split: RecurrentSplitReport {
                recurrent_enabled: false,
                active_step_ms: 0.0,
                inactive_step_ms: 0.0,
                active_minus_inactive_ms: 0.0,
                active_source: String::new(),
                inactive_source: String::new(),
                exact_boundary_fusion: String::new(),
                exact_boundary_layers: Vec::new(),
            },
            optimizer_update_ms_estimate: 0.0,
            output_ce_ms_estimate: 0.0,
            expected_train_steps_in_600s: 0,
            eval_time_ms_estimate: 0.0,
            artifact_bytes_estimate: None,
            budget_status: Vec::new(),
            top_bottleneck: String::new(),
            top_bottlenecks: Vec::new(),
            next_recommended_experiment: String::new(),
            recommendations: Vec::new(),
            risk_flags: Vec::new(),
            what_if_scenarios: Vec::new(),
            experiment_rankings: Vec::new(),
            stage_estimates: Vec::new(),
            operation_dag: Vec::new(),
        };
        let calibration_role = trace_calibration_role(&report_stub);
        report_stub.trace_coverage = coverage.clone();
        let trace_total_ms = trace_number_for_aliases(
            trace,
            &[
                "timing_measured_ms_per_step",
                "timing_train_step_ms_per_step",
            ],
        );
        let active = trace_number(trace, "timing_recurrent_active_ms_per_step");
        let inactive = trace_number(trace, "timing_recurrent_inactive_ms_per_step");
        let canonical = trace_bool_chain(
            trace,
            audit.as_ref(),
            &["canonical_caseops_dataset", "frontier_2135_dataset_matches"],
        );
        let mut tags = trace_tags(
            &path,
            trace_total_ms,
            active,
            &calibration_role,
            canonical,
            trace_string_chain(trace, audit.as_ref(), &["mode"]),
            trace_string_chain(trace, audit.as_ref(), &["recurrent_backward_profile"]),
        );
        tags.sort();
        tags.dedup();
        entries.push(TraceIndexEntry {
            trace_path: path.display().to_string(),
            parsed_trace: true,
            calibration_role,
            signal_quality: coverage.signal_quality.clone(),
            trace_total_ms,
            active_recurrent_ms_per_step: active,
            inactive_recurrent_ms_per_step: inactive,
            active_recurrent_steps: trace_usize(trace, "timing_recurrent_active_steps"),
            timing_steps: trace_usize(trace, "timing_steps"),
            steps_completed: trace_usize_chain(trace, audit.as_ref(), &["steps_completed"]),
            run_name: trace_string_chain(trace, audit.as_ref(), &["run_name", "exact_profile"]),
            mode: trace_string_chain(trace, audit.as_ref(), &["mode"]),
            record_profile: trace_string_chain(trace, audit.as_ref(), &["record_profile"]),
            recurrent_backward_profile: trace_string_chain(
                trace,
                audit.as_ref(),
                &["recurrent_backward_profile"],
            ),
            cuda_graph_profile: trace_string_chain(trace, audit.as_ref(), &["cuda_graph_profile"]),
            world_size: trace_usize_chain(trace, audit.as_ref(), &["world_size"]),
            seq_len: trace_usize_chain(trace, audit.as_ref(), &["seq_len"]),
            canonical_caseops_dataset: canonical,
            train_shards: trace_usize_chain(trace, audit.as_ref(), &["train_shards"]),
            val_tokens: trace_usize_chain(
                trace,
                audit.as_ref(),
                &["val_tokens", "preflight_val_tokens"],
            ),
            f32_to_bf16_bridge_launches: trace_usize_chain(
                trace,
                audit.as_ref(),
                &["f32_to_bf16_bridge_launches"],
            ),
            bf16_to_f32_bridge_launches: trace_usize_chain(
                trace,
                audit.as_ref(),
                &["bf16_to_f32_bridge_launches"],
            ),
            host_batch_flatten_calls: trace_usize_chain(
                trace,
                audit.as_ref(),
                &["host_batch_flatten_calls"],
            ),
            host_to_device_batch_bytes: trace_usize_chain(
                trace,
                audit.as_ref(),
                &["host_to_device_batch_bytes"],
            ),
            useful_stage_coverage_ratio: coverage.useful_stage_coverage_ratio,
            traced_stage_ms_ratio: coverage.traced_stage_ms_ratio,
            nonzero_stage_fields: coverage.nonzero_stage_fields,
            zero_stage_fields: coverage.zero_stage_fields,
            tags,
        });
    }
    let parseable_traces = entries.iter().filter(|entry| entry.parsed_trace).count();
    let traces_with_step_timing = entries
        .iter()
        .filter(|entry| entry.trace_total_ms.is_some())
        .count();
    let stage_calibration_traces = entries
        .iter()
        .filter(|entry| entry.calibration_role == "stage_calibration")
        .count();
    let partial_stage_calibration_traces = entries
        .iter()
        .filter(|entry| entry.calibration_role == "partial_stage_calibration")
        .count();
    let total_calibration_traces = entries
        .iter()
        .filter(|entry| entry.calibration_role == "total_calibration")
        .count();
    let shape_model_only_traces = entries
        .iter()
        .filter(|entry| entry.calibration_role == "shape_model_only")
        .count();
    let canonical_caseops_traces = entries
        .iter()
        .filter(|entry| entry.canonical_caseops_dataset == Some(true))
        .count();
    let best = entries
        .iter()
        .filter_map(|entry| {
            entry
                .trace_total_ms
                .map(|ms| (ms, entry.trace_path.clone()))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let fastest_exact = entries
        .iter()
        .filter(|entry| {
            entry
                .tags
                .iter()
                .any(|tag| tag == "frontier_2135" || tag == "exact_2135")
        })
        .filter_map(|entry| {
            entry
                .trace_total_ms
                .map(|ms| (ms, entry.trace_path.clone()))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let active_best = entries
        .iter()
        .filter_map(|entry| {
            entry
                .active_recurrent_ms_per_step
                .map(|ms| (ms, entry.trace_path.clone()))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let report = TraceIndexReport {
        kind: "pg_local_trace_index",
        spec_path: options.spec.display().to_string(),
        trace_dir: options.trace_dir.display().to_string(),
        traces_found: entries.len(),
        parseable_traces,
        traces_with_step_timing,
        stage_calibration_traces,
        partial_stage_calibration_traces,
        total_calibration_traces,
        shape_model_only_traces,
        canonical_caseops_traces,
        best_step_ms: best.as_ref().map(|(ms, _)| *ms),
        best_step_trace: best.map(|(_, path)| path),
        fastest_exact_2135_ms: fastest_exact.as_ref().map(|(ms, _)| *ms),
        fastest_exact_2135_trace: fastest_exact.map(|(_, path)| path),
        active_recurrent_best_ms: active_best.as_ref().map(|(ms, _)| *ms),
        active_recurrent_best_trace: active_best.map(|(_, path)| path),
        entries,
        status: if traces_with_step_timing > 0 {
            "pass"
        } else {
            "fail"
        }
        .to_string(),
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_wind_tunnel_corpus(
    options: WindTunnelCorpusOptions,
) -> PgResult<WindTunnelCorpusReport> {
    let index_scratch = std::env::temp_dir().join(format!(
        "pg_local_trace_index_{}_{}.json",
        std::process::id(),
        monotonic_nanos()
    ));
    let index = run_trace_index(TraceIndexOptions {
        spec: options.spec.clone(),
        trace_dir: options.trace_dir.clone(),
        output: Some(index_scratch.clone()),
    })?;
    let _ = fs::remove_file(index_scratch);
    let mut groups: BTreeMap<String, Vec<TraceIndexEntry>> = BTreeMap::new();
    for entry in index.entries.into_iter().filter(|entry| entry.parsed_trace) {
        groups
            .entry(trace_entry_run_key(&entry))
            .or_default()
            .push(entry);
    }
    let mut runs = Vec::with_capacity(groups.len());
    for (run_key, mut entries) in groups {
        entries.sort_by_key(|entry| std::cmp::Reverse(trace_entry_selection_score(entry)));
        let selected = entries
            .first()
            .cloned()
            .expect("trace corpus group must contain at least one entry");
        let mut source_paths = entries
            .iter()
            .map(|entry| entry.trace_path.clone())
            .collect::<Vec<_>>();
        source_paths.sort();
        source_paths.dedup();
        runs.push(WindTunnelCorpusRun {
            run_key,
            selected_trace_path: selected.trace_path.clone(),
            source_paths,
            selection_score: trace_entry_selection_score(&selected),
            selection_reason: trace_entry_selection_reason(&selected),
            calibration_role: selected.calibration_role,
            signal_quality: selected.signal_quality,
            trace_total_ms: selected.trace_total_ms,
            active_recurrent_ms_per_step: selected.active_recurrent_ms_per_step,
            inactive_recurrent_ms_per_step: selected.inactive_recurrent_ms_per_step,
            active_recurrent_steps: selected.active_recurrent_steps,
            timing_steps: selected.timing_steps,
            steps_completed: selected.steps_completed,
            run_name: selected.run_name,
            mode: selected.mode,
            record_profile: selected.record_profile,
            recurrent_backward_profile: selected.recurrent_backward_profile,
            cuda_graph_profile: selected.cuda_graph_profile,
            world_size: selected.world_size,
            seq_len: selected.seq_len,
            canonical_caseops_dataset: selected.canonical_caseops_dataset,
            train_shards: selected.train_shards,
            val_tokens: selected.val_tokens,
            f32_to_bf16_bridge_launches: selected.f32_to_bf16_bridge_launches,
            bf16_to_f32_bridge_launches: selected.bf16_to_f32_bridge_launches,
            host_batch_flatten_calls: selected.host_batch_flatten_calls,
            host_to_device_batch_bytes: selected.host_to_device_batch_bytes,
            useful_stage_coverage_ratio: selected.useful_stage_coverage_ratio,
            traced_stage_ms_ratio: selected.traced_stage_ms_ratio,
            nonzero_stage_fields: selected.nonzero_stage_fields,
            zero_stage_fields: selected.zero_stage_fields,
            tags: selected.tags,
        });
    }
    runs.sort_by(|a, b| {
        a.selected_trace_path
            .cmp(&b.selected_trace_path)
            .then_with(|| a.run_key.cmp(&b.run_key))
    });
    let timed_runs = runs
        .iter()
        .filter(|run| run.trace_total_ms.is_some())
        .count();
    let stage_calibration_runs = runs
        .iter()
        .filter(|run| run.calibration_role == "stage_calibration")
        .count();
    let partial_stage_calibration_runs = runs
        .iter()
        .filter(|run| run.calibration_role == "partial_stage_calibration")
        .count();
    let total_calibration_runs = runs
        .iter()
        .filter(|run| run.calibration_role == "total_calibration")
        .count();
    let exact_2135_runs = runs
        .iter()
        .filter(|run| corpus_run_is_exact_2135(run))
        .count();
    let canonical_caseops_runs = runs
        .iter()
        .filter(|run| run.canonical_caseops_dataset == Some(true))
        .count();
    let allst_diagnostic_runs = runs.iter().filter(|run| corpus_run_is_allst(run)).count();
    let best_exact = runs
        .iter()
        .filter(|run| corpus_run_is_exact_2135(run))
        .filter_map(|run| run.trace_total_ms.map(|ms| (ms, run.run_key.clone())))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let best_active = runs
        .iter()
        .filter_map(|run| {
            run.active_recurrent_ms_per_step
                .map(|ms| (ms, run.run_key.clone()))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let duplicate_source_files = index.parseable_traces.saturating_sub(runs.len());
    let report = WindTunnelCorpusReport {
        kind: "pg_local_wind_tunnel_corpus".to_string(),
        spec_path: options.spec.display().to_string(),
        trace_dir: options.trace_dir.display().to_string(),
        traces_found: index.traces_found,
        parseable_traces: index.parseable_traces,
        deduped_runs: runs.len(),
        duplicate_source_files,
        timed_runs,
        stage_calibration_runs,
        partial_stage_calibration_runs,
        total_calibration_runs,
        exact_2135_runs,
        canonical_caseops_runs,
        allst_diagnostic_runs,
        best_exact_2135_ms: best_exact.as_ref().map(|(ms, _)| *ms),
        best_exact_2135_run_key: best_exact.map(|(_, key)| key),
        best_active_recurrent_ms: best_active.as_ref().map(|(ms, _)| *ms),
        best_active_recurrent_run_key: best_active.map(|(_, key)| key),
        runs,
        status: "pass".to_string(),
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_wind_tunnel_calibrate(
    options: WindTunnelCalibrationOptions,
) -> PgResult<WindTunnelCalibrationReport> {
    let corpus = if let Some(path) = options.corpus.as_ref() {
        serde_json::from_str::<WindTunnelCorpusReport>(&fs::read_to_string(path)?)
            .map_err(|err| PgError::DataFormat(format!("invalid Wind Tunnel corpus JSON: {err}")))?
    } else if let Some(trace_dir) = options.trace_dir.as_ref() {
        run_wind_tunnel_corpus(WindTunnelCorpusOptions {
            spec: options.spec.clone(),
            trace_dir: trace_dir.clone(),
            output: None,
        })?
    } else {
        return Err(PgError::InvalidOp(
            "wind-tunnel calibrate requires --trace-dir or --corpus".into(),
        ));
    };
    let timed = corpus
        .runs
        .iter()
        .filter_map(|run| run.trace_total_ms.map(|ms| (run.run_key.clone(), ms)))
        .collect::<Vec<_>>();
    let exact = corpus
        .runs
        .iter()
        .filter(|run| corpus_run_is_exact_2135(run))
        .filter_map(|run| run.trace_total_ms.map(|ms| (run.run_key.clone(), ms)))
        .collect::<Vec<_>>();
    let active = corpus
        .runs
        .iter()
        .filter_map(|run| {
            run.active_recurrent_ms_per_step
                .map(|ms| (run.run_key.clone(), ms))
        })
        .collect::<Vec<_>>();
    let inactive = corpus
        .runs
        .iter()
        .filter_map(|run| {
            run.inactive_recurrent_ms_per_step
                .map(|ms| (run.run_key.clone(), ms))
        })
        .collect::<Vec<_>>();
    let recurrent_delta = corpus
        .runs
        .iter()
        .filter_map(|run| {
            run.active_recurrent_ms_per_step
                .zip(run.inactive_recurrent_ms_per_step)
                .map(|(active, inactive)| (run.run_key.clone(), active - inactive))
        })
        .collect::<Vec<_>>();
    let stage_metrics = calibrated_stage_metrics(&corpus.runs);
    let mut warnings = Vec::new();
    if corpus.stage_calibration_runs < 2 {
        warnings.push(
            "fewer than two rich stage-calibration traces; use stage estimates as low-confidence attribution"
                .to_string(),
        );
    }
    if active.is_empty() || inactive.is_empty() {
        warnings.push(
            "active/inactive recurrent split is missing or incomplete in the corpus".to_string(),
        );
    }
    if corpus.canonical_caseops_runs == 0 {
        warnings.push(
            "no canonical CaseOps trace was found; use timing calibration separately from record-readiness evidence"
                .to_string(),
        );
    }
    let total_step_ms = calibrated_metric("total_step_ms", timed);
    let exact_2135_total_step_ms = calibrated_metric("exact_2135_total_step_ms", exact);
    let active_recurrent_ms = calibrated_metric("active_recurrent_ms", active);
    let inactive_recurrent_ms = calibrated_metric("inactive_recurrent_ms", inactive);
    let recurrent_active_minus_inactive_ms =
        calibrated_metric("recurrent_active_minus_inactive_ms", recurrent_delta);
    let confidence_summary = calibration_confidence_summary(
        exact_2135_total_step_ms.as_ref(),
        active_recurrent_ms.as_ref(),
        corpus.stage_calibration_runs,
    );
    let recommended_trace_to_collect = if corpus.stage_calibration_runs < 2 {
        "collect graph-disabled exact #2135 stage timing with active recurrent windows and nonzero qkv/mlp/attention/output/optimizer fields"
    } else if active_recurrent_ms.is_none() || inactive_recurrent_ms.is_none() {
        "collect exact #2135 record-shaped proxy with timing_recurrent_active_ms_per_step and timing_recurrent_inactive_ms_per_step"
    } else {
        "collect one fresh exact #2135 trace after the next recurrent replay cut to validate the calibration"
    }
    .to_string();
    let status = if total_step_ms.is_some() {
        "pass"
    } else {
        "estimate_only_no_timed_runs"
    }
    .to_string();
    let report = WindTunnelCalibrationReport {
        kind: "pg_local_wind_tunnel_calibration".to_string(),
        spec_path: options.spec.display().to_string(),
        source_kind: if options.corpus.is_some() {
            "corpus_json".to_string()
        } else {
            "trace_dir".to_string()
        },
        corpus_path: options
            .corpus
            .as_ref()
            .map(|path| path.display().to_string()),
        trace_dir: options
            .trace_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        runs_considered: corpus.runs.len(),
        timed_runs: corpus.timed_runs,
        exact_2135_timed_runs: corpus
            .runs
            .iter()
            .filter(|run| corpus_run_is_exact_2135(run) && run.trace_total_ms.is_some())
            .count(),
        stage_calibration_runs: corpus.stage_calibration_runs,
        total_step_ms,
        exact_2135_total_step_ms,
        active_recurrent_ms,
        inactive_recurrent_ms,
        recurrent_active_minus_inactive_ms,
        stage_metrics,
        recommended_trace_to_collect,
        confidence_summary,
        warnings,
        status,
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn run_wind_tunnel_scenario(
    options: WindTunnelScenarioOptions,
) -> PgResult<WindTunnelScenarioReport> {
    let spec = RunSpec::load(&options.spec)?;
    let plan = ExecutionPlan::from_run_spec(&spec)?;
    let (base_step_ms, base_step_source, active_ms, confidence, source_kind) =
        if let Some(path) = options.calibration.as_ref() {
            let calibration =
                serde_json::from_str::<WindTunnelCalibrationReport>(&fs::read_to_string(path)?)
                    .map_err(|err| {
                        PgError::DataFormat(format!("invalid Wind Tunnel calibration JSON: {err}"))
                    })?;
            let metric = calibration
                .exact_2135_total_step_ms
                .as_ref()
                .or(calibration.total_step_ms.as_ref());
            let base = metric
                .map(|metric| metric.best_ms)
                .unwrap_or_else(|| default_stage_estimates(&plan).iter().map(|s| s.ms).sum());
            let source = metric
                .map(|metric| format!("calibration:{}:best_ms", metric.name))
                .unwrap_or_else(|| "shape_model_default_no_calibrated_metric".to_string());
            let active = calibration
                .active_recurrent_ms
                .as_ref()
                .map(|metric| metric.best_ms);
            let confidence = metric
                .map(|metric| metric.confidence.clone())
                .unwrap_or_else(|| "low".to_string());
            (
                base,
                source,
                active,
                confidence,
                "calibration_json".to_string(),
            )
        } else {
            let report = run_wind_tunnel(WindTunnelOptions {
                spec: options.spec.clone(),
                trace: options.trace.clone(),
                output: None,
            })?;
            (
                report.train_step_ms_estimate,
                report.prediction_source,
                report.active_recurrent_trace_ms_per_step,
                report.trace_coverage.signal_quality,
                if options.trace.is_some() {
                    "trace_or_shape_model".to_string()
                } else {
                    "shape_model".to_string()
                },
            )
        };
    let active_cut = options.active_recurrent_replay_cut_ms.max(0.0);
    let bank_cut = options.bank_update_cut_ms.max(0.0);
    let graph_cut = options.graph_overhead_cut_ms.max(0.0);
    let total_cut = active_cut + bank_cut + graph_cut;
    let predicted_step_ms = (base_step_ms - total_cut).max(1.0);
    let predicted_active_recurrent_ms = active_ms.map(|ms| (ms - active_cut).max(1.0));
    let expected_train_steps_in_600s = (600_000.0 / predicted_step_ms.max(1.0)).floor() as usize;
    let expected_train_wall_seconds =
        spec.train.total_iterations as f64 * predicted_step_ms / 1_000.0;
    let remaining_gap_ms = (predicted_step_ms - options.target_step_ms).max(0.0);
    let clears_target = predicted_step_ms <= options.target_step_ms;
    let mut warnings = Vec::new();
    if total_cut <= 0.0 {
        warnings.push("scenario has no positive requested cut".to_string());
    }
    if confidence.contains("low")
        || confidence.contains("total_only")
        || confidence.contains("shape")
    {
        warnings.push("scenario is based on low-confidence or total-only calibration".to_string());
    }
    if active_ms.is_none() && active_cut > 0.0 {
        warnings.push(
            "active recurrent cut was requested without calibrated active recurrent timing"
                .to_string(),
        );
    }
    if total_cut > base_step_ms * 0.30 {
        warnings.push("requested cut exceeds 30% of the calibrated step time".to_string());
    }
    let risk_level = if warnings.len() >= 3 {
        "high"
    } else if warnings.is_empty() && clears_target {
        "medium_low"
    } else {
        "medium"
    }
    .to_string();
    let recommendation = if clears_target {
        "run the cheapest exact #2135 H100 A/B that exercises active recurrence; require median/p90 and bridge/host-copy counters before promotion"
    } else {
        "do not spend a full record run yet; stack another exact recurrent replay or optimizer-launch cut and rerun the scenario"
    }
    .to_string();
    let report = WindTunnelScenarioReport {
        kind: "pg_local_wind_tunnel_scenario".to_string(),
        spec_fingerprint: plan.variant_fingerprint,
        estimate_only: true,
        source_kind,
        calibration_path: options
            .calibration
            .as_ref()
            .map(|path| path.display().to_string()),
        trace_path: options
            .trace
            .as_ref()
            .map(|path| path.display().to_string()),
        base_step_ms,
        base_step_source,
        target_step_ms: options.target_step_ms,
        active_recurrent_replay_cut_ms: active_cut,
        bank_update_cut_ms: bank_cut,
        graph_overhead_cut_ms: graph_cut,
        total_requested_cut_ms: total_cut,
        predicted_step_ms,
        predicted_active_recurrent_ms,
        expected_train_steps_in_600s,
        expected_train_wall_seconds,
        clears_target,
        remaining_gap_ms,
        risk_level,
        recommendation,
        warnings,
        status: "pass".to_string(),
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

pub fn load_lite_spec(path: &Path) -> PgResult<LiteSpec> {
    let body = fs::read_to_string(path)?;
    toml::from_str(&body).map_err(|err| PgError::DataFormat(format!("invalid PG-Lite TOML: {err}")))
}

pub fn run_lite_init(options: LiteInitOptions) -> PgResult<LiteInitReport> {
    let input = fs::read(&options.input)?;
    if input.len() < 2 {
        return Err(PgError::DataFormat(
            "PG-Lite init requires at least two bytes so train and validation splits are non-empty"
                .into(),
        ));
    }
    if options.context == 0 {
        return Err(PgError::InvalidOp(
            "PG-Lite init requires --context > 0".into(),
        ));
    }
    if options.residual_buckets == 0 {
        return Err(PgError::InvalidOp(
            "PG-Lite init requires --residual-buckets > 0".into(),
        ));
    }
    let requested_val = options.val_bytes.unwrap_or_else(|| {
        ((input.len() as f64) * options.val_fraction.clamp(0.01, 0.90)).round() as usize
    });
    let val_len = requested_val.clamp(1, input.len() - 1);
    let train_len = input.len() - val_len;

    fs::create_dir_all(&options.output_dir)?;
    let output_dir = absolute_path(&options.output_dir)?;
    let data_dir = output_dir.join("data");
    let config_dir = output_dir.join("configs");
    fs::create_dir_all(&data_dir)?;
    fs::create_dir_all(&config_dir)?;

    let train_path = data_dir.join("train.bytes");
    let val_path = data_dir.join("val.bytes");
    fs::write(&train_path, &input[..train_len])?;
    fs::write(&val_path, &input[train_len..])?;

    let mut configs = Vec::new();
    let base = LiteSpec {
        artifact_budget_bytes: options.artifact_budget_bytes,
        train_time_seconds: options.train_time_seconds,
        eval_time_seconds: options.eval_time_seconds,
        memory_budget_bytes: options.memory_budget_bytes,
        data: LiteDataSpec {
            train_path: train_path.display().to_string(),
            val_path: val_path.display().to_string(),
            format: "bytes".to_string(),
        },
        model: LiteModelSpec {
            family: LiteModelFamily::NgramResidual,
            vocab: "byte".to_string(),
            context: options.context,
            residual_buckets: options.residual_buckets,
            residual_weight: options.residual_weight,
            artifact_path: None,
        },
        backend: LiteBackendSpec {
            kind: options.backend,
            allow_cpu_fallback: false,
        },
        ..LiteSpec::default()
    };

    let residual = write_lite_init_config(
        &config_dir,
        "byte_golf_ngram_residual",
        LiteSpec {
            track: LiteTrack::ByteGolf,
            ..base.clone()
        },
    )?;
    configs.push(residual);

    let baseline = write_lite_init_config(
        &config_dir,
        "byte_golf_ngram_baseline",
        LiteSpec {
            track: LiteTrack::ByteGolf,
            model: LiteModelSpec {
                family: LiteModelFamily::ByteNgram,
                residual_buckets: 1,
                residual_weight: 0.0,
                ..base.model.clone()
            },
            ..base.clone()
        },
    )?;
    configs.push(baseline);

    let stream = write_lite_init_config(
        &config_dir,
        "stream_golf_score_first",
        LiteSpec {
            track: LiteTrack::StreamGolf,
            ..base.clone()
        },
    )?;
    configs.push(stream);

    let artifact = write_lite_init_config(
        &config_dir,
        "artifact_golf_empty",
        LiteSpec {
            track: LiteTrack::ArtifactGolf,
            train_time_seconds: 0.0,
            model: LiteModelSpec {
                family: LiteModelFamily::ArtifactOnly,
                residual_weight: 0.0,
                artifact_path: None,
                ..base.model.clone()
            },
            backend: LiteBackendSpec {
                kind: LiteBackendKind::CpuReference,
                allow_cpu_fallback: false,
            },
            ..base.clone()
        },
    )?;
    configs.push(artifact);

    if options.backend != LiteBackendKind::MetalApple {
        let metal_probe = write_lite_init_config(
            &config_dir,
            "byte_golf_metal_probe",
            LiteSpec {
                track: LiteTrack::ByteGolf,
                backend: LiteBackendSpec {
                    kind: LiteBackendKind::MetalApple,
                    allow_cpu_fallback: true,
                },
                ..base.clone()
            },
        )?;
        configs.push(metal_probe);
    }

    let readme_path = output_dir.join("README.md");
    let report_path = output_dir.join("init_report.json");
    let report = LiteInitReport {
        kind: "pg_lite_init",
        input_path: absolute_path(&options.input)?.display().to_string(),
        output_dir: output_dir.display().to_string(),
        input_bytes: input.len(),
        input_crc32: crc32_label(&input),
        train_path: train_path.display().to_string(),
        val_path: val_path.display().to_string(),
        train_bytes: train_len,
        train_crc32: crc32_label(&input[..train_len]),
        val_bytes: val_len,
        val_crc32: crc32_label(&input[train_len..]),
        val_fraction_effective: val_len as f64 / input.len() as f64,
        configs,
        readme_path: readme_path.display().to_string(),
        report_path: report_path.display().to_string(),
        status: "pass".to_string(),
    };
    fs::write(&readme_path, lite_init_readme(&report))?;
    write_json_pretty(&report_path, &report)?;
    Ok(report)
}

pub fn run_lite_verify_artifact(
    options: LiteVerifyArtifactOptions,
) -> PgResult<LiteVerifyArtifactReport> {
    let artifact_bytes = fs::read(&options.artifact)?;
    let artifact_size = artifact_bytes.len();
    let artifact_crc32 = crc32_label(&artifact_bytes);
    let artifact = load_lite_stored_artifact(&options.artifact)?;
    let model = LiteNgramResidual::from_stored_artifact(&artifact)?;
    let mut notes = Vec::new();

    let (expected_config_fingerprint, artifact_budget_bytes, config_val_path) =
        if let Some(config_path) = options.config.as_ref() {
            let spec = load_lite_spec(config_path)?;
            validate_lite_spec(&spec)?;
            (
                Some(spec.config_fingerprint()?),
                Some(spec.artifact_budget_bytes),
                Some(PathBuf::from(spec.data.val_path)),
            )
        } else {
            (None, None, None)
        };
    let source_config_matches_expected = expected_config_fingerprint
        .as_ref()
        .map(|expected| expected == &artifact.source_config_fingerprint);
    if source_config_matches_expected == Some(false) {
        notes.push(
            "artifact source_config_fingerprint does not match the supplied config".to_string(),
        );
    }
    let artifact_budget_ok = artifact_budget_bytes.map(|budget| artifact_size <= budget);
    let val_path = options.val.clone().or(config_val_path);
    let (val_bytes, validation_bpb) = if let Some(path) = val_path.as_ref() {
        let val = fs::read(path)?;
        if val.is_empty() {
            notes.push("validation bytes are empty; BPB was not computed".to_string());
            (Some(0), None)
        } else {
            let eval = model.loss_on_bytes(&val, false);
            let avg_loss = eval.loss / eval.tokens.max(1) as f64;
            (
                Some(val.len()),
                Some(compute_bpb(
                    avg_loss,
                    eval.tokens as f64,
                    eval.tokens as f64,
                )),
            )
        }
    } else {
        notes.push(
            "no validation path supplied; artifact decode was checked without BPB".to_string(),
        );
        (None, None)
    };
    let status = if artifact.kind == "pg_lite_model_artifact"
        && artifact.version == 1
        && artifact_budget_ok.unwrap_or(true)
        && source_config_matches_expected.unwrap_or(true)
        && validation_bpb.map(|bpb| bpb.is_finite()).unwrap_or(true)
    {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = LiteVerifyArtifactReport {
        kind: "pg_lite_verify_artifact",
        artifact_path: options.artifact.display().to_string(),
        artifact_bytes: artifact_size,
        artifact_crc32,
        artifact_kind: artifact.kind,
        decoded_ok: true,
        model_reconstruct_ok: true,
        source_config_fingerprint: artifact.source_config_fingerprint,
        model_fingerprint: artifact.model_fingerprint,
        expected_config_fingerprint,
        source_config_matches_expected,
        model_family: artifact.model_family,
        context: artifact.context,
        residual_weight: artifact.residual_weight,
        bigram_rows: artifact.bigram_rows,
        residual_buckets: artifact.residual_buckets,
        bigram_entries: artifact.bigram_entries.len(),
        residual_entries: artifact.residual_entries.len(),
        artifact_budget_bytes,
        artifact_budget_ok,
        val_path: val_path.as_ref().map(|path| path.display().to_string()),
        val_bytes,
        validation_bpb,
        validation_bpb_scope: "local_proxy_bpb_not_leaderboard",
        notes,
        status,
    };
    write_report_if_requested(&report, options.output.as_deref())?;
    Ok(report)
}

fn write_lite_init_config(
    config_dir: &Path,
    name: &str,
    spec: LiteSpec,
) -> PgResult<LiteInitConfigReport> {
    validate_lite_spec(&spec)?;
    let path = config_dir.join(format!("{name}.toml"));
    let body = toml::to_string_pretty(&spec).map_err(|err| {
        PgError::DataFormat(format!("PG-Lite init TOML encode failed for {name}: {err}"))
    })?;
    fs::write(&path, body)?;
    Ok(LiteInitConfigReport {
        name: name.to_string(),
        path: path.display().to_string(),
        track: spec.track,
        model_family: spec.model.family,
        backend: spec.backend.kind,
        allow_cpu_fallback: spec.backend.allow_cpu_fallback,
        artifact_budget_bytes: spec.artifact_budget_bytes,
    })
}

fn lite_init_readme(report: &LiteInitReport) -> String {
    let mut body = String::new();
    body.push_str("# PG-Lite Local Benchmark\n\n");
    body.push_str(
        "This directory was generated by `pg-local lite init`. It is a local proxy benchmark and does not claim Parameter Golf leaderboard validity.\n\n",
    );
    body.push_str("## Data\n\n");
    body.push_str(&format!(
        "- Input bytes: `{}` (`{}`)\n- Train bytes: `{}` (`{}`)\n- Validation bytes: `{}` (`{}`)\n- Effective validation fraction: `{:.4}`\n\n",
        report.input_bytes,
        report.input_crc32,
        report.train_bytes,
        report.train_crc32,
        report.val_bytes,
        report.val_crc32,
        report.val_fraction_effective
    ));
    body.push_str("## Configs\n\n");
    body.push_str("| Name | Track | Family | Backend | Budget bytes | Path |\n");
    body.push_str("|---|---|---|---|---:|---|\n");
    for config in &report.configs {
        body.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}`{} | {} | `{}` |\n",
            config.name,
            lite_track_label(config.track),
            lite_model_family_label(config.model_family),
            lite_backend_label(config.backend),
            if config.allow_cpu_fallback {
                " fallback"
            } else {
                ""
            },
            config.artifact_budget_bytes,
            config.path
        ));
    }
    body.push_str("\n## Commands\n\n");
    if let Some(first) = report
        .configs
        .iter()
        .find(|config| config.name == "byte_golf_ngram_residual")
    {
        body.push_str(&format!(
            "```bash\npg-local lite run --config '{}' --output run_byte_golf\npg-local lite benchmark --config '{}' --output benchmark_byte_golf --repeats 3\npg-local lite suite --config-dir '{}/configs' --output suite\npg-local lite verify-artifact --artifact run_byte_golf/model.pglite.bin --config '{}' --output verify_artifact.json\n```\n",
            first.path, first.path, report.output_dir, first.path
        ));
    }
    body.push_str("\nPG-Lite BPB is local proxy evidence only.\n");
    body
}

fn crc32_label(bytes: &[u8]) -> String {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    format!("crc32:{:08x}", hasher.finalize())
}

fn absolute_path(path: &Path) -> PgResult<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

pub fn write_report_if_requested<T: Serialize>(report: &T, output: Option<&Path>) -> PgResult<()> {
    let body = serde_json::to_string_pretty(report)
        .map_err(|err| PgError::DataFormat(format!("JSON encode failed: {err}")))?;
    if let Some(path) = output {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, body)?;
    } else {
        println!("{body}");
    }
    Ok(())
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> PgResult<()> {
    let body = serde_json::to_string_pretty(value)
        .map_err(|err| PgError::DataFormat(format!("JSON encode failed: {err}")))?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)?;
    Ok(())
}

fn glob_paths(dir: &Path, pattern: &str) -> PgResult<Vec<PathBuf>> {
    let glob_pattern = dir.join(pattern).to_string_lossy().into_owned();
    let mut paths = Vec::new();
    for entry in glob::glob(&glob_pattern)
        .map_err(|err| PgError::DataFormat(format!("invalid glob pattern: {err}")))?
    {
        paths.push(entry.map_err(|err| PgError::DataFormat(format!("glob entry failed: {err}")))?);
    }
    Ok(paths)
}

fn wind_tunnel_trace_paths(dir: &Path) -> PgResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for pattern in [
        "*.json",
        "*.jsonl",
        "*.log",
        "*.txt",
        "*.out",
        "**/*.json",
        "**/*.jsonl",
        "**/*.log",
        "**/*.txt",
        "**/*.out",
    ] {
        paths.extend(glob_paths(dir, pattern)?);
    }
    paths.retain(|path| {
        let label = path.to_string_lossy();
        !label.ends_with(".wind_tunnel.json")
            && !label.ends_with("baseline_shape_model.json")
            && !label.ends_with("summary.json")
            && !label.ends_with("summary.md")
            && !label.ends_with("trace_index.json")
            && !label.ends_with("trace_corpus.json")
            && !label.ends_with("calibration_model.json")
            && !label.ends_with("scenario_exact_recurrent_replay.json")
            && !label.ends_with("proof_bundle.json")
            && !label.ends_with("evidence_manifest.json")
            && !label.ends_with("proposal_features.json")
    });
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn trace_calibration_role(report: &WindTunnelReport) -> String {
    let coverage = &report.trace_coverage;
    if !coverage.trace_supplied {
        return "shape_model_only".to_string();
    }
    if report.trace_total_ms.is_none() && coverage.nonzero_mapped_stage_count > 0 {
        return "stage_only_diagnostic".to_string();
    }
    if report.trace_total_ms.is_none() {
        return "shape_model_only".to_string();
    }
    if coverage.useful_stage_coverage_ratio >= 0.5 && coverage.traced_stage_ms_ratio >= 0.35 {
        "stage_calibration".to_string()
    } else if coverage.nonzero_mapped_stage_count > 0 {
        "partial_stage_calibration".to_string()
    } else {
        "total_calibration".to_string()
    }
}

fn pg_lite_metal_kernel_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/pg_lite_ngram.metal")
}

#[cfg(test)]
fn pg_lite_metal_expected_symbols() -> Vec<String> {
    pg_lite_metal_kernel_contracts()
        .into_iter()
        .map(|contract| contract.symbol)
        .collect()
}

fn pg_lite_metal_kernel_contracts() -> Vec<BackendKernelContractReport> {
    [
        (
            "pg_lite_init_bigram_counts_kernel",
            "initialize add-one-smoothed bigram count table",
            "parallel dense table fill; removes CPU-side count initialization",
        ),
        (
            "pg_lite_init_residual_counts_u16_kernel",
            "initialize compact u16 residual count table",
            "parallel dense table fill for exported/eval compact path",
        ),
        (
            "pg_lite_init_residual_counts_u32_kernel",
            "initialize u32 residual count table",
            "training path uses u32 atomics before saturating downcast",
        ),
        (
            "pg_lite_bigram_row_sums_kernel",
            "precompute 257 bigram row sums",
            "avoids summing 256 counters per token during loss",
        ),
        (
            "pg_lite_residual_row_sums_u16_kernel",
            "precompute compact residual row sums",
            "avoids per-token residual row scans in u16 eval",
        ),
        (
            "pg_lite_residual_row_sums_u32_kernel",
            "precompute u32 residual row sums",
            "avoids per-token residual row scans in training/eval",
        ),
        (
            "pg_lite_ngram_train_kernel",
            "atomic train pass over byte stream",
            "parallel count accumulation for bigram and hashed residual tables",
        ),
        (
            "pg_lite_downcast_residual_u32_to_u16_kernel",
            "saturating residual table compaction",
            "converts atomic u32 training counts into compact artifact/eval counts",
        ),
        (
            "pg_lite_ngram_loss_presummed_u16_kernel",
            "per-token loss with u16 residual counts and precomputed sums",
            "fast eval path that writes token losses for optional downstream reductions",
        ),
        (
            "pg_lite_ngram_loss_presummed_u32_kernel",
            "per-token loss with u32 residual counts and precomputed sums",
            "fast train/eval path before residual downcast",
        ),
        (
            "pg_lite_ngram_loss_kernel",
            "legacy u16 residual loss without precomputed sums",
            "reference-compatible fallback for launch bring-up",
        ),
        (
            "pg_lite_ngram_loss_u32_residual_kernel",
            "legacy u32 residual loss without precomputed sums",
            "reference-compatible fallback for launch bring-up",
        ),
        (
            "pg_lite_ngram_loss_reduce_u16_presummed_kernel",
            "fused u16 residual loss and block reduction",
            "avoids materializing token_losses for the primary eval path",
        ),
        (
            "pg_lite_ngram_loss_reduce_u32_presummed_kernel",
            "fused u32 residual loss and block reduction",
            "avoids materializing token_losses before residual compaction",
        ),
        (
            "pg_lite_reduce_loss_kernel",
            "block reduce token loss buffer",
            "shared-memory reduction for fallback loss buffers",
        ),
        (
            "pg_lite_score_first_eval_u32_kernel",
            "score-first sequential eval/update correctness kernel",
            "single-lane legality path because online updates are order-dependent",
        ),
    ]
    .into_iter()
    .map(
        |(symbol, role, optimization_note)| BackendKernelContractReport {
            symbol: symbol.to_string(),
            role: role.to_string(),
            optimization_note: optimization_note.to_string(),
            executable_required_for_acceleration: true,
        },
    )
    .collect()
}

fn metal_source_contains_kernel(source: &str, symbol: &str) -> bool {
    source.contains(&format!("kernel void {symbol}"))
}

#[derive(Debug, Clone)]
struct MetalCompilerProbe {
    path: Option<String>,
    error: Option<String>,
}

fn find_metal_compiler() -> MetalCompilerProbe {
    let output = match std::process::Command::new("xcrun")
        .args(["--find", "metal"])
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return MetalCompilerProbe {
                path: None,
                error: Some(format!("failed to invoke xcrun: {err}")),
            };
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return MetalCompilerProbe {
            path: None,
            error: Some(if stderr.is_empty() {
                format!("xcrun exited with status {}", output.status)
            } else {
                stderr
            }),
        };
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        MetalCompilerProbe {
            path: None,
            error: Some("xcrun returned an empty metal compiler path".to_string()),
        }
    } else {
        MetalCompilerProbe {
            path: Some(path),
            error: None,
        }
    }
}

fn compile_metal_kernel_smoke(source: &Path) -> Result<(), String> {
    let stem = format!(
        "pg_lite_ngram_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    );
    let air = std::env::temp_dir().join(format!("{stem}.air"));
    let metallib = std::env::temp_dir().join(format!("{stem}.metallib"));

    let metal = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(source)
        .args(["-o"])
        .arg(&air)
        .output()
        .map_err(|err| format!("failed to invoke xcrun metal: {err}"))?;
    if !metal.status.success() {
        let stderr = String::from_utf8_lossy(&metal.stderr).trim().to_string();
        let _ = fs::remove_file(&air);
        let _ = fs::remove_file(&metallib);
        return Err(if stderr.is_empty() {
            format!("xcrun metal exited with status {}", metal.status)
        } else {
            stderr
        });
    }

    let link = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .args(["-o"])
        .arg(&metallib)
        .output()
        .map_err(|err| format!("failed to invoke xcrun metallib: {err}"))?;
    let _ = fs::remove_file(&air);
    let _ = fs::remove_file(&metallib);
    if !link.status.success() {
        let stderr = String::from_utf8_lossy(&link.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("xcrun metallib exited with status {}", link.status)
        } else {
            stderr
        });
    }
    Ok(())
}

fn sanitize_path_component(raw: &str) -> String {
    let clean = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if clean.is_empty() {
        "run".to_string()
    } else {
        clean
    }
}

fn monotonic_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn scratch_report_path(stem: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{stem}_{}_{}.json",
        std::process::id(),
        monotonic_nanos()
    ))
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        None
    } else {
        Some(finite.iter().copied().sum::<f64>() / finite.len() as f64)
    }
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[mid - 1] + values[mid]) * 0.5)
    } else {
        Some(values[mid])
    }
}

fn lite_track_label(track: LiteTrack) -> &'static str {
    match track {
        LiteTrack::ByteGolf => "byte_golf",
        LiteTrack::ArtifactGolf => "artifact_golf",
        LiteTrack::StreamGolf => "stream_golf",
    }
}

fn lite_model_family_label(family: LiteModelFamily) -> &'static str {
    match family {
        LiteModelFamily::ByteNgram => "byte_ngram",
        LiteModelFamily::NgramResidual => "ngram_residual",
        LiteModelFamily::ArtifactOnly => "artifact_only",
    }
}

fn lite_backend_label(backend: LiteBackendKind) -> &'static str {
    match backend {
        LiteBackendKind::CpuReference => "cpu_reference",
        LiteBackendKind::MetalApple => "metal_apple",
        LiteBackendKind::MlxPrototype => "mlx_prototype",
    }
}

#[derive(Debug)]
struct LiteMetalTrainEval {
    model: LiteNgramResidual,
    eval: LiteEvalStats,
}

#[cfg(feature = "metal_apple")]
fn run_metal_lite_train_eval(
    train: &[u8],
    val: &[u8],
    spec: &LiteSpec,
) -> PgResult<LiteMetalTrainEval> {
    let out = metal_lite::train_and_eval_fixed(train, val, spec)?;
    Ok(LiteMetalTrainEval {
        model: out.model,
        eval: out.eval,
    })
}

#[cfg(not(feature = "metal_apple"))]
fn run_metal_lite_train_eval(
    _train: &[u8],
    _val: &[u8],
    _spec: &LiteSpec,
) -> PgResult<LiteMetalTrainEval> {
    Err(PgError::InvalidOp(
        "PG-Lite Metal train/eval requested without `metal_apple` feature".into(),
    ))
}

#[cfg(feature = "metal_apple")]
fn run_metal_lite_fixed_eval(model: &LiteNgramResidual, val: &[u8]) -> PgResult<LiteEvalStats> {
    metal_lite::eval_fixed_model(model, val)
}

#[cfg(not(feature = "metal_apple"))]
fn run_metal_lite_fixed_eval(_model: &LiteNgramResidual, _val: &[u8]) -> PgResult<LiteEvalStats> {
    Err(PgError::InvalidOp(
        "PG-Lite Metal fixed eval requested without `metal_apple` feature".into(),
    ))
}

#[cfg(feature = "metal_apple")]
fn run_metal_lite_score_first_eval(
    model: &LiteNgramResidual,
    val: &[u8],
) -> PgResult<LiteEvalStats> {
    metal_lite::score_first_eval_model(model, val)
}

#[cfg(not(feature = "metal_apple"))]
fn run_metal_lite_score_first_eval(
    _model: &LiteNgramResidual,
    _val: &[u8],
) -> PgResult<LiteEvalStats> {
    Err(PgError::InvalidOp(
        "PG-Lite Metal score-first eval requested without `metal_apple` feature".into(),
    ))
}

fn lite_suite_markdown(report: &LiteSuiteReport) -> String {
    let mut body = String::new();
    body.push_str("# PG-Lite Suite\n\n");
    body.push_str(&format!(
        "- Configs: `{}`\n- Status: `{}`\n",
        report.configs_found, report.status
    ));
    if let (Some(best_config), Some(best_bpb)) = (&report.best_config, report.best_bpb) {
        body.push_str(&format!(
            "- Best local proxy BPB: `{best_bpb:.6}` from `{best_config}`\n"
        ));
    }
    body.push('\n');
    body.push_str("| Config | Track | Family | Backend | Status | BPB | Artifact bytes | Artifact budget | Score-first updates |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for run in &report.runs {
        body.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` -> `{}` | `{}` | {:.6} | {} | `{}` | {}/{} |\n",
            run.config_path,
            lite_track_label(run.track),
            lite_model_family_label(run.model_family),
            lite_backend_label(run.requested_backend),
            lite_backend_label(run.execution_backend),
            run.status,
            run.validation_bpb,
            run.artifact_actual_bytes,
            if run.artifact_budget_ok {
                "pass"
            } else {
                "fail"
            },
            run.score_first_update_count,
            run.score_first_tokens_scored
        ));
    }
    body.push_str("\nPG-Lite BPB is local proxy evidence only; it is not Parameter Golf leaderboard evidence.\n");
    body
}

fn lite_benchmark_markdown(report: &LiteBenchmarkReport) -> String {
    let mut body = String::new();
    body.push_str("# PG-Lite Backend Benchmark\n\n");
    body.push_str(&format!(
        "- Config: `{}`\n- Repeats: `{}`\n- Status: `{}`\n",
        report.config_path, report.repeats, report.status
    ));
    if let Some(speedup) = report.cpu_vs_metal_speedup_total {
        body.push_str(&format!(
            "- CPU-vs-Metal total wall speedup: `{speedup:.3}x`\n"
        ));
    }
    if let Some(speedup) = report.cpu_vs_metal_speedup_train {
        body.push_str(&format!(
            "- CPU-vs-Metal train wall speedup: `{speedup:.3}x`\n"
        ));
    }
    if let Some(delta) = report.cpu_vs_metal_bpb_delta {
        body.push_str(&format!(
            "- Metal minus CPU local BPB delta: `{delta:.9}`\n"
        ));
    }
    if let Some(backend) = report.best_total_wall_backend {
        body.push_str(&format!(
            "- Fastest local backend: `{}`\n",
            lite_backend_label(backend)
        ));
    }
    if let Some(backend) = report.best_bpb_backend {
        body.push_str(&format!(
            "- Best local proxy BPB backend: `{}`\n",
            lite_backend_label(backend)
        ));
    }
    body.push('\n');
    body.push_str("| Requested backend | Status | Runs | Accelerated | Execution backends | Median total s | Median train s | Median eval s | Median BPB | Artifact bytes | Error |\n");
    body.push_str("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|\n");
    for backend in &report.backends {
        let execution_backends = backend
            .execution_backends
            .iter()
            .map(|backend| lite_backend_label(*backend))
            .collect::<Vec<_>>()
            .join(", ");
        body.push_str(&format!(
            "| `{}` | `{}` | {} | {}/{} | `{}` | {} | {} | {} | {} | {} | {} |\n",
            lite_backend_label(backend.requested_backend),
            backend.status,
            backend.run_count,
            backend.accelerated_run_count,
            backend.run_count,
            execution_backends,
            optional_ms_like_seconds(backend.median_total_wall_seconds),
            optional_ms_like_seconds(backend.median_train_wall_seconds),
            optional_ms_like_seconds(backend.median_eval_wall_seconds),
            backend
                .median_validation_bpb
                .map(|value| format!("{value:.9}"))
                .unwrap_or_else(|| "n/a".to_string()),
            backend
                .artifact_actual_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            backend.error.as_deref().unwrap_or("")
        ));
    }
    body.push_str(
        "\nPG-Lite benchmark results are local proxy evidence. They do not establish H100 throughput, final BPB, or leaderboard validity.\n",
    );
    body
}

fn optional_ms_like_seconds(value: Option<f64>) -> String {
    value
        .map(|seconds| format!("{seconds:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn wind_tunnel_suite_markdown(report: &WindTunnelSuiteReport) -> String {
    let mut body = String::new();
    body.push_str("# Wind Tunnel Trace Suite\n\n");
    body.push_str(&format!(
        "- Traces: `{}`\n- Traces with measured step timing: `{}`\n- Traces with stage attribution: `{}`\n- Total-only traces: `{}`\n- Shape-only traces: `{}`\n- Status: `{}`\n- Baseline shape-model estimate: `{:.3} ms/step`\n",
        report.traces_found,
        report.traces_with_step_timing,
        report.traces_with_stage_attribution,
        report.traces_total_only,
        report.traces_shape_only,
        report.status,
        report.baseline_train_step_ms_estimate
    ));
    if let Some(error) = report.mean_abs_baseline_step_error_ms {
        body.push_str(&format!(
            "- Mean absolute baseline error: `{error:.3} ms/step`\n"
        ));
    }
    if let Some(error) = report.mean_abs_baseline_pct_error {
        body.push_str(&format!("- Mean absolute baseline error: `{error:.2}%`\n"));
    }
    body.push('\n');
    body.push_str(
        "| Trace | Trace total | Estimate | Baseline error | Coverage | Useful stages | Role | Top bottleneck | Risks |\n",
    );
    body.push_str("|---|---:|---:|---:|---:|---:|---|---|---|\n");
    for run in &report.reports {
        let trace_total = run
            .trace_total_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        let error = run
            .baseline_abs_step_error_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        body.push_str(&format!(
            "| `{}` | {} | {:.3} | {} | {:.2}% | {:.2}% | `{}` | `{}` | {} |\n",
            run.trace_path,
            trace_total,
            run.train_step_ms_estimate,
            error,
            run.trace_coverage_ratio * 100.0,
            run.useful_stage_coverage_ratio * 100.0,
            run.calibration_role,
            run.top_bottleneck,
            run.risk_flags.join(", ")
        ));
    }
    body.push_str("\nWind Tunnel predictions are planning estimates, not H100 validation.\n");
    body
}

fn wind_tunnel_holdout_report(corpus: &WindTunnelCorpusReport) -> WindTunnelHoldoutReport {
    let mut timed_runs = corpus
        .runs
        .iter()
        .filter(|run| run.trace_total_ms.is_some_and(f64::is_finite))
        .collect::<Vec<_>>();
    timed_runs.sort_by(|a, b| a.run_key.cmp(&b.run_key));

    let mut train = Vec::new();
    let mut holdout = Vec::new();
    for (idx, run) in timed_runs.into_iter().enumerate() {
        if idx % 2 == 0 {
            train.push(run);
        } else {
            holdout.push(run);
        }
    }

    let predictor = WindTunnelHoldoutPredictor::fit(&train);
    let samples = holdout
        .iter()
        .filter_map(|run| predictor.predict(run))
        .collect::<Vec<_>>();
    let predicted = samples
        .iter()
        .map(|sample| sample.predicted_ms)
        .collect::<Vec<_>>();
    let observed = samples
        .iter()
        .map(|sample| sample.observed_ms)
        .collect::<Vec<_>>();
    let errors = samples
        .iter()
        .map(|sample| sample.abs_error_ms)
        .collect::<Vec<_>>();
    let pct_errors = samples
        .iter()
        .filter_map(|sample| sample.pct_error)
        .collect::<Vec<_>>();
    let status = if predictor.global_median_ms.is_some() && !holdout.is_empty() {
        "pass"
    } else if predictor.global_median_ms.is_some() {
        "insufficient_holdout"
    } else {
        "no_timed_runs"
    }
    .to_string();

    WindTunnelHoldoutReport {
        kind: "pg_local_wind_tunnel_holdout",
        estimate_only: true,
        predictor_kind: "train_only_stage_residual_plus_recurrent_split",
        train_sample_count: train.len(),
        holdout_sample_count: holdout.len(),
        baseline_ms: predictor.global_median_ms,
        mean_predicted_ms: mean_f64(&predicted),
        mean_abs_error_ms: mean_f64(&errors),
        mean_abs_pct_error: mean_f64(&pct_errors),
        spearman_rank_correlation: spearman_rank_correlation(&predicted, &observed),
        samples,
        status,
    }
}

#[derive(Debug, Clone, Default)]
struct WindTunnelHoldoutPredictor {
    global_median_ms: Option<f64>,
    global_stage_residual_ms: Option<f64>,
    global_active_recurrent_ms: Option<f64>,
    global_inactive_recurrent_ms: Option<f64>,
    class_median_ms: BTreeMap<String, f64>,
    class_stage_residual_ms: BTreeMap<String, f64>,
    class_active_recurrent_ms: BTreeMap<String, f64>,
    class_inactive_recurrent_ms: BTreeMap<String, f64>,
}

impl WindTunnelHoldoutPredictor {
    fn fit(runs: &[&WindTunnelCorpusRun]) -> Self {
        let mut observed = Vec::new();
        let mut stage_residuals = Vec::new();
        let mut active = Vec::new();
        let mut inactive = Vec::new();
        let mut observed_by_class = BTreeMap::<String, Vec<f64>>::new();
        let mut stage_residuals_by_class = BTreeMap::<String, Vec<f64>>::new();
        let mut active_by_class = BTreeMap::<String, Vec<f64>>::new();
        let mut inactive_by_class = BTreeMap::<String, Vec<f64>>::new();

        for run in runs {
            let Some(ms) = run.trace_total_ms.filter(|value| value.is_finite()) else {
                continue;
            };
            let class = wind_tunnel_holdout_class(run);
            observed.push(ms);
            observed_by_class.entry(class.clone()).or_default().push(ms);
            if let Some(stage_sum) = wind_tunnel_trace_stage_sum(run) {
                let residual = ms - stage_sum;
                if residual.is_finite() {
                    stage_residuals.push(residual);
                    stage_residuals_by_class
                        .entry(class.clone())
                        .or_default()
                        .push(residual);
                }
            }
            if let Some(ms) = run
                .active_recurrent_ms_per_step
                .filter(|value| value.is_finite())
            {
                active.push(ms);
                active_by_class.entry(class.clone()).or_default().push(ms);
            }
            if let Some(ms) = run
                .inactive_recurrent_ms_per_step
                .filter(|value| value.is_finite())
            {
                inactive.push(ms);
                inactive_by_class.entry(class).or_default().push(ms);
            }
        }

        Self {
            global_median_ms: median_f64(observed),
            global_stage_residual_ms: median_f64(stage_residuals),
            global_active_recurrent_ms: median_f64(active),
            global_inactive_recurrent_ms: median_f64(inactive),
            class_median_ms: median_by_key(observed_by_class),
            class_stage_residual_ms: median_by_key(stage_residuals_by_class),
            class_active_recurrent_ms: median_by_key(active_by_class),
            class_inactive_recurrent_ms: median_by_key(inactive_by_class),
        }
    }

    fn predict(&self, run: &WindTunnelCorpusRun) -> Option<WindTunnelHoldoutSampleReport> {
        let observed_ms = run.trace_total_ms.filter(|value| value.is_finite())?;
        let class_label = wind_tunnel_holdout_class(run);
        let mut sources = Vec::new();
        let mut predicted = if let Some(stage_sum) = wind_tunnel_trace_stage_sum(run) {
            if let Some(residual) = self.class_stage_residual_ms.get(&class_label) {
                sources.push("class_stage_residual");
                stage_sum + residual
            } else if let Some(residual) = self.global_stage_residual_ms {
                sources.push("global_stage_residual");
                stage_sum + residual
            } else {
                sources.push("global_median_total_no_stage_residual");
                self.global_median_ms?
            }
        } else if let Some(class_median) = self.class_median_ms.get(&class_label) {
            sources.push("class_median_total");
            *class_median
        } else {
            sources.push("global_median_total");
            self.global_median_ms?
        };

        if let Some(active_ms) = run
            .active_recurrent_ms_per_step
            .filter(|value| value.is_finite())
        {
            if let Some(reference) = self
                .class_active_recurrent_ms
                .get(&class_label)
                .copied()
                .or(self.global_active_recurrent_ms)
            {
                predicted += active_ms - reference;
                sources.push("active_recurrent_offset");
            }
        } else if let Some(inactive_ms) = run
            .inactive_recurrent_ms_per_step
            .filter(|value| value.is_finite())
            && let Some(reference) = self
                .class_inactive_recurrent_ms
                .get(&class_label)
                .copied()
                .or(self.global_inactive_recurrent_ms)
        {
            predicted += inactive_ms - reference;
            sources.push("inactive_recurrent_offset");
        }

        let predicted_ms = predicted.max(1.0);
        let abs_error_ms = (predicted_ms - observed_ms).abs();
        let pct_error = if observed_ms.abs() > f64::EPSILON {
            Some(abs_error_ms / observed_ms.abs() * 100.0)
        } else {
            None
        };
        Some(WindTunnelHoldoutSampleReport {
            run_key: run.run_key.clone(),
            selected_trace_path: run.selected_trace_path.clone(),
            class_label,
            prediction_source: sources.join("+"),
            observed_ms,
            predicted_ms,
            abs_error_ms,
            pct_error,
        })
    }
}

fn wind_tunnel_holdout_class(run: &WindTunnelCorpusRun) -> String {
    if corpus_run_is_allst(run) {
        "allst_diagnostic".to_string()
    } else if corpus_run_is_exact_2135(run) {
        "exact_2135".to_string()
    } else if run
        .active_recurrent_ms_per_step
        .zip(run.inactive_recurrent_ms_per_step)
        .is_some_and(|(active, inactive)| active > inactive)
    {
        "active_recurrent".to_string()
    } else {
        "other".to_string()
    }
}

fn wind_tunnel_trace_stage_sum(run: &WindTunnelCorpusRun) -> Option<f64> {
    let body = fs::read_to_string(&run.selected_trace_path).ok()?;
    let trace = parse_trace_body(&body)?;
    let mut total = 0.0;
    let mut count = 0usize;
    for mapping in TRACE_STAGE_MAPPINGS {
        if let Some(ms) = trace_number_for_aliases(&trace, mapping.aliases)
            && ms.is_finite()
            && ms.abs() > 1e-9
        {
            total += ms;
            count += 1;
        }
    }
    (count > 0).then_some(total)
}

fn median_by_key(values: BTreeMap<String, Vec<f64>>) -> BTreeMap<String, f64> {
    values
        .into_iter()
        .filter_map(|(key, values)| median_f64(values).map(|median| (key, median)))
        .collect()
}

fn spearman_rank_correlation(predicted: &[f64], observed: &[f64]) -> Option<f64> {
    if predicted.len() != observed.len() || predicted.len() < 2 {
        return None;
    }
    let predicted_ranks = rank_finite_values(predicted)?;
    let observed_ranks = rank_finite_values(observed)?;
    pearson_correlation(&predicted_ranks, &observed_ranks)
}

fn rank_finite_values(values: &[f64]) -> Option<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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

fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = mean_f64(x)?;
    let mean_y = mean_f64(y)?;
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
    let denominator = (x_sq * y_sq).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((numerator / denominator).clamp(-1.0, 1.0))
    }
}

fn wind_tunnel_reviewer_markdown(
    report: &WindTunnelReviewerReport,
    index: &TraceIndexReport,
    corpus: &WindTunnelCorpusReport,
    calibration: &WindTunnelCalibrationReport,
    scenario: &WindTunnelScenarioReport,
    suite: &WindTunnelSuiteReport,
) -> String {
    let mut body = String::new();
    body.push_str("# PG Wind Tunnel Reviewer Report\n\n");
    body.push_str(
        "This report is local planning evidence. It indexes historical traces, deduplicates runs, calibrates a simple stage model, and estimates the next recurrent-replay cut. It is not an H100 validation run.\n\n",
    );
    body.push_str("## Summary\n\n");
    body.push_str(&format!(
        "- Spec: `{}`\n- Trace dir: `{}`\n- Status: `{}`\n- Traces found: `{}`\n- Parseable traces: `{}`\n- Deduped runs: `{}`\n- Timed runs: `{}`\n- Exact #2135 timed runs: `{}`\n- Stage calibration runs: `{}`\n- Canonical CaseOps runs: `{}`\n",
        report.spec_path,
        report.trace_dir,
        report.status,
        report.traces_found,
        report.parseable_traces,
        report.deduped_runs,
        report.timed_runs,
        report.exact_2135_timed_runs,
        report.stage_calibration_runs,
        report.canonical_caseops_runs
    ));
    if let Some(ms) = report.best_exact_2135_ms {
        body.push_str(&format!(
            "- Best exact #2135 timed trace: `{ms:.3} ms/step`\n"
        ));
    }
    if let Some(ms) = report.best_active_recurrent_ms {
        body.push_str(&format!(
            "- Best active recurrent trace: `{ms:.3} ms/active step`\n"
        ));
    }
    body.push_str(&format!(
        "- Calibration confidence: `{}`\n- Scenario prediction after requested cut: `{:.3} ms/step`\n- Scenario clears target: `{}`\n- Recommended next experiment: {}\n",
        report.calibration_confidence_summary,
        report.scenario_predicted_step_ms,
        report.scenario_clears_target,
        report.recommended_next_experiment
    ));

    body.push_str("\n## Holdout Check\n\n");
    body.push_str(&format!(
        "- Status: `{}`\n- Train samples: `{}`\n- Holdout samples: `{}`\n",
        report.holdout_validation.status,
        report.holdout_validation.train_sample_count,
        report.holdout_validation.holdout_sample_count
    ));
    if let Some(value) = report.holdout_validation.baseline_ms {
        body.push_str(&format!("- Median train baseline: `{value:.3} ms/step`\n"));
    }
    if let Some(value) = report.holdout_validation.mean_abs_error_ms {
        body.push_str(&format!(
            "- Mean absolute holdout error: `{value:.3} ms/step`\n"
        ));
    }
    if let Some(value) = report.holdout_validation.mean_abs_pct_error {
        body.push_str(&format!("- Mean absolute holdout error: `{value:.2}%`\n"));
    }
    if let Some(value) = report.holdout_validation.mean_predicted_ms {
        body.push_str(&format!(
            "- Mean predicted holdout step: `{value:.3} ms/step`\n"
        ));
    }
    if let Some(value) = report.holdout_validation.spearman_rank_correlation {
        body.push_str(&format!("- Spearman rank correlation: `{value:.3}`\n"));
    }
    if !report.holdout_validation.samples.is_empty() {
        body.push_str("\n| Run | Class | Predicted ms | Observed ms | Abs error | Source |\n");
        body.push_str("|---|---|---:|---:|---:|---|\n");
        for sample in &report.holdout_validation.samples {
            body.push_str(&format!(
                "| `{}` | `{}` | {:.3} | {:.3} | {:.3} | `{}` |\n",
                sample.run_key,
                sample.class_label,
                sample.predicted_ms,
                sample.observed_ms,
                sample.abs_error_ms,
                sample.prediction_source
            ));
        }
    }

    body.push_str("\n## Top Calibrated Stages\n\n");
    body.push_str("| Stage | Estimate ms | Samples | Confidence | Source fields |\n");
    body.push_str("|---|---:|---:|---|---|\n");
    for metric in &report.top_stage_metrics {
        body.push_str(&format!(
            "| `{}` | {:.3} | {} | `{}` | `{}` |\n",
            metric.stage,
            metric.estimate_ms,
            metric.sample_count,
            metric.confidence,
            metric.source_fields.join(", ")
        ));
    }

    body.push_str("\n## Scenario\n\n");
    body.push_str(&format!(
        "- Base step: `{:.3} ms/step` from `{}`\n- Requested active recurrent replay cut: `{:.3} ms`\n- Target: `{:.3} ms/step`\n- Predicted step: `{:.3} ms/step`\n- Remaining gap: `{:.3} ms`\n- Risk: `{}`\n",
        scenario.base_step_ms,
        scenario.base_step_source,
        scenario.active_recurrent_replay_cut_ms,
        scenario.target_step_ms,
        scenario.predicted_step_ms,
        scenario.remaining_gap_ms,
        scenario.risk_level
    ));

    body.push_str("\n## Corpus And Suite\n\n");
    body.push_str(&format!(
        "- Index status: `{}`\n- Corpus status: `{}`\n- Calibration status: `{}`\n- Suite status: `{}`\n- Suite mean absolute baseline error: `{}`\n",
        index.status,
        corpus.status,
        calibration.status,
        suite.status,
        suite
            .mean_abs_baseline_step_error_ms
            .map(|value| format!("{value:.3} ms/step"))
            .unwrap_or_else(|| "n/a".to_string())
    ));

    if !report.warnings.is_empty() {
        body.push_str("\n## Warnings\n\n");
        for warning in &report.warnings {
            body.push_str(&format!("- {warning}\n"));
        }
    }

    body.push_str("\n## Generated Files\n\n");
    body.push_str(&format!(
        "- Trace index: `{}`\n- Corpus: `{}`\n- Calibration: `{}`\n- Scenario: `{}`\n- Suite summary: `{}`\n- JSON report: `{}`\n",
        report.trace_index_path,
        report.corpus_path,
        report.calibration_path,
        report.scenario_path,
        report.suite_summary_path,
        report.report_json_path
    ));
    body
}

fn data_preflight_report(spec: &RunSpec) -> PgResult<DataPreflightReport> {
    let canonical_target = spec.name == "frontier_2135_audit_target";
    let train_paths = glob_optional_pattern(spec.train.train_data_pattern.as_deref())?;
    let validation_paths = glob_optional_pattern(spec.train.validation_data_pattern.as_deref())?;
    let sidecar_paths = glob_optional_pattern(spec.eval.caseops_byte_sidecar_pattern.as_deref())?;
    let validation_tokens_found = shard_token_count_sum(&validation_paths);
    let sidecar_tokens_found = shard_token_count_sum(&sidecar_paths);
    let sidecar_structural_ok = if spec.model.caseops.byte_sidecar {
        !sidecar_paths.is_empty() && sidecar_tokens_found.unwrap_or(0) > 0
    } else {
        true
    };
    let train_shards_required = canonical_target.then_some(80);
    let validation_tokens_required = canonical_target.then_some(47_851_520);
    let validation_docs_required = canonical_target.then_some(50_000);
    let train_shards_ok = train_shards_required.map(|required| train_paths.len() == required);
    let validation_tokens_ok = validation_tokens_required
        .zip(validation_tokens_found)
        .map(|(required, found)| found == required);
    let validation_docs_found = None;
    let validation_docs_ok = validation_docs_required
        .zip(validation_docs_found)
        .map(|(required, found)| found == required);
    let sidecar_tokens_match_validation = sidecar_tokens_found
        .zip(validation_tokens_found)
        .map(|(sidecar, validation)| sidecar == validation);

    let mut notes = Vec::new();
    if spec.train.train_data_pattern.is_none() {
        notes.push("train_data_pattern is not configured".to_string());
    }
    if spec.train.validation_data_pattern.is_none() {
        notes.push("validation_data_pattern is not configured".to_string());
    }
    if spec.model.caseops.byte_sidecar && spec.eval.caseops_byte_sidecar_pattern.is_none() {
        notes.push(
            "CaseOps byte sidecar is required but no sidecar pattern is configured".to_string(),
        );
    }
    if let Some(false) = train_shards_ok {
        notes.push(format!(
            "train shard count mismatch: found {} required {}",
            train_paths.len(),
            train_shards_required.unwrap_or_default()
        ));
    }
    if let Some(false) = validation_tokens_ok {
        notes.push(format!(
            "validation token count mismatch: found {:?} required {}",
            validation_tokens_found,
            validation_tokens_required.unwrap_or_default()
        ));
    }
    if validation_docs_required.is_some() && validation_docs_found.is_none() {
        notes.push("validation doc count is not available from local shard headers".to_string());
    }
    if spec.model.caseops.byte_sidecar && !sidecar_structural_ok {
        notes.push(format!(
            "CaseOps sidecar pattern did not resolve to readable token shards: {:?}",
            spec.eval.caseops_byte_sidecar_pattern
        ));
    }
    if let Some(false) = sidecar_tokens_match_validation {
        notes.push(format!(
            "CaseOps sidecar token count {:?} does not match validation token count {:?}",
            sidecar_tokens_found, validation_tokens_found
        ));
    }

    let ready = if canonical_target {
        train_shards_ok == Some(true)
            && validation_tokens_ok == Some(true)
            && sidecar_structural_ok
            && sidecar_tokens_match_validation == Some(true)
            && validation_docs_ok.unwrap_or(false)
    } else {
        !spec.model.caseops.byte_sidecar || sidecar_structural_ok
    };

    Ok(DataPreflightReport {
        kind: "pg_local_data_preflight",
        canonical_target,
        train_pattern: spec.train.train_data_pattern.clone(),
        validation_pattern: spec.train.validation_data_pattern.clone(),
        caseops_sidecar_pattern: spec.eval.caseops_byte_sidecar_pattern.clone(),
        caseops_enabled: spec.model.caseops.enabled,
        caseops_byte_sidecar_required: spec.model.caseops.byte_sidecar,
        train_shards_found: train_paths.len(),
        train_shards_required,
        train_shards_ok,
        validation_shards_found: validation_paths.len(),
        validation_tokens_found,
        validation_tokens_required,
        validation_tokens_ok,
        validation_docs_found,
        validation_docs_required,
        validation_docs_ok,
        sidecar_shards_found: sidecar_paths.len(),
        sidecar_tokens_found,
        sidecar_tokens_match_validation,
        sidecar_structural_ok,
        ready,
        notes,
    })
}

fn glob_optional_pattern(pattern: Option<&str>) -> PgResult<Vec<PathBuf>> {
    let Some(pattern) = pattern else {
        return Ok(Vec::new());
    };
    let mut paths = Vec::new();
    for entry in glob::glob(pattern)
        .map_err(|err| PgError::DataFormat(format!("invalid data glob pattern: {err}")))?
    {
        paths.push(
            entry.map_err(|err| PgError::DataFormat(format!("data glob entry failed: {err}")))?,
        );
    }
    paths.sort();
    Ok(paths)
}

fn shard_token_count_sum(paths: &[PathBuf]) -> Option<usize> {
    let mut total = 0usize;
    for path in paths {
        let count = shard_token_count(path)?;
        total = total.checked_add(count)?;
    }
    Some(total)
}

fn shard_token_count(path: &Path) -> Option<usize> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 12 {
        return None;
    }
    let count = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
    usize::try_from(count).ok()
}

fn artifact_audit_file_ok(path: &Path, total_limit: usize) -> PgResult<bool> {
    let body = fs::read_to_string(path)?;
    let values = json_values_from_body(&body, "record_artifact_audit_json=")?;
    let Some(value) = values
        .iter()
        .rev()
        .find(|value| json_str(value, "event") == Some("record_artifact_audit"))
        .or_else(|| values.last())
    else {
        return Ok(false);
    };
    let total = json_usize(value, "artifact_total_bytes");
    let model = json_usize(value, "artifact_model_bytes");
    let code = json_usize(value, "artifact_code_bytes");
    let limit = json_usize(value, "artifact_total_limit").unwrap_or(total_limit);
    let budget_ok = value
        .get("artifact_budget_ok")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let budget_known = value
        .get("artifact_budget_known")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(total.is_some());
    let Some(total) = total else {
        return Ok(false);
    };
    let Some(model) = model else {
        return Ok(false);
    };
    let Some(code) = code else {
        return Ok(false);
    };
    Ok(budget_known
        && budget_ok
        && model + code <= total
        && total <= total_limit
        && total <= limit
        && json_nonempty_str(value, "artifact_model_sha256")
        && json_nonempty_str(value, "artifact_code_sha256"))
}

fn score_first_log_file_ok(path: &Path) -> PgResult<bool> {
    let body = fs::read_to_string(path)?;
    let values = json_values_from_body(&body, "ttt_audit_json=")?;
    if values.len() > 1 {
        return Ok(ttt_events_score_before_update(&values));
    }
    let Some(value) = values.first() else {
        return Ok(false);
    };
    if let Some(legal) = value
        .get("score_first_legal")
        .and_then(serde_json::Value::as_bool)
    {
        let contaminated = value
            .get("validation_contamination")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        return Ok(legal && !contaminated);
    }
    if let Some(events) = value.get("events").and_then(serde_json::Value::as_array) {
        return Ok(events_score_before_update(events) || ttt_events_score_before_update(events));
    }
    Ok(false)
}

fn local_eval_legality_report() -> PgResult<EvalLegalityReport> {
    let loss = 8.0 * std::f64::consts::LN_2;
    let bpb = compute_bpb(loss, 8.0, 8.0);
    let bpb_byte_accounting_ok = (bpb - 8.0).abs() < 1e-9;

    let spec = LiteSpec {
        model: LiteModelSpec {
            family: LiteModelFamily::NgramResidual,
            vocab: "byte".to_string(),
            context: 8,
            residual_buckets: 16,
            residual_weight: 0.25,
            artifact_path: None,
        },
        ..LiteSpec::default()
    };
    let model = LiteNgramResidual::train(b"aaaaabbbbbabababab", &spec);
    let static_eval = model.loss_on_bytes(b"abababab", false);
    let score_first = model.loss_on_bytes_score_first(b"abababab");
    let score_before_update_order_ok = score_first.score_first_tokens_scored
        == score_first.score_first_update_count
        && score_first.score_first_tokens_scored == 8;
    let score_first_loss_not_worse_than_static = score_first.loss <= static_eval.loss + 1e-9;

    let doc_a = b"abab";
    let doc_b = b"baba";
    let separate_docs =
        model.loss_on_bytes(doc_a, false).loss + model.loss_on_bytes(doc_b, false).loss;
    let doc_eval = model.loss_on_documents(&[doc_a.as_slice(), doc_b.as_slice()], false);
    let document_boundary_reset_ok = (doc_eval.loss - separate_docs).abs() < 1e-9
        && doc_eval.tokens == doc_a.len() + doc_b.len();

    let artifact = model.to_stored_artifact("local-eval-legality".to_string());
    let decoded = LiteNgramResidual::from_stored_artifact(&artifact)?;
    let decoded_eval = decoded.loss_on_bytes(b"abababab", false);
    let artifact_decode_eval_equivalence_ok = (decoded_eval.loss - static_eval.loss).abs() < 1e-9
        && decoded_eval.tokens == static_eval.tokens;

    let artifact_fingerprint = artifact.model_fingerprint;
    let status = if bpb_byte_accounting_ok
        && score_before_update_order_ok
        && score_first_loss_not_worse_than_static
        && document_boundary_reset_ok
        && artifact_decode_eval_equivalence_ok
    {
        "pass"
    } else {
        "fail"
    }
    .to_string();

    Ok(EvalLegalityReport {
        kind: "pg_local_eval_legality",
        bpb_byte_accounting_ok,
        score_before_update_order_ok,
        score_first_tokens_scored: score_first.score_first_tokens_scored,
        score_first_update_count: score_first.score_first_update_count,
        score_first_loss_not_worse_than_static,
        no_future_token_access: score_before_update_order_ok,
        document_boundary_reset_ok,
        artifact_decode_eval_equivalence_ok,
        artifact_fingerprint,
        status,
    })
}

fn json_values_from_body(body: &str, line_prefix: &str) -> PgResult<Vec<serde_json::Value>> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        return Ok(match value {
            serde_json::Value::Array(values) => values,
            value => vec![value],
        });
    }

    let mut values = Vec::new();
    for line in body.lines() {
        let Some((_, raw)) = line.split_once(line_prefix) else {
            continue;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .map_err(|err| PgError::DataFormat(format!("invalid log JSON record: {err}")))?;
        values.push(value);
    }
    Ok(values)
}

fn parse_trace_body(body: &str) -> Option<serde_json::Value> {
    let mut candidates = Vec::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        collect_json_trace_candidates(&value, &mut candidates);
    }
    collect_prefixed_json_trace_candidates(body, "run_timing_json=", &mut candidates);
    collect_prefixed_json_trace_candidates(body, "finish_status_json=", &mut candidates);
    if let Some(flat) = flat_timing_trace_from_body(body) {
        candidates.push(flat);
    }
    candidates
        .into_iter()
        .max_by(|a, b| trace_candidate_score(a).cmp(&trace_candidate_score(b)))
}

fn parse_trace_aux_body(body: &str) -> Option<serde_json::Value> {
    let mut candidates = Vec::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if value.is_object() {
            candidates.push(value.clone());
        }
        if let Some(events) = value
            .get("json_events")
            .and_then(serde_json::Value::as_object)
        {
            for key in ["record_audit_json", "finish_status_json", "finish_status"] {
                append_trace_values_from_json_events(events.get(key), &mut candidates);
            }
        }
    }
    collect_prefixed_json_trace_candidates(body, "record_audit_json=", &mut candidates);
    collect_prefixed_json_trace_candidates(body, "finish_status_json=", &mut candidates);
    candidates.into_iter().max_by_key(|value| {
        [
            "canonical_caseops_dataset",
            "record_profile",
            "recurrent_backward_profile",
            "cuda_graph_profile",
            "train_shards",
            "val_tokens",
            "mode",
            "run_name",
        ]
        .iter()
        .filter(|key| value.get(**key).is_some())
        .count()
    })
}

fn collect_json_trace_candidates(
    value: &serde_json::Value,
    candidates: &mut Vec<serde_json::Value>,
) {
    if value.is_object() {
        candidates.push(value.clone());
    }
    if let Some(events) = value
        .get("json_events")
        .and_then(serde_json::Value::as_object)
    {
        for key in [
            "run_timing_json",
            "run_timing",
            "finish_status_json",
            "finish_status",
            "record_audit_json",
        ] {
            append_trace_values_from_json_events(events.get(key), candidates);
        }
    }
    for key in ["run_timing_json", "run_timing", "metrics"] {
        append_trace_values_from_json_events(value.get(key), candidates);
    }
}

fn append_trace_values_from_json_events(
    value: Option<&serde_json::Value>,
    candidates: &mut Vec<serde_json::Value>,
) {
    match value {
        Some(serde_json::Value::Array(values)) => {
            for entry in values {
                if entry.is_object() {
                    candidates.push(entry.clone());
                }
            }
        }
        Some(value) if value.is_object() => candidates.push(value.clone()),
        _ => {}
    }
}

fn collect_prefixed_json_trace_candidates(
    body: &str,
    line_prefix: &str,
    candidates: &mut Vec<serde_json::Value>,
) {
    if let Ok(values) = json_values_from_body(body, line_prefix) {
        for value in values {
            if value.is_object() {
                candidates.push(value);
            }
        }
    }
}

fn flat_timing_trace_from_body(body: &str) -> Option<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for line in body.lines() {
        let Some((key, raw)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if !is_trace_scalar_key(key) {
            continue;
        }
        let raw = raw.trim();
        if let Ok(number) = raw.parse::<f64>()
            && let Some(value) = serde_json::Number::from_f64(number)
        {
            map.insert(key.to_string(), serde_json::Value::Number(value));
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(map))
    }
}

fn is_trace_scalar_key(key: &str) -> bool {
    key.starts_with("timing_")
        || key.ends_with("_ms_per_step")
        || key.ends_with("_bridge_launches")
        || key == "ms_per_step"
        || key == "steps_completed"
        || key == "timing_steps"
        || key == "world_size"
        || key == "seq_len"
        || key == "global_batch_tokens"
}

fn trace_candidate_score(value: &serde_json::Value) -> usize {
    let trace_fields = TRACE_COVERAGE_FIELDS
        .iter()
        .filter(|field| trace_number_for_aliases(value, field.aliases).is_some())
        .count();
    let stage_fields = TRACE_STAGE_MAPPINGS
        .iter()
        .filter(|mapping| trace_number_for_aliases(value, mapping.aliases).is_some())
        .count();
    let total_bonus =
        usize::from(trace_number(value, "timing_measured_ms_per_step").is_some()) * 10;
    total_bonus + trace_fields * 3 + stage_fields
}

fn trace_usize(value: &serde_json::Value, key: &str) -> Option<usize> {
    trace_number(value, key).and_then(|number| {
        if number.is_finite() && number >= 0.0 {
            Some(number.round() as usize)
        } else {
            None
        }
    })
}

fn trace_usize_chain(
    primary: &serde_json::Value,
    aux: Option<&serde_json::Value>,
    keys: &[&str],
) -> Option<usize> {
    for key in keys {
        if let Some(value) = trace_usize(primary, key) {
            return Some(value);
        }
        if let Some(value) = aux.and_then(|aux| trace_usize(aux, key)) {
            return Some(value);
        }
    }
    None
}

fn trace_string(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("metrics")
                .and_then(|metrics| metrics.get(key))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
        })
        .or_else(|| {
            value
                .get("run_timing")
                .and_then(|timing| timing.get(key))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
        })
}

fn trace_string_chain(
    primary: &serde_json::Value,
    aux: Option<&serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    for key in keys {
        if let Some(value) = trace_string(primary, key) {
            return Some(value);
        }
        if let Some(value) = aux.and_then(|aux| trace_string(aux, key)) {
            return Some(value);
        }
    }
    None
}

fn trace_bool(value: &serde_json::Value, key: &str) -> Option<bool> {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            value
                .get("metrics")
                .and_then(|metrics| metrics.get(key))
                .and_then(serde_json::Value::as_bool)
        })
        .or_else(|| {
            value
                .get("run_timing")
                .and_then(|timing| timing.get(key))
                .and_then(serde_json::Value::as_bool)
        })
}

fn trace_bool_chain(
    primary: &serde_json::Value,
    aux: Option<&serde_json::Value>,
    keys: &[&str],
) -> Option<bool> {
    for key in keys {
        if let Some(value) = trace_bool(primary, key) {
            return Some(value);
        }
        if let Some(value) = aux.and_then(|aux| trace_bool(aux, key)) {
            return Some(value);
        }
    }
    None
}

fn trace_tags(
    path: &Path,
    total_ms: Option<f64>,
    active_ms: Option<f64>,
    calibration_role: &str,
    canonical_caseops: Option<bool>,
    mode: Option<String>,
    recurrent_profile: Option<String>,
) -> Vec<String> {
    let label = path.to_string_lossy().to_ascii_lowercase();
    let mut tags = vec![calibration_role.to_string()];
    if total_ms.is_some() {
        tags.push("timed".to_string());
    }
    if active_ms.is_some() {
        tags.push("active_recurrent_timed".to_string());
    }
    if label.contains("2135") {
        tags.push("frontier_2135".to_string());
    }
    if label.contains("1855") || label.contains("v86") || label.contains("v87") {
        tags.push("frontier_1855_or_legacy".to_string());
    }
    if label.contains("exact") {
        tags.push("exact_2135".to_string());
    }
    if label.contains("allst") || label.contains("straight") {
        tags.push("straight_through_diagnostic".to_string());
    }
    if label.contains("stage") {
        tags.push("stage_probe".to_string());
    }
    if label.contains("finish_status") {
        tags.push("finish_status".to_string());
    }
    if canonical_caseops == Some(true) {
        tags.push("canonical_caseops".to_string());
    } else if canonical_caseops == Some(false) {
        tags.push("noncanonical_or_unknown_caseops".to_string());
    }
    if mode
        .as_deref()
        .is_some_and(|mode| mode.to_ascii_lowercase().contains("recordshaped"))
    {
        tags.push("record_shaped_proxy".to_string());
    }
    if recurrent_profile
        .as_deref()
        .is_some_and(|profile| profile.to_ascii_lowercase().contains("exact"))
    {
        tags.push("exact_recurrent_profile".to_string());
    }
    tags
}

fn events_score_before_update(events: &[serde_json::Value]) -> bool {
    use std::collections::{HashMap, HashSet};

    let mut scored = HashSet::new();
    let mut updated = HashMap::<String, usize>::new();
    for (idx, event) in events.iter().enumerate() {
        let doc = event
            .get("doc")
            .or_else(|| event.get("doc_id"))
            .or_else(|| event.get("chunk_id"))
            .map(|value| match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => String::new(),
            })
            .unwrap_or_else(|| idx.to_string());
        let kind = event
            .get("kind")
            .or_else(|| event.get("event"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        match kind {
            "score" | "scored" | "score_start" | "score_end" => {
                if updated.contains_key(&doc) {
                    return false;
                }
                scored.insert(doc);
            }
            "update" | "adapt" | "ttt_update" => {
                if !scored.contains(&doc) {
                    return false;
                }
                updated.insert(doc, idx);
            }
            _ => {}
        }
    }
    !scored.is_empty()
}

fn ttt_events_score_before_update(events: &[serde_json::Value]) -> bool {
    let mut saw_start = false;
    let mut saw_chunk = false;
    let mut saw_done = false;
    let mut previous_cumulative = 0u64;

    for event in events {
        if event
            .get("future_token_access")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return false;
        }
        match json_str(event, "event").unwrap_or_default() {
            "gpu_lora_phased_ttt_start" => {
                if !event
                    .get("score_first")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    return false;
                }
                saw_start = true;
            }
            "gpu_lora_phased_ttt_chunk" => {
                let scored = json_u64(event, "tokens_scored_before_update").unwrap_or(0);
                let cumulative = json_u64(event, "cumulative_tokens_scored").unwrap_or(0);
                if scored == 0 || cumulative < previous_cumulative + scored {
                    return false;
                }
                previous_cumulative = cumulative;

                if event
                    .get("ttt_update_after_score")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    let chunk_start = json_u64(event, "chunk_start").unwrap_or(0);
                    let chunk_end = json_u64(event, "chunk_end").unwrap_or(0);
                    let update_start = json_u64(event, "update_start").unwrap_or(u64::MAX);
                    let update_end = json_u64(event, "update_end").unwrap_or(u64::MAX);
                    if update_start < chunk_start
                        || update_end > chunk_end
                        || update_end < update_start
                    {
                        return false;
                    }
                }
                saw_chunk = true;
            }
            "gpu_lora_phased_ttt_done" => {
                if !event
                    .get("score_first")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    return false;
                }
                if json_u64(event, "tokens_scored").unwrap_or(0) == 0 {
                    return false;
                }
                saw_done = true;
            }
            _ => {}
        }
    }

    saw_start && saw_chunk && saw_done
}

fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(serde_json::Value::as_u64)
}

fn json_usize(value: &serde_json::Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
}

fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
}

fn json_nonempty_str(value: &serde_json::Value, key: &str) -> bool {
    json_str(value, key).is_some_and(|raw| !raw.is_empty())
}

fn proposal_feature_report() -> ProposalFeatureReport {
    ProposalFeatureReport {
        bf16_direct_compact: "h100_validated".to_string(),
        device_resident_sampler: "h100_validated".to_string(),
        no_loss_backward_graph: "h100_validated".to_string(),
        full_train_step_graph: "not_implemented".to_string(),
        persistent_cta_block_backward: "not_implemented".to_string(),
        xsa_inside_sdpa: "local_parity_tested_not_record_active".to_string(),
        adjacent_sparse_xsa: "implemented".to_string(),
        quant_layout_compiler: "local_parity_tested".to_string(),
        bigramhash_fusion: "local_parity_tested".to_string(),
        pg_lite: "local_validated".to_string(),
        pg_lite_metal_backend: if cfg!(feature = "metal_apple") {
            "runtime_linked_requires_local_backend_check".to_string()
        } else {
            "source_boundary_checked_feature_gated".to_string()
        },
        wind_tunnel: "local_validated".to_string(),
    }
}

fn deterministic_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((state >> 40) as u32) & 0xffff;
        let centered = bits as f32 / 65535.0 - 0.5;
        out.push(centered * scale);
    }
    out
}

fn simulate_bank_dist(
    name: &str,
    shape: [usize; 3],
    bank_idx: usize,
    world_size: usize,
    steps: usize,
    seed: u64,
) -> DistSimBankReport {
    let sim_batches = 1usize;
    let sim_rows = world_size.max(1);
    let sim_cols = shape[2].clamp(1, 8);
    let sim_shape = [sim_batches, sim_rows, sim_cols];
    let sim_len = sim_batches * sim_rows * sim_cols;
    let rank_grads = (0..world_size)
        .map(|rank| {
            deterministic_values(
                sim_len,
                seed ^ ((bank_idx as u64 + 1) << 32) ^ rank as u64,
                0.02,
            )
        })
        .collect::<Vec<_>>();

    let mut reference_full =
        deterministic_values(sim_len, seed ^ 0x55aa_0101_u64 ^ bank_idx as u64, 0.05);
    let mut sharded_full = reference_full.clone();

    for step in 0..steps {
        let shifted_grads = rank_grads
            .iter()
            .enumerate()
            .map(|(rank, grad)| {
                grad.iter()
                    .enumerate()
                    .map(|(idx, value)| {
                        let salt = ((step + 1) * (rank + 3) * (idx + 5)) as f32;
                        value + salt.sin() * 0.0001
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let grad_shards = reduce_scatter_rows(&shifted_grads, world_size, sim_rows, sim_cols);
        let reference_param_shards = scatter_rows(&reference_full, world_size, sim_rows, sim_cols);
        let sharded_param_shards = scatter_rows(&sharded_full, world_size, sim_rows, sim_cols);
        let mut reference_updated_shards = Vec::with_capacity(world_size);
        let mut updated_shards = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let rows = row_range_for_rank(rank, world_size, sim_rows);
            let shard_shape = [1, rows.1.saturating_sub(rows.0), sim_cols];
            let mut reference_param = reference_param_shards[rank].clone();
            let mut sharded_param = sharded_param_shards[rank].clone();
            let mut reference_muon = Muon::new(0.01, 0.9, 2, true, 0.0, &[shard_shape]);
            let mut sharded_muon = Muon::new(0.01, 0.9, 2, true, 0.0, &[shard_shape]);
            reference_muon.step_bank(0, &mut reference_param, &grad_shards[rank], &shard_shape);
            sharded_muon.step_bank(0, &mut sharded_param, &grad_shards[rank], &shard_shape);
            reference_updated_shards.push(reference_param);
            updated_shards.push(sharded_param);
        }
        reference_full = all_gather_rows(&reference_updated_shards);
        sharded_full = all_gather_rows(&updated_shards);
    }

    let max_abs = reference_full
        .iter()
        .zip(sharded_full.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let elems = shape[0] * shape[1] * shape[2];
    let total_bytes = elems * std::mem::size_of::<f32>();
    DistSimBankReport {
        name: name.to_string(),
        shape,
        simulated_shape: sim_shape,
        reduce_scatter_bytes_per_step: total_bytes / world_size.max(1),
        all_gather_bytes_per_step: total_bytes / world_size.max(1),
        optimizer_parity_max_abs_diff: max_abs,
        equivalent: max_abs <= 1e-5,
    }
}

fn sum_rank_grads(rank_grads: &[Vec<f32>]) -> Vec<f32> {
    let mut summed = vec![0.0; rank_grads.first().map(Vec::len).unwrap_or(0)];
    for grad in rank_grads {
        for (dst, src) in summed.iter_mut().zip(grad.iter()) {
            *dst += *src;
        }
    }
    summed
}

fn reduce_scatter_rows(
    rank_grads: &[Vec<f32>],
    world_size: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<f32>> {
    let summed = sum_rank_grads(rank_grads);
    scatter_rows(&summed, world_size, rows, cols)
}

fn scatter_rows(values: &[f32], world_size: usize, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut shards = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let (start, end) = row_range_for_rank(rank, world_size, rows);
        shards.push(values[start * cols..end * cols].to_vec());
    }
    shards
}

fn all_gather_rows(shards: &[Vec<f32>]) -> Vec<f32> {
    let total = shards.iter().map(Vec::len).sum();
    let mut gathered = Vec::with_capacity(total);
    for shard in shards {
        gathered.extend_from_slice(shard);
    }
    gathered
}

fn row_range_for_rank(rank: usize, world_size: usize, rows: usize) -> (usize, usize) {
    let start = rank * rows / world_size.max(1);
    let end = (rank + 1) * rows / world_size.max(1);
    (start, end)
}

fn normalize_sweeps(raw: &[String]) -> Vec<String> {
    let mut sweeps = raw
        .iter()
        .flat_map(|entry| entry.split(','))
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if sweeps.is_empty() {
        sweeps.extend([
            "quant".to_string(),
            "lqer".to_string(),
            "compression".to_string(),
        ]);
    }
    sweeps.sort();
    sweeps.dedup();
    sweeps
}

fn artifact_sweeps(
    kernel_set: &CompiledQuantKernelSet,
    requested_sweeps: &[String],
) -> PgResult<Vec<ArtifactSweepReport>> {
    let values = (-64..64)
        .cycle()
        .take(512)
        .map(|x| x as f32 / 16.0)
        .collect::<Vec<_>>();
    let mut reports = Vec::new();
    if requested_sweeps.iter().any(|sweep| sweep == "quant") {
        for bits in 4..=8 {
            let (bytes, mse, max_abs) = quant_roundtrip_stats(kernel_set, &values, bits)?;
            reports.push(ArtifactSweepReport {
                name: format!("synthetic_int{bits}_per_row"),
                sweep_kind: "quant".to_string(),
                bits: Some(bits),
                bytes,
                reconstruction_mse: mse,
                reconstruction_max_abs: max_abs,
                proxy_only: true,
            });
        }
    }
    if requested_sweeps.iter().any(|sweep| sweep == "lqer") {
        let (bytes, mse, max_abs) = lqer_proxy_stats(&values);
        reports.push(ArtifactSweepReport {
            name: "synthetic_lqer_rank1_residual_proxy".to_string(),
            sweep_kind: "lqer".to_string(),
            bits: None,
            bytes,
            reconstruction_mse: mse,
            reconstruction_max_abs: max_abs,
            proxy_only: true,
        });
    }
    if requested_sweeps.iter().any(|sweep| sweep == "compression") {
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = zstd::encode_all(raw.as_slice(), 19)?;
        reports.push(ArtifactSweepReport {
            name: "synthetic_zstd19_weight_stream".to_string(),
            sweep_kind: "compression".to_string(),
            bits: None,
            bytes: compressed.len(),
            reconstruction_mse: 0.0,
            reconstruction_max_abs: 0.0,
            proxy_only: true,
        });
    }
    Ok(reports)
}

fn mixed_bit_allocation_reports(
    groups: &[pg_quant::CompiledQuantGroupManifest],
    target_artifact_bytes: usize,
    current_total_bytes: Option<usize>,
) -> Vec<ArtifactMixedBitAllocationReport> {
    let current = groups.iter().fold((6u8, 5u8, 8u8, 6u8), |mut acc, group| {
        match quant_group_role(group.name) {
            QuantGroupRole::Matrix => acc.0 = group.bits,
            QuantGroupRole::Mlp => acc.1 = group.bits,
            QuantGroupRole::Embed => acc.2 = group.bits,
            QuantGroupRole::AttnGate => acc.3 = group.bits,
        }
        acc
    });
    [
        (
            "current_manifest",
            current.0,
            current.1,
            current.2,
            current.3,
        ),
        ("all_int5", 5, 5, 5, 5),
        ("all_int4_floor", 4, 4, 4, 4),
        ("matrix6_mlp4_embed8", 6, 4, 8, current.3),
        ("matrix5_mlp4_embed6", 5, 4, 6, current.3.min(5)),
        ("embed8_mlp5_gate4", current.0, 5, 8, 4),
    ]
    .into_iter()
    .map(
        |(name, matrix_bits, mlp_bits, embed_bits, attn_gate_bits)| {
            let (weights, scales, lqer) = estimate_group_bytes_for_bits(
                groups,
                matrix_bits,
                mlp_bits,
                embed_bits,
                attn_gate_bits,
            );
            let total = weights + scales + lqer;
            ArtifactMixedBitAllocationReport {
                name: name.to_string(),
                matrix_bits,
                mlp_bits,
                embed_bits,
                attn_gate_bits,
                estimated_weight_bytes: weights,
                scale_bytes: scales,
                lqer_bytes: lqer,
                total_bytes_estimate: total,
                target_artifact_bytes,
                budget_ok: total <= target_artifact_bytes,
                delta_vs_current_bytes: current_total_bytes
                    .map(|current| total as isize - current as isize),
                proxy_only: true,
            }
        },
    )
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantGroupRole {
    Matrix,
    Mlp,
    Embed,
    AttnGate,
}

fn quant_group_role(name: &str) -> QuantGroupRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("embed") || lower.contains("tok_emb") {
        QuantGroupRole::Embed
    } else if lower.contains("mlp") {
        QuantGroupRole::Mlp
    } else if lower.contains("gate") {
        QuantGroupRole::AttnGate
    } else {
        QuantGroupRole::Matrix
    }
}

fn estimate_group_bytes_for_bits(
    groups: &[pg_quant::CompiledQuantGroupManifest],
    matrix_bits: u8,
    mlp_bits: u8,
    embed_bits: u8,
    attn_gate_bits: u8,
) -> (usize, usize, usize) {
    let mut weight_bytes = 0usize;
    let mut scale_bytes = 0usize;
    let mut lqer_bytes = 0usize;
    for group in groups {
        let bits = match quant_group_role(group.name) {
            QuantGroupRole::Matrix => matrix_bits,
            QuantGroupRole::Mlp => mlp_bits,
            QuantGroupRole::Embed => embed_bits,
            QuantGroupRole::AttnGate => attn_gate_bits,
        };
        let elems = group.rows.zip(group.cols).map(|(r, c)| r.saturating_mul(c));
        weight_bytes = weight_bytes.saturating_add(
            elems
                .map(|elems| elems.saturating_mul(bits as usize).div_ceil(8))
                .unwrap_or(group.packed_weight_bytes.unwrap_or(0)),
        );
        scale_bytes = scale_bytes.saturating_add(group.scale_bytes.unwrap_or(0));
        lqer_bytes = lqer_bytes.saturating_add(group.lqer_bytes.unwrap_or(0));
    }
    (weight_bytes, scale_bytes, lqer_bytes)
}

fn artifact_mini_bpb(
    mini_train: Option<&Path>,
    mini_val: Option<&Path>,
) -> PgResult<Option<ArtifactMiniBpbReport>> {
    match (mini_train, mini_val) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(PgError::InvalidOp(
            "artifact-lab mini BPB requires both --mini-train and --mini-val".into(),
        )),
        (Some(train_path), Some(val_path)) => {
            let train = fs::read(train_path)?;
            let val = fs::read(val_path)?;
            if val.is_empty() {
                return Err(PgError::DataFormat(
                    "artifact-lab mini validation bytes are empty".into(),
                ));
            }
            let spec = LiteSpec {
                model: LiteModelSpec {
                    family: LiteModelFamily::NgramResidual,
                    vocab: "byte".to_string(),
                    context: 512,
                    residual_buckets: 1024,
                    residual_weight: 0.15,
                    artifact_path: None,
                },
                ..LiteSpec::default()
            };
            let model = LiteNgramResidual::train(&train, &spec);
            let eval = model.loss_on_bytes(&val, false);
            let avg_loss = eval.loss / eval.tokens.max(1) as f64;
            let validation_bpb = compute_bpb(avg_loss, eval.tokens as f64, eval.tokens as f64);
            Ok(Some(ArtifactMiniBpbReport {
                train_path: train_path.display().to_string(),
                val_path: val_path.display().to_string(),
                train_bytes: train.len(),
                val_bytes: val.len(),
                model_family: LiteModelFamily::NgramResidual,
                validation_bpb,
                score_scope: "artifact_lab_local_proxy_bpb_not_leaderboard",
                proxy_only: true,
            }))
        }
    }
}

fn quant_roundtrip_stats(
    kernel_set: &CompiledQuantKernelSet,
    values: &[f32],
    bits: u8,
) -> PgResult<(usize, f64, f32)> {
    let qmax = pg_quant::pack::qmax_for_bits(bits) as f32;
    let scale = values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
        .max(1e-6)
        / qmax.max(1.0);
    let q = values
        .iter()
        .map(|value| (value / scale).round().clamp(-qmax - 1.0, qmax) as i8)
        .collect::<Vec<_>>();
    let packed = kernel_set.pack_signed(&q, bits)?;
    let scales = half::f16::from_f32(scale).to_bits().to_le_bytes().to_vec();
    let mut dequant = vec![0.0f32; values.len()];
    kernel_set.dequant_per_row(&packed, &scales, 1, values.len(), bits, &mut dequant)?;
    let (mse, max_abs) = reconstruction_stats(values, &dequant);
    Ok((packed.len() + scales.len(), mse, max_abs))
}

fn lqer_proxy_stats(values: &[f32]) -> (usize, f64, f32) {
    let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
    let candidate = vec![mean; values.len()];
    let (mse, max_abs) = reconstruction_stats(values, &candidate);
    // rank-1 proxy: one f16 row factor plus one f16 col factor for a 1xN row.
    let bytes = 2 + values.len() * 2;
    (bytes, mse, max_abs)
}

fn reconstruction_stats(reference: &[f32], candidate: &[f32]) -> (f64, f32) {
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&a, &b) in reference.iter().zip(candidate.iter()) {
        let diff = a - b;
        sum_sq += (diff as f64) * (diff as f64);
        max_abs = max_abs.max(diff.abs());
    }
    (sum_sq / reference.len().max(1) as f64, max_abs)
}

fn default_stage_estimates(plan: &ExecutionPlan) -> Vec<StageEstimate> {
    let cfg = plan.run_spec.model.to_model_config();
    let tokens = plan.run_spec.train.batch_tokens.max(1) as f64;
    let layer_scale = cfg.num_layers as f64 / 11.0;
    let mut stages = vec![
        StageEstimate {
            name: "forward_replay".to_string(),
            ms: 42.0 * layer_scale * tokens / 524_288.0,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "mlp_backward".to_string(),
            ms: 20.0 * layer_scale,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "qkv_backward".to_string(),
            ms: 21.0 * layer_scale,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "attention_backward".to_string(),
            ms: 14.0 * layer_scale,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "gate_xsa_backward".to_string(),
            ms: 12.5 * layer_scale,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "output_ce".to_string(),
            ms: 7.2,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "optimizer_update".to_string(),
            ms: 15.0,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
        StageEstimate {
            name: "non_bank_update".to_string(),
            ms: 2.5,
            pct_of_step: 0.0,
            source: "shape_model_default".to_string(),
        },
    ];
    if plan.run_spec.model.recurrence.enabled {
        stages.push(StageEstimate {
            name: "recurrent_pass2_backward".to_string(),
            ms: 0.0,
            pct_of_step: 0.0,
            source: "trace_only_recurrent_stage".to_string(),
        });
        stages.push(StageEstimate {
            name: "recurrent_pass1_backward".to_string(),
            ms: 0.0,
            pct_of_step: 0.0,
            source: "trace_only_recurrent_stage".to_string(),
        });
    }
    stages
}

fn recurrent_profile_label(profile: RecurrentBackwardProfile) -> &'static str {
    match profile {
        RecurrentBackwardProfile::Full => "full",
        RecurrentBackwardProfile::ExactFused => "exact_fused",
        RecurrentBackwardProfile::Pass1StraightThrough => "pass1_straight_through",
        RecurrentBackwardProfile::AllStraightThrough => "all_straight_through",
    }
}

fn exact_recurrent_boundary_fusion_status(spec: &RunSpec) -> String {
    if !spec.model.recurrence.enabled {
        return "inactive_no_recurrence".to_string();
    }
    match spec.runtime.recurrent_backward_profile {
        RecurrentBackwardProfile::AllStraightThrough
        | RecurrentBackwardProfile::Pass1StraightThrough => {
            "inactive_straight_through_profile".to_string()
        }
        RecurrentBackwardProfile::ExactFused => "enabled_exact_boundary_only".to_string(),
        RecurrentBackwardProfile::Full if spec.runtime.recurrent_fused_pass_boundary_backward => {
            "enabled_exact_boundary_only".to_string()
        }
        RecurrentBackwardProfile::Full => "disabled".to_string(),
    }
}

fn recurrent_default_active_penalty_ms(spec: &RunSpec) -> f64 {
    if !spec.model.recurrence.enabled {
        return 0.0;
    }
    match spec.runtime.recurrent_backward_profile {
        RecurrentBackwardProfile::AllStraightThrough => 0.0,
        RecurrentBackwardProfile::Pass1StraightThrough => 9.0,
        RecurrentBackwardProfile::Full | RecurrentBackwardProfile::ExactFused => 18.0,
    }
}

fn exact_recurrent_boundary_layers(plan: &ExecutionPlan) -> Vec<usize> {
    let cfg = plan.run_spec.model.to_model_config();
    (0..cfg.num_layers)
        .filter(|&layer| cfg.is_recurrent_layer(layer))
        .collect()
}

fn build_operation_dag(plan: &ExecutionPlan, exact_boundary_fusion: &str) -> Vec<OperationDagNode> {
    let cfg = plan.run_spec.model.to_model_config();
    let tokens = plan.run_spec.train.batch_tokens.max(1) as u64;
    let d = cfg.model_dim as u64;
    let kv = cfg.kv_dim() as u64;
    let mlp = cfg.mlp_dim as u64;
    let vocab = cfg.vocab_size as u64;
    let mut nodes = Vec::new();
    nodes.push(OperationDagNode {
        name: "token_embedding".to_string(),
        op_kind: "embedding_lookup".to_string(),
        status: "implemented".to_string(),
        layer: None,
        flops_estimate: 0,
        hbm_bytes_estimate: tokens * d * 2,
        record_relevant: true,
    });
    for layer in 0..cfg.num_layers {
        let recurrent_multiplier = if cfg.is_recurrent_layer(layer) { 2 } else { 1 };
        nodes.push(OperationDagNode {
            name: format!("layer_{layer}_qkv_projection"),
            op_kind: "bf16_gemm_qkv".to_string(),
            status: "implemented".to_string(),
            layer: Some(layer),
            flops_estimate: recurrent_multiplier * 2 * tokens * d * (d + 2 * kv),
            hbm_bytes_estimate: recurrent_multiplier * tokens * (d + 2 * kv) * 2,
            record_relevant: true,
        });
        nodes.push(OperationDagNode {
            name: format!("layer_{layer}_sdpa"),
            op_kind: if layer >= cfg.num_layers.saturating_sub(cfg.xsa_last_n) {
                "sdpa_adjacent_xsa"
            } else {
                "sdpa"
            }
            .to_string(),
            status: if layer >= cfg.num_layers.saturating_sub(cfg.xsa_last_n) {
                "adjacent_sparse_xsa"
            } else {
                "implemented"
            }
            .to_string(),
            layer: Some(layer),
            flops_estimate: recurrent_multiplier * 4 * tokens * d * cfg.train_seq_len as u64,
            hbm_bytes_estimate: recurrent_multiplier * tokens * d * 4,
            record_relevant: true,
        });
        nodes.push(OperationDagNode {
            name: format!("layer_{layer}_mlp"),
            op_kind: "bf16_mlp_up_down".to_string(),
            status: "implemented".to_string(),
            layer: Some(layer),
            flops_estimate: recurrent_multiplier * 4 * tokens * d * mlp,
            hbm_bytes_estimate: recurrent_multiplier * tokens * (d + mlp) * 2,
            record_relevant: true,
        });
        if cfg.is_recurrent_layer(layer) {
            nodes.push(OperationDagNode {
                name: format!("layer_{layer}_recurrent_pass_boundary"),
                op_kind: "exact_recurrent_boundary_fusion".to_string(),
                status: exact_boundary_fusion.to_string(),
                layer: Some(layer),
                flops_estimate: 0,
                hbm_bytes_estimate: 0,
                record_relevant: exact_boundary_fusion == "enabled_exact_boundary_only",
            });
        }
    }
    nodes.push(OperationDagNode {
        name: "output_projection_ce".to_string(),
        op_kind: "chunked_bf16_cache_ce".to_string(),
        status: "implemented".to_string(),
        layer: None,
        flops_estimate: 2 * tokens * d * vocab,
        hbm_bytes_estimate: tokens * d * 2,
        record_relevant: true,
    });
    nodes.push(OperationDagNode {
        name: "sharded_parallel_muon".to_string(),
        op_kind: "optimizer_reduce_scatter_update_all_gather".to_string(),
        status: "implemented".to_string(),
        layer: None,
        flops_estimate: 0,
        hbm_bytes_estimate: (plan.bank_layout.qo_bank_elems
            + plan.bank_layout.kv_bank_elems
            + plan.bank_layout.mlp_up_bank_elems
            + plan.bank_layout.mlp_down_bank_elems) as u64
            * 4,
        record_relevant: true,
    });
    nodes.push(OperationDagNode {
        name: "persistent_cta_block_backward".to_string(),
        op_kind: "persistent_cta_recurrent_backward".to_string(),
        status: "not_implemented_unclaimed".to_string(),
        layer: None,
        flops_estimate: 0,
        hbm_bytes_estimate: 0,
        record_relevant: false,
    });
    nodes
}

struct TraceStageMapping {
    canonical: &'static str,
    aliases: &'static [&'static str],
    stage_name: &'static str,
}

struct TraceCoverageField {
    canonical: &'static str,
    aliases: &'static [&'static str],
}

const TRACE_STAGE_MAPPINGS: &[TraceStageMapping] = &[
    TraceStageMapping {
        canonical: "backward_block_mlp_ms",
        aliases: &[
            "backward_block_mlp_ms",
            "timing_cuda_backward_block_mlp_ms_per_step",
        ],
        stage_name: "mlp_backward",
    },
    TraceStageMapping {
        canonical: "backward_block_qkv_ms",
        aliases: &[
            "backward_block_qkv_ms",
            "timing_cuda_backward_block_qkv_ms_per_step",
        ],
        stage_name: "qkv_backward",
    },
    TraceStageMapping {
        canonical: "backward_block_attention_sdpa_ms",
        aliases: &[
            "backward_block_attention_sdpa_ms",
            "timing_cuda_backward_block_attention_sdpa_ms_per_step",
            "timing_cuda_backward_block_attention_ms_per_step",
        ],
        stage_name: "attention_backward",
    },
    TraceStageMapping {
        canonical: "backward_block_attn_out_gate_xsa_ms",
        aliases: &[
            "backward_block_attn_out_gate_xsa_ms",
            "timing_cuda_backward_block_attn_out_gate_xsa_ms_per_step",
            "timing_cuda_backward_block_attention_xsa_accum_ms_per_step",
        ],
        stage_name: "gate_xsa_backward",
    },
    TraceStageMapping {
        canonical: "timing_cuda_backward_recurrent_pass2_ms_per_step",
        aliases: &[
            "timing_cuda_backward_recurrent_pass2_ms_per_step",
            "timing_cuda_backward_recurrent_pass2_ms_per_active_step",
        ],
        stage_name: "recurrent_pass2_backward",
    },
    TraceStageMapping {
        canonical: "timing_cuda_backward_recurrent_pass1_ms_per_step",
        aliases: &[
            "timing_cuda_backward_recurrent_pass1_ms_per_step",
            "timing_cuda_backward_recurrent_pass1_ms_per_active_step",
        ],
        stage_name: "recurrent_pass1_backward",
    },
    TraceStageMapping {
        canonical: "output_ms",
        aliases: &[
            "output_ms",
            "timing_cuda_backward_output_ms_per_step",
            "timing_cuda_backward_forward_logits_ms_per_step",
        ],
        stage_name: "output_ce",
    },
    TraceStageMapping {
        canonical: "bank_update_ms_per_step",
        aliases: &[
            "bank_update_ms_per_step",
            "timing_cuda_bank_update_ms_per_step",
        ],
        stage_name: "optimizer_update",
    },
    TraceStageMapping {
        canonical: "non_bank_update_ms_per_step",
        aliases: &[
            "non_bank_update_ms_per_step",
            "timing_cuda_non_bank_update_ms_per_step",
        ],
        stage_name: "non_bank_update",
    },
];

const TRACE_COVERAGE_FIELDS: &[TraceCoverageField] = &[
    TraceCoverageField {
        canonical: "timing_measured_ms_per_step",
        aliases: &[
            "timing_measured_ms_per_step",
            "timing_train_step_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "backward_block_mlp_ms",
        aliases: &[
            "backward_block_mlp_ms",
            "timing_cuda_backward_block_mlp_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "backward_block_qkv_ms",
        aliases: &[
            "backward_block_qkv_ms",
            "timing_cuda_backward_block_qkv_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "backward_block_attention_sdpa_ms",
        aliases: &[
            "backward_block_attention_sdpa_ms",
            "timing_cuda_backward_block_attention_sdpa_ms_per_step",
            "timing_cuda_backward_block_attention_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "backward_block_attn_out_gate_xsa_ms",
        aliases: &[
            "backward_block_attn_out_gate_xsa_ms",
            "timing_cuda_backward_block_attn_out_gate_xsa_ms_per_step",
            "timing_cuda_backward_block_attention_xsa_accum_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "timing_cuda_backward_recurrent_pass2_ms_per_step",
        aliases: &[
            "timing_cuda_backward_recurrent_pass2_ms_per_step",
            "timing_cuda_backward_recurrent_pass2_ms_per_active_step",
        ],
    },
    TraceCoverageField {
        canonical: "timing_cuda_backward_recurrent_pass1_ms_per_step",
        aliases: &[
            "timing_cuda_backward_recurrent_pass1_ms_per_step",
            "timing_cuda_backward_recurrent_pass1_ms_per_active_step",
        ],
    },
    TraceCoverageField {
        canonical: "output_ms",
        aliases: &[
            "output_ms",
            "timing_cuda_backward_output_ms_per_step",
            "timing_cuda_backward_forward_logits_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "bank_update_ms_per_step",
        aliases: &[
            "bank_update_ms_per_step",
            "timing_cuda_bank_update_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "non_bank_update_ms_per_step",
        aliases: &[
            "non_bank_update_ms_per_step",
            "timing_cuda_non_bank_update_ms_per_step",
        ],
    },
    TraceCoverageField {
        canonical: "timing_recurrent_active_ms_per_step",
        aliases: &["timing_recurrent_active_ms_per_step"],
    },
    TraceCoverageField {
        canonical: "timing_recurrent_inactive_ms_per_step",
        aliases: &["timing_recurrent_inactive_ms_per_step"],
    },
];

fn apply_trace_to_stages(value: &serde_json::Value, stages: &mut [StageEstimate]) {
    for mapping in TRACE_STAGE_MAPPINGS {
        if let Some((trace_key, ms)) = trace_number_with_alias(value, mapping.aliases)
            && let Some(stage) = stages
                .iter_mut()
                .find(|stage| stage.name == mapping.stage_name)
        {
            stage.ms = ms;
            stage.source = if trace_key == mapping.canonical {
                format!("trace_field:{trace_key}")
            } else {
                format!("trace_field:{trace_key};canonical:{}", mapping.canonical)
            };
        }
    }
}

fn annotate_stage_percentages(stages: &mut [StageEstimate], total: f64) {
    let denominator = total.max(1e-9);
    for stage in stages {
        stage.pct_of_step = stage.ms.max(0.0) * 100.0 / denominator;
    }
}

fn trace_coverage_report(
    value: Option<&serde_json::Value>,
    stages: &[StageEstimate],
    total: f64,
) -> TraceCoverageReport {
    let has_total_timing = value
        .and_then(|trace| {
            trace_number_for_aliases(
                trace,
                &[
                    "timing_measured_ms_per_step",
                    "timing_train_step_ms_per_step",
                ],
            )
        })
        .is_some();
    let expected_fields = TRACE_COVERAGE_FIELDS
        .iter()
        .map(|field| field.canonical.to_string())
        .collect::<Vec<_>>();
    let mut present_fields = Vec::new();
    let mut missing_fields = Vec::new();
    for field in TRACE_COVERAGE_FIELDS {
        if value
            .and_then(|trace| trace_number_with_alias(trace, field.aliases))
            .is_some()
        {
            present_fields.push(field.canonical.to_string());
        } else {
            missing_fields.push(field.canonical.to_string());
        }
    }
    let mapped_stage_count = stages
        .iter()
        .filter(|stage| stage.source.starts_with("trace_field:"))
        .count();
    let mut nonzero_stage_fields = Vec::new();
    let mut zero_stage_fields = Vec::new();
    for mapping in TRACE_STAGE_MAPPINGS {
        if let Some(ms) = value.and_then(|trace| trace_number_for_aliases(trace, mapping.aliases)) {
            if ms.abs() > 1e-9 {
                nonzero_stage_fields.push(mapping.canonical.to_string());
            } else {
                zero_stage_fields.push(mapping.canonical.to_string());
            }
        }
    }
    nonzero_stage_fields.sort();
    nonzero_stage_fields.dedup();
    zero_stage_fields.sort();
    zero_stage_fields.dedup();
    let nonzero_mapped_stage_count = nonzero_stage_fields.len();
    let defaulted_stage_count = stages
        .iter()
        .filter(|stage| {
            stage.source == "shape_model_default" || stage.source == "trace_only_recurrent_stage"
        })
        .count();
    let traced_stage_ms = stages
        .iter()
        .filter(|stage| stage.source.starts_with("trace_field:"))
        .map(|stage| stage.ms)
        .sum::<f64>();
    let useful_stage_coverage_ratio =
        nonzero_mapped_stage_count as f64 / TRACE_STAGE_MAPPINGS.len().max(1) as f64;
    let traced_stage_ms_ratio = traced_stage_ms / total.max(1e-9);
    let signal_quality = trace_signal_quality(
        value.is_some(),
        has_total_timing,
        nonzero_mapped_stage_count,
        useful_stage_coverage_ratio,
        traced_stage_ms_ratio,
    );
    TraceCoverageReport {
        trace_supplied: value.is_some(),
        has_total_timing,
        expected_fields,
        coverage_ratio: present_fields.len() as f64 / TRACE_COVERAGE_FIELDS.len().max(1) as f64,
        present_fields,
        missing_fields,
        mapped_stage_count,
        nonzero_mapped_stage_count,
        defaulted_stage_count,
        useful_stage_coverage_ratio,
        nonzero_stage_fields,
        zero_stage_fields,
        traced_stage_ms,
        traced_stage_ms_ratio,
        signal_quality,
    }
}

fn prediction_source(trace_total_ms: Option<f64>, coverage: &TraceCoverageReport) -> String {
    if trace_total_ms.is_some() && coverage.nonzero_mapped_stage_count > 0 {
        "trace_calibrated".to_string()
    } else if trace_total_ms.is_some() {
        "trace_total_calibrated".to_string()
    } else if coverage.nonzero_mapped_stage_count > 0 {
        "trace_stage_calibrated".to_string()
    } else {
        "shape_model_default".to_string()
    }
}

fn trace_signal_quality(
    trace_supplied: bool,
    has_total_timing: bool,
    nonzero_mapped_stage_count: usize,
    useful_stage_coverage_ratio: f64,
    traced_stage_ms_ratio: f64,
) -> String {
    if !trace_supplied {
        "no_trace".to_string()
    } else if nonzero_mapped_stage_count == 0 {
        if has_total_timing {
            "total_only".to_string()
        } else {
            "metadata_only".to_string()
        }
    } else if useful_stage_coverage_ratio >= 0.5 && traced_stage_ms_ratio >= 0.35 {
        "stage_attributed".to_string()
    } else {
        "partial_stage_attribution".to_string()
    }
}

fn trace_number(value: &serde_json::Value, key: &str) -> Option<f64> {
    if let Some(number) = value.get(key).and_then(serde_json::Value::as_f64) {
        return Some(number);
    }
    value
        .get("metrics")
        .and_then(|metrics| metrics.get(key))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            value
                .get("run_timing")
                .and_then(|timing| timing.get(key))
                .and_then(serde_json::Value::as_f64)
        })
}

fn trace_number_for_aliases(value: &serde_json::Value, aliases: &[&str]) -> Option<f64> {
    trace_number_with_alias(value, aliases).map(|(_, number)| number)
}

fn trace_number_with_alias<'a>(
    value: &serde_json::Value,
    aliases: &'a [&'a str],
) -> Option<(&'a str, f64)> {
    aliases
        .iter()
        .find_map(|alias| trace_number(value, alias).map(|number| (*alias, number)))
}

fn stage_ms(stages: &[StageEstimate], name: &str) -> f64 {
    stages
        .iter()
        .find(|stage| stage.name == name)
        .map(|stage| stage.ms)
        .unwrap_or(0.0)
}

fn top_bottlenecks(stages: &[StageEstimate], limit: usize) -> Vec<BottleneckReport> {
    let mut ranked = stages.to_vec();
    ranked.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(idx, stage)| BottleneckReport {
            rank: idx + 1,
            stage: stage.name,
            ms: stage.ms,
            pct_of_step: stage.pct_of_step,
            source: stage.source,
        })
        .collect()
}

fn wind_tunnel_budget_status(
    spec: &RunSpec,
    expected_train_wall_seconds: f64,
    train_step_ms: f64,
    artifact_bytes: Option<usize>,
) -> Vec<BudgetStatusReport> {
    let train_budget = spec.train.max_wallclock_seconds.max(0.0) as f64;
    let train_status = if expected_train_wall_seconds <= train_budget {
        "pass"
    } else {
        "fail"
    };
    let eval_seconds = spec.eval.chunk_tokens as f64 / 64_000.0;
    let artifact_budget = spec.quant.target_artifact_bytes as f64;
    let artifact_status = artifact_bytes
        .map(|bytes| {
            if bytes as f64 <= artifact_budget {
                "pass"
            } else {
                "fail"
            }
        })
        .unwrap_or("unknown");
    vec![
        BudgetStatusReport {
            name: "train".to_string(),
            status: train_status.to_string(),
            estimate: Some(expected_train_wall_seconds),
            budget: Some(train_budget),
            unit: "seconds".to_string(),
            source: format!("total_iterations*{train_step_ms:.3}ms_step_estimate"),
        },
        BudgetStatusReport {
            name: "eval".to_string(),
            status: "unknown_budget".to_string(),
            estimate: Some(eval_seconds),
            budget: None,
            unit: "seconds".to_string(),
            source: "chunk_tokens/64000_local_proxy_no_run_spec_wall_budget".to_string(),
        },
        BudgetStatusReport {
            name: "artifact".to_string(),
            status: artifact_status.to_string(),
            estimate: artifact_bytes.map(|bytes| bytes as f64),
            budget: Some(artifact_budget),
            unit: "bytes".to_string(),
            source: "quant_layout_manifest_estimated_raw_weight_bytes".to_string(),
        },
    ]
}

fn budget_status<'a>(
    budgets: &'a [BudgetStatusReport],
    name: &str,
) -> Option<&'a BudgetStatusReport> {
    budgets.iter().find(|budget| budget.name == name)
}

fn wind_tunnel_risk_flags(
    spec: &RunSpec,
    coverage: &TraceCoverageReport,
    budgets: &[BudgetStatusReport],
    active_trace: Option<f64>,
    inactive_trace: Option<f64>,
    top_bottleneck: &str,
) -> Vec<String> {
    let mut flags = vec!["estimate_only_not_record_claim".to_string()];
    if !coverage.trace_supplied {
        flags.push("no_trace_calibration".to_string());
    } else if coverage.coverage_ratio < 0.5 {
        flags.push("low_trace_field_coverage".to_string());
    }
    if coverage.trace_supplied
        && coverage.has_total_timing
        && coverage.nonzero_mapped_stage_count == 0
    {
        flags.push("trace_total_only_no_stage_attribution".to_string());
    } else if coverage.trace_supplied
        && coverage.has_total_timing
        && coverage.useful_stage_coverage_ratio < 0.5
    {
        flags.push("trace_partial_stage_attribution".to_string());
    }
    if coverage.trace_supplied && !coverage.zero_stage_fields.is_empty() {
        flags.push("trace_contains_zero_stage_fields".to_string());
    }
    if coverage.trace_supplied && top_bottleneck == "unattributed_trace_overhead" {
        flags.push("trace_unattributed_overhead_dominates".to_string());
    }
    if spec.model.recurrence.enabled && (active_trace.is_none() || inactive_trace.is_none()) {
        flags.push("recurrent_active_inactive_split_defaulted".to_string());
    }
    if spec.model.recurrence.enabled
        && exact_recurrent_boundary_fusion_status(spec) != "enabled_exact_boundary_only"
    {
        flags.push("exact_recurrent_boundary_fusion_inactive".to_string());
    }
    if budget_status(budgets, "train")
        .map(|budget| budget.status == "fail")
        .unwrap_or(false)
    {
        flags.push("train_wall_budget_risk".to_string());
    }
    if budget_status(budgets, "artifact")
        .map(|budget| budget.status == "fail")
        .unwrap_or(false)
    {
        flags.push("artifact_budget_risk".to_string());
    }
    flags
}

fn recommendation_for_bottleneck(name: &str) -> String {
    match name {
        "optimizer_update" => "profile sharded Muon graph/update path before kernel rewrites",
        "qkv_backward" => "try exact recurrent QKV tail deferral or compact pack fusion",
        "mlp_backward" => "try BF16 MLP backward bridge removal and fused activation tail",
        "attention_backward" => "inspect recurrent replay and SDPA backward decomposition",
        "output_ce" => "only re-test fused CE if trace confirms output CE remains material",
        "unattributed_trace_overhead" => {
            "expand trace markers before attributing the remaining step overhead"
        }
        _ => "collect a stage-timing trace and update the local calibration",
    }
    .to_string()
}

fn recommendations_for_wind_tunnel(
    bottlenecks: &[BottleneckReport],
    risk_flags: &[String],
) -> Vec<String> {
    let mut recommendations = Vec::new();
    for bottleneck in bottlenecks {
        push_unique(
            &mut recommendations,
            recommendation_for_bottleneck(&bottleneck.stage),
        );
    }
    if risk_flags.iter().any(|flag| flag == "no_trace_calibration") {
        push_unique(
            &mut recommendations,
            "collect a stage-timing trace before treating rankings as calibrated".to_string(),
        );
    }
    if risk_flags
        .iter()
        .any(|flag| flag == "low_trace_field_coverage")
    {
        push_unique(
            &mut recommendations,
            "add missing trace markers for qkv/mlp/attention/output/optimizer stages".to_string(),
        );
    }
    if risk_flags
        .iter()
        .any(|flag| flag == "trace_total_only_no_stage_attribution")
    {
        push_unique(
            &mut recommendations,
            "use this trace for total-step calibration only; collect graph-disabled stage timing for attribution"
                .to_string(),
        );
    }
    if risk_flags
        .iter()
        .any(|flag| flag == "trace_contains_zero_stage_fields")
    {
        push_unique(
            &mut recommendations,
            "treat zero-valued stage fields as disabled instrumentation, not proof that the stage is free"
                .to_string(),
        );
    }
    if risk_flags
        .iter()
        .any(|flag| flag == "recurrent_active_inactive_split_defaulted")
    {
        push_unique(
            &mut recommendations,
            "record active and inactive recurrent timings in the next trace".to_string(),
        );
    }
    if risk_flags
        .iter()
        .any(|flag| flag == "train_wall_budget_risk")
    {
        push_unique(
            &mut recommendations,
            "reduce step time or total iterations before relying on train budget fit".to_string(),
        );
    }
    if risk_flags.iter().any(|flag| flag == "artifact_budget_risk") {
        push_unique(
            &mut recommendations,
            "tighten quantization or compression before artifact verification".to_string(),
        );
    }
    if recommendations.is_empty() {
        recommendations
            .push("collect a stage-timing trace and update the local calibration".to_string());
    }
    recommendations
}

fn wind_tunnel_what_if_scenarios(
    spec: &RunSpec,
    base_step_ms: f64,
    active_recurrent_ms: f64,
    optimizer_ms: f64,
    ce_ms: f64,
) -> Vec<WindTunnelWhatIfReport> {
    let mut scenarios = Vec::new();
    let recurrent_gap = if spec.model.recurrence.enabled {
        (active_recurrent_ms - base_step_ms).max(0.0)
    } else {
        0.0
    };
    let recurrent_cut = recurrent_gap
        .clamp(0.0, 17.0)
        .max(if spec.model.recurrence.enabled {
            8.0_f64.min(base_step_ms * 0.10)
        } else {
            0.0
        });
    if spec.model.recurrence.enabled && recurrent_cut > 0.0 {
        scenarios.push(wind_tunnel_scenario(
            spec,
            "exact_recurrent_replay_cut",
            "Exact recurrent replay fusion reduces active-window replay work without straight-through gradients.",
            "candidate_unimplemented_or_partially_implemented",
            recurrent_cut.min(base_step_ms - 1.0),
            base_step_ms,
            Some((active_recurrent_ms - recurrent_cut).max(1.0)),
            "Uses active/inactive recurrent gap and caps the estimate at the known ~8-17 ms blocker band.",
        ));
        scenarios.push(wind_tunnel_scenario(
            spec,
            "persistent_cta_recurrent_boundary",
            "Persistent-CTA recurrent boundary kernel removes launch/HBM replay around recurrent layers.",
            "speculative_requires_kernel_and_h100_validation",
            (recurrent_cut + 6.0).min(base_step_ms * 0.22).min(base_step_ms - 1.0),
            base_step_ms,
            Some((active_recurrent_ms - recurrent_cut - 6.0).max(1.0)),
            "Upper-bound planning scenario for the proposal kernel; not implementation evidence.",
        ));
    }
    if optimizer_ms > 1.0 {
        scenarios.push(wind_tunnel_scenario(
            spec,
            "optimizer_launch_collapse",
            "Collapse/capture optimizer launch groups after recurrent speed is under control.",
            "candidate_needs_trace_ab",
            optimizer_ms.min(6.0).min(base_step_ms - 1.0),
            base_step_ms,
            None,
            "Uses the optimizer stage estimate; capped because prior bank-update work already removed the largest wall-time issue.",
        ));
    }
    if ce_ms > 1.0 {
        scenarios.push(wind_tunnel_scenario(
            spec,
            "output_ce_retest",
            "Retest fused output projection/CE only if stage timing keeps CE in the top bottlenecks.",
            "low_priority_unless_trace_shows_ce_hot",
            ce_ms.min(4.0).min(base_step_ms - 1.0),
            base_step_ms,
            None,
            "Chunked BF16 CE is normally not the limiting stage; this is a bounded diagnostic scenario.",
        ));
    }
    scenarios.push(wind_tunnel_scenario(
        spec,
        "full_train_step_graph",
        "Extend graph capture from no-loss backward to the full train-step boundary.",
        "future_work_graph_safety_required",
        (base_step_ms * 0.04).clamp(1.0, 8.0).min(base_step_ms - 1.0),
        base_step_ms,
        None,
        "Planning estimate for host launch overhead; requires numerical parity and graph-safe updates.",
    ));
    scenarios
}

#[allow(clippy::too_many_arguments)]
fn wind_tunnel_scenario(
    spec: &RunSpec,
    name: &str,
    description: &str,
    status: &str,
    delta_ms: f64,
    base_step_ms: f64,
    active_recurrent_step_ms: Option<f64>,
    rationale: &str,
) -> WindTunnelWhatIfReport {
    let step = (base_step_ms - delta_ms.max(0.0)).max(1.0);
    let wall = spec.train.total_iterations as f64 * step / 1_000.0;
    let train_budget = spec.train.max_wallclock_seconds.max(0.0) as f64;
    WindTunnelWhatIfReport {
        name: name.to_string(),
        description: description.to_string(),
        status: status.to_string(),
        delta_ms_per_step: base_step_ms - step,
        train_step_ms_estimate: step,
        active_recurrent_step_ms_estimate: active_recurrent_step_ms,
        expected_train_steps_in_600s: (600_000.0 / step.max(1.0)).floor() as usize,
        expected_train_wall_seconds: wall,
        train_budget_status: if wall <= train_budget { "pass" } else { "fail" }.to_string(),
        rationale: rationale.to_string(),
        estimate_only: true,
    }
}

fn wind_tunnel_experiment_rankings(
    scenarios: &[WindTunnelWhatIfReport],
    coverage: &TraceCoverageReport,
    active_trace: Option<f64>,
    inactive_trace: Option<f64>,
) -> Vec<WindTunnelExperimentRankReport> {
    let mut ranked = scenarios
        .iter()
        .map(|scenario| {
            let mut confidence = coverage.coverage_ratio.clamp(0.20, 1.0);
            let trace_confidence = if !coverage.trace_supplied {
                confidence *= 0.60;
                "uncalibrated_shape_model"
            } else if coverage.coverage_ratio < 0.5 {
                confidence *= 0.75;
                "low_trace_coverage"
            } else {
                "trace_calibrated"
            };
            if scenario.scenario_needs_recurrent_split()
                && (active_trace.is_none() || inactive_trace.is_none())
            {
                confidence *= 0.70;
            }
            if scenario.status.contains("speculative") {
                confidence *= 0.65;
            }
            if scenario.status.contains("future_work") {
                confidence *= 0.80;
            }
            let budget_bonus = if scenario.train_budget_status == "pass" {
                5.0
            } else {
                0.0
            };
            let priority_score = scenario.delta_ms_per_step.max(0.0) * confidence + budget_bonus;
            WindTunnelExperimentRankReport {
                rank: 0,
                scenario: scenario.name.clone(),
                priority_score,
                delta_ms_per_step: scenario.delta_ms_per_step,
                train_step_ms_estimate: scenario.train_step_ms_estimate,
                train_budget_status: scenario.train_budget_status.clone(),
                trace_confidence: trace_confidence.to_string(),
                rationale: scenario.rationale.clone(),
                recommended_validation: wind_tunnel_validation_for_scenario(&scenario.name),
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.priority_score
            .partial_cmp(&a.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, entry) in ranked.iter_mut().enumerate() {
        entry.rank = idx + 1;
    }
    ranked
}

trait WindTunnelScenarioExt {
    fn scenario_needs_recurrent_split(&self) -> bool;
}

impl WindTunnelScenarioExt for WindTunnelWhatIfReport {
    fn scenario_needs_recurrent_split(&self) -> bool {
        self.name.contains("recurrent")
    }
}

fn wind_tunnel_validation_for_scenario(name: &str) -> String {
    match name {
        "exact_recurrent_replay_cut" => {
            "run exact #2135 record-shaped proxy with active recurrent timed steps and BF16/F32 bridge counters".to_string()
        }
        "persistent_cta_recurrent_boundary" => {
            "first run CUDA parity smoke for recurrent boundary kernel, then a short active-window H100 A/B".to_string()
        }
        "optimizer_launch_collapse" => {
            "run graph-on/off and sharded Muon update timing A/B under the exact recurrent profile".to_string()
        }
        "output_ce_retest" => {
            "run stage-timing A/B only if output_ce remains a top-three measured bottleneck".to_string()
        }
        "full_train_step_graph" => {
            "run numerical parity smoke before any wall-clock A/B because the graph boundary changes update ordering".to_string()
        }
        _ => "collect a calibrated trace and compare median/p90 against the previous profile".to_string(),
    }
}

fn trace_entry_run_key(entry: &TraceIndexEntry) -> String {
    if let Some(run_name) = entry.run_name.as_deref() {
        let mut key = format!("run:{}", sanitize_path_component(run_name));
        if let Some(total) = entry.trace_total_ms {
            key.push_str(&format!(":{total:.6}ms"));
        }
        return key;
    }
    let mut path = entry.trace_path.clone();
    for suffix in [
        ".finish_status.json",
        ".wind_tunnel.json",
        ".json",
        ".log",
        ".txt",
    ] {
        if path.ends_with(suffix) {
            path.truncate(path.len() - suffix.len());
            break;
        }
    }
    format!("path:{}", sanitize_path_component(&path))
}

fn trace_entry_selection_score(entry: &TraceIndexEntry) -> i32 {
    let mut score = if entry.parsed_trace { 10 } else { 0 };
    score += match entry.calibration_role.as_str() {
        "stage_calibration" => 100,
        "partial_stage_calibration" => 80,
        "total_calibration" => 60,
        "shape_model_only" => 20,
        _ => 0,
    };
    if entry.trace_total_ms.is_some() {
        score += 25;
    }
    if entry.active_recurrent_ms_per_step.is_some() {
        score += 12;
    }
    if entry.inactive_recurrent_ms_per_step.is_some() {
        score += 8;
    }
    if entry.canonical_caseops_dataset == Some(true) {
        score += 8;
    }
    if entry.host_batch_flatten_calls == Some(0) {
        score += 3;
    }
    if entry.f32_to_bf16_bridge_launches == Some(0) && entry.bf16_to_f32_bridge_launches == Some(0)
    {
        score += 3;
    }
    score + entry.nonzero_stage_fields.len() as i32 * 3
}

fn trace_entry_selection_reason(entry: &TraceIndexEntry) -> String {
    format!(
        "{}; signal={}; total={}; active_recurrent={}; nonzero_stage_fields={}",
        entry.calibration_role,
        entry.signal_quality,
        entry.trace_total_ms.is_some(),
        entry.active_recurrent_ms_per_step.is_some(),
        entry.nonzero_stage_fields.len()
    )
}

fn corpus_run_text(run: &WindTunnelCorpusRun) -> String {
    [
        run.run_key.as_str(),
        run.selected_trace_path.as_str(),
        run.run_name.as_deref().unwrap_or(""),
        run.mode.as_deref().unwrap_or(""),
        run.record_profile.as_deref().unwrap_or(""),
        run.recurrent_backward_profile.as_deref().unwrap_or(""),
        run.cuda_graph_profile.as_deref().unwrap_or(""),
    ]
    .join(" ")
    .to_ascii_lowercase()
}

fn corpus_run_is_exact_2135(run: &WindTunnelCorpusRun) -> bool {
    let text = corpus_run_text(run);
    !corpus_run_is_allst(run)
        && (run
            .tags
            .iter()
            .any(|tag| tag == "frontier_2135" || tag == "exact_2135")
            || text.contains("2135")
            || text.contains("frontier2135"))
}

fn corpus_run_is_allst(run: &WindTunnelCorpusRun) -> bool {
    let text = corpus_run_text(run);
    text.contains("allst") || text.contains("all_st") || text.contains("hybridst")
}

fn calibrated_metric(name: &str, samples: Vec<(String, f64)>) -> Option<CalibratedMetricReport> {
    if samples.is_empty() {
        return None;
    }
    let mut values = samples
        .iter()
        .map(|(_, value)| *value)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sample_count = values.len();
    let best_ms = values[0];
    let median_ms = percentile_sorted(&values, 0.50);
    let p90_ms = percentile_sorted(&values, 0.90);
    let source_run_keys = samples
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    Some(CalibratedMetricReport {
        name: name.to_string(),
        estimate_ms: median_ms,
        best_ms,
        median_ms,
        p90_ms,
        sample_count,
        confidence: confidence_for_sample_count(sample_count).to_string(),
        source_run_keys,
    })
}

fn percentile_sorted(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let idx = ((values.len() - 1) as f64 * pct.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn confidence_for_sample_count(sample_count: usize) -> &'static str {
    match sample_count {
        0 => "none",
        1 => "low_single_trace",
        2..=4 => "medium_multi_trace",
        _ => "high_multi_trace",
    }
}

fn calibrated_stage_metrics(runs: &[WindTunnelCorpusRun]) -> Vec<CalibratedStageMetricReport> {
    let mut by_stage: BTreeMap<String, Vec<(String, f64, String)>> = BTreeMap::new();
    for run in runs {
        let body = fs::read_to_string(&run.selected_trace_path).unwrap_or_default();
        let Some(trace) = parse_trace_body(&body) else {
            continue;
        };
        for mapping in TRACE_STAGE_MAPPINGS {
            if let Some((field, ms)) = trace_number_with_alias(&trace, mapping.aliases) {
                if ms.abs() <= 1e-9 {
                    continue;
                }
                by_stage
                    .entry(mapping.stage_name.to_string())
                    .or_default()
                    .push((run.run_key.clone(), ms, field.to_string()));
            }
        }
    }
    by_stage
        .into_iter()
        .filter_map(|(stage, samples)| {
            let metric = calibrated_metric(
                &stage,
                samples
                    .iter()
                    .map(|(run_key, value, _)| (run_key.clone(), *value))
                    .collect(),
            )?;
            let mut source_fields = samples
                .into_iter()
                .map(|(_, _, field)| field)
                .collect::<Vec<_>>();
            source_fields.sort();
            source_fields.dedup();
            Some(CalibratedStageMetricReport {
                stage,
                estimate_ms: metric.median_ms,
                sample_count: metric.sample_count,
                confidence: metric.confidence,
                source_fields,
            })
        })
        .collect()
}

fn calibration_confidence_summary(
    exact_total: Option<&CalibratedMetricReport>,
    active: Option<&CalibratedMetricReport>,
    stage_runs: usize,
) -> String {
    let total = exact_total
        .map(|metric| metric.confidence.as_str())
        .unwrap_or("none");
    let active = active
        .map(|metric| metric.confidence.as_str())
        .unwrap_or("none");
    let stage = confidence_for_sample_count(stage_runs);
    format!("exact_total={total}; active_recurrent={active}; stage_attribution={stage}")
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn select_lite_backend(spec: &LiteSpec) -> PgResult<LiteBackendSelection> {
    match spec.backend.kind {
        LiteBackendKind::CpuReference => Ok(LiteBackendSelection {
            requested_backend: LiteBackendKind::CpuReference,
            execution_backend: LiteBackendKind::CpuReference,
            status: "cpu_reference_only".to_string(),
            accelerated: false,
            fallback_reason: None,
        }),
        LiteBackendKind::MetalApple => select_metal_lite_backend(spec.backend.allow_cpu_fallback),
        LiteBackendKind::MlxPrototype => select_unimplemented_lite_backend(
            LiteBackendKind::MlxPrototype,
            "mlx_prototype",
            spec.backend.allow_cpu_fallback,
            false,
            "PG-Lite MLX interop is not implemented in the Rust CLI",
        ),
    }
}

fn select_metal_lite_backend(allow_cpu_fallback: bool) -> PgResult<LiteBackendSelection> {
    if !cfg!(feature = "metal_apple") {
        return select_unimplemented_lite_backend(
            LiteBackendKind::MetalApple,
            "metal_apple",
            allow_cpu_fallback,
            false,
            "PG-Lite Metal executor is compiled behind the `metal_apple` Cargo feature",
        );
    }
    match metal_lite_runtime_probe() {
        Ok(_) => Ok(LiteBackendSelection {
            requested_backend: LiteBackendKind::MetalApple,
            execution_backend: LiteBackendKind::MetalApple,
            status: "metal_apple_runtime".to_string(),
            accelerated: true,
            fallback_reason: None,
        }),
        Err(err) if allow_cpu_fallback => Ok(LiteBackendSelection {
            requested_backend: LiteBackendKind::MetalApple,
            execution_backend: LiteBackendKind::CpuReference,
            status: "metal_apple_requested_cpu_reference_fallback".to_string(),
            accelerated: false,
            fallback_reason: Some(format!("runtime_probe_failed: {err}")),
        }),
        Err(err) => Err(PgError::InvalidOp(format!(
            "PG-Lite backend metal_apple is not executable in this build: {err}. Set [backend].allow_cpu_fallback = true for an explicitly labeled local fallback run."
        ))),
    }
}

#[cfg(feature = "metal_apple")]
fn metal_lite_runtime_probe() -> Result<String, String> {
    metal_lite::runtime_probe()
}

#[cfg(not(feature = "metal_apple"))]
fn metal_lite_runtime_probe() -> Result<String, String> {
    Err("`metal_apple` Cargo feature is disabled".to_string())
}

fn select_unimplemented_lite_backend(
    requested: LiteBackendKind,
    label: &str,
    allow_cpu_fallback: bool,
    feature_enabled: bool,
    unavailable_reason: &str,
) -> PgResult<LiteBackendSelection> {
    if allow_cpu_fallback {
        let feature_note = if feature_enabled {
            "feature_enabled_executor_unavailable"
        } else {
            "feature_not_enabled"
        };
        return Ok(LiteBackendSelection {
            requested_backend: requested,
            execution_backend: LiteBackendKind::CpuReference,
            status: format!("{label}_requested_cpu_reference_fallback"),
            accelerated: false,
            fallback_reason: Some(format!("{feature_note}: {unavailable_reason}")),
        });
    }
    Err(PgError::InvalidOp(format!(
        "PG-Lite backend {label} is not executable in this build. {unavailable_reason}. Set [backend].allow_cpu_fallback = true for an explicitly labeled local fallback run."
    )))
}

#[derive(Debug, Clone)]
struct LiteNgramResidual {
    bigram_counts: Vec<[u32; 256]>,
    residual_counts: Vec<[u16; 256]>,
    residual_weight: f64,
    context: usize,
    family: LiteModelFamily,
}

#[derive(Debug, Clone, Copy)]
struct LiteEvalStats {
    loss: f64,
    tokens: usize,
    score_first_tokens_scored: usize,
    score_first_update_count: usize,
}

impl LiteNgramResidual {
    fn train(bytes: &[u8], spec: &LiteSpec) -> Self {
        let mut bigram_counts = vec![[1u32; 256]; 257];
        let buckets = spec.model.residual_buckets.max(1);
        let mut residual_counts = vec![[1u16; 256]; buckets];
        let train_bytes = if spec.model.family == LiteModelFamily::ArtifactOnly {
            &[][..]
        } else {
            bytes
        };
        let mut prev = 256usize;
        for (idx, &byte) in train_bytes.iter().enumerate() {
            bigram_counts[prev][byte as usize] =
                bigram_counts[prev][byte as usize].saturating_add(1);
            let bucket = context_bucket(train_bytes, idx, spec.model.context, buckets);
            residual_counts[bucket][byte as usize] =
                residual_counts[bucket][byte as usize].saturating_add(1);
            prev = byte as usize;
        }
        let residual_weight = match spec.model.family {
            LiteModelFamily::ByteNgram | LiteModelFamily::ArtifactOnly => 0.0,
            LiteModelFamily::NgramResidual => spec.model.residual_weight.clamp(0.0, 0.95),
        };
        Self {
            bigram_counts,
            residual_counts,
            residual_weight,
            context: spec.model.context,
            family: spec.model.family,
        }
    }

    fn loss_on_bytes(&self, bytes: &[u8], update_after_score: bool) -> LiteEvalStats {
        let mut model = self.clone();
        let mut loss = 0.0;
        let mut prev = 256usize;
        let mut score_first_tokens_scored = 0usize;
        let mut score_first_update_count = 0usize;
        for (idx, &byte) in bytes.iter().enumerate() {
            let bucket = context_bucket(bytes, idx, model.context, model.residual_counts.len());
            let p = model.prob(prev, bucket, byte as usize).max(1e-12);
            loss += -p.ln();
            if update_after_score {
                score_first_tokens_scored += 1;
                model.bigram_counts[prev][byte as usize] =
                    model.bigram_counts[prev][byte as usize].saturating_add(1);
                model.residual_counts[bucket][byte as usize] =
                    model.residual_counts[bucket][byte as usize].saturating_add(1);
                score_first_update_count += 1;
            }
            prev = byte as usize;
        }
        LiteEvalStats {
            loss,
            tokens: bytes.len(),
            score_first_tokens_scored,
            score_first_update_count,
        }
    }

    fn loss_on_bytes_score_first(&self, bytes: &[u8]) -> LiteEvalStats {
        self.loss_on_bytes(bytes, true)
    }

    fn loss_on_documents(&self, docs: &[&[u8]], update_after_score: bool) -> LiteEvalStats {
        let mut model = self.clone();
        let mut loss = 0.0;
        let mut tokens = 0usize;
        let mut score_first_tokens_scored = 0usize;
        let mut score_first_update_count = 0usize;
        for doc in docs {
            let mut prev = 256usize;
            for (idx, &byte) in doc.iter().enumerate() {
                let bucket = context_bucket(doc, idx, model.context, model.residual_counts.len());
                let p = model.prob(prev, bucket, byte as usize).max(1e-12);
                loss += -p.ln();
                tokens += 1;
                if update_after_score {
                    score_first_tokens_scored += 1;
                    model.bigram_counts[prev][byte as usize] =
                        model.bigram_counts[prev][byte as usize].saturating_add(1);
                    model.residual_counts[bucket][byte as usize] =
                        model.residual_counts[bucket][byte as usize].saturating_add(1);
                    score_first_update_count += 1;
                }
                prev = byte as usize;
            }
        }
        LiteEvalStats {
            loss,
            tokens,
            score_first_tokens_scored,
            score_first_update_count,
        }
    }

    fn prob(&self, prev: usize, bucket: usize, byte: usize) -> f64 {
        let row = &self.bigram_counts[prev.min(256)];
        let row_sum: u64 = row.iter().map(|&v| v as u64).sum();
        let p_bigram = row[byte] as f64 / row_sum.max(1) as f64;
        if self.family != LiteModelFamily::NgramResidual {
            return p_bigram;
        }
        let residual = &self.residual_counts[bucket % self.residual_counts.len()];
        let residual_sum: u64 = residual.iter().map(|&v| v as u64).sum();
        let p_residual = residual[byte] as f64 / residual_sum.max(1) as f64;
        (1.0 - self.residual_weight) * p_bigram + self.residual_weight * p_residual
    }

    fn artifact_bytes_estimate(&self) -> usize {
        let bigram_nonzero = self
            .bigram_counts
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&count| count > 1)
            .count();
        let residual_nonzero = self
            .residual_counts
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&count| count > 1)
            .count();
        64 + bigram_nonzero * 5 + residual_nonzero * 4
    }

    fn memory_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.bigram_counts.len() * std::mem::size_of::<[u32; 256]>()
            + self.residual_counts.len() * std::mem::size_of::<[u16; 256]>()
    }

    fn artifact_fingerprint(&self) -> String {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.residual_weight.to_le_bytes());
        hasher.update(&(self.context as u64).to_le_bytes());
        hasher.update(&[match self.family {
            LiteModelFamily::ByteNgram => 0,
            LiteModelFamily::NgramResidual => 1,
            LiteModelFamily::ArtifactOnly => 2,
        }]);
        for row in &self.bigram_counts {
            for &count in row {
                hasher.update(&count.to_le_bytes());
            }
        }
        for row in &self.residual_counts {
            for &count in row {
                hasher.update(&count.to_le_bytes());
            }
        }
        format!("lite-artifact-crc32:{:08x}", hasher.finalize())
    }

    fn to_stored_artifact(&self, source_config_fingerprint: String) -> LiteStoredArtifact {
        let mut bigram_entries = Vec::new();
        for (row_idx, row) in self.bigram_counts.iter().enumerate() {
            for (byte, &count) in row.iter().enumerate() {
                if count > 1 {
                    bigram_entries.push([row_idx as u32, byte as u32, count]);
                }
            }
        }

        let mut residual_entries = Vec::new();
        for (row_idx, row) in self.residual_counts.iter().enumerate() {
            for (byte, &count) in row.iter().enumerate() {
                if count > 1 {
                    residual_entries.push([row_idx as u32, byte as u32, count as u32]);
                }
            }
        }

        LiteStoredArtifact {
            kind: "pg_lite_model_artifact".to_string(),
            version: 1,
            source_config_fingerprint,
            model_fingerprint: self.artifact_fingerprint(),
            model_family: self.family,
            context: self.context,
            residual_weight: self.residual_weight,
            bigram_rows: self.bigram_counts.len(),
            residual_buckets: self.residual_counts.len(),
            bigram_entries,
            residual_entries,
        }
    }

    fn from_stored_artifact(artifact: &LiteStoredArtifact) -> PgResult<Self> {
        if artifact.kind != "pg_lite_model_artifact" {
            return Err(PgError::DataFormat(format!(
                "unsupported PG-Lite artifact kind {}; expected pg_lite_model_artifact",
                artifact.kind
            )));
        }
        if artifact.version != 1 {
            return Err(PgError::DataFormat(format!(
                "unsupported PG-Lite artifact version {}; expected 1",
                artifact.version
            )));
        }
        if artifact.bigram_rows != 257 {
            return Err(PgError::DataFormat(format!(
                "invalid PG-Lite artifact bigram_rows {}; expected 257",
                artifact.bigram_rows
            )));
        }
        if artifact.residual_buckets == 0 {
            return Err(PgError::DataFormat(
                "invalid PG-Lite artifact with zero residual buckets".into(),
            ));
        }
        let mut bigram_counts = vec![[1u32; 256]; artifact.bigram_rows];
        for [row, byte, count] in &artifact.bigram_entries {
            let row = *row as usize;
            let byte = *byte as usize;
            if row >= bigram_counts.len() || byte >= 256 || *count == 0 {
                return Err(PgError::DataFormat(
                    "PG-Lite artifact contains an invalid bigram entry".into(),
                ));
            }
            bigram_counts[row][byte] = *count;
        }
        let mut residual_counts = vec![[1u16; 256]; artifact.residual_buckets];
        for [row, byte, count] in &artifact.residual_entries {
            let row = *row as usize;
            let byte = *byte as usize;
            if row >= residual_counts.len()
                || byte >= 256
                || *count == 0
                || *count > u16::MAX as u32
            {
                return Err(PgError::DataFormat(
                    "PG-Lite artifact contains an invalid residual entry".into(),
                ));
            }
            residual_counts[row][byte] = *count as u16;
        }
        let model = Self {
            bigram_counts,
            residual_counts,
            residual_weight: artifact.residual_weight.clamp(0.0, 0.95),
            context: artifact.context,
            family: artifact.model_family,
        };
        if model.artifact_fingerprint() != artifact.model_fingerprint {
            return Err(PgError::DataFormat(
                "PG-Lite artifact fingerprint does not match decoded model".into(),
            ));
        }
        Ok(model)
    }
}

const PGLITE_BIN_MAGIC: &[u8; 8] = b"PGLITEB1";

fn write_lite_stored_artifact_json(path: &Path, artifact: &LiteStoredArtifact) -> PgResult<usize> {
    let body = serde_json::to_string_pretty(artifact)
        .map_err(|err| PgError::DataFormat(format!("PG-Lite artifact encode failed: {err}")))?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let bytes = body.as_bytes();
    fs::write(path, bytes)?;
    Ok(bytes.len())
}

fn write_lite_stored_artifact_binary(
    path: &Path,
    artifact: &LiteStoredArtifact,
) -> PgResult<usize> {
    let bytes = encode_lite_stored_artifact_binary(artifact)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, &bytes)?;
    Ok(bytes.len())
}

pub fn load_lite_stored_artifact(path: &Path) -> PgResult<LiteStoredArtifact> {
    let bytes = fs::read(path)?;
    let artifact = if bytes.starts_with(PGLITE_BIN_MAGIC) {
        decode_lite_stored_artifact_binary(&bytes)?
    } else {
        let body = std::str::from_utf8(&bytes).map_err(|err| {
            PgError::DataFormat(format!(
                "invalid PG-Lite artifact bytes; not UTF-8 JSON: {err}"
            ))
        })?;
        serde_json::from_str(body)
            .map_err(|err| PgError::DataFormat(format!("invalid PG-Lite artifact JSON: {err}")))?
    };
    LiteNgramResidual::from_stored_artifact(&artifact)?;
    Ok(artifact)
}

fn encode_lite_stored_artifact_binary(artifact: &LiteStoredArtifact) -> PgResult<Vec<u8>> {
    let mut out = Vec::with_capacity(
        64 + artifact.bigram_entries.len() * 7 + artifact.residual_entries.len() * 7,
    );
    out.extend_from_slice(PGLITE_BIN_MAGIC);
    push_u32(&mut out, artifact.version);
    out.push(match artifact.model_family {
        LiteModelFamily::ByteNgram => 0,
        LiteModelFamily::NgramResidual => 1,
        LiteModelFamily::ArtifactOnly => 2,
    });
    push_u64(&mut out, artifact.context as u64);
    out.extend_from_slice(&artifact.residual_weight.to_le_bytes());
    push_u32(&mut out, artifact.bigram_rows as u32);
    push_u32(&mut out, artifact.residual_buckets as u32);
    push_short_string(&mut out, &artifact.source_config_fingerprint)?;
    push_short_string(&mut out, &artifact.model_fingerprint)?;
    push_u32(&mut out, artifact.bigram_entries.len() as u32);
    for [row, byte, count] in &artifact.bigram_entries {
        if *row > u16::MAX as u32 || *byte > u8::MAX as u32 {
            return Err(PgError::DataFormat(
                "PG-Lite bigram entry does not fit compact binary format".into(),
            ));
        }
        push_u16(&mut out, *row as u16);
        out.push(*byte as u8);
        push_u32(&mut out, *count);
    }
    push_u32(&mut out, artifact.residual_entries.len() as u32);
    for [row, byte, count] in &artifact.residual_entries {
        if *byte > u8::MAX as u32 || *count > u16::MAX as u32 {
            return Err(PgError::DataFormat(
                "PG-Lite residual entry does not fit compact binary format".into(),
            ));
        }
        push_u32(&mut out, *row);
        out.push(*byte as u8);
        push_u16(&mut out, *count as u16);
    }
    Ok(out)
}

fn decode_lite_stored_artifact_binary(bytes: &[u8]) -> PgResult<LiteStoredArtifact> {
    let mut cursor = BinaryCursor::new(bytes);
    cursor.expect_magic(PGLITE_BIN_MAGIC)?;
    let version = cursor.read_u32()?;
    let model_family = match cursor.read_u8()? {
        0 => LiteModelFamily::ByteNgram,
        1 => LiteModelFamily::NgramResidual,
        2 => LiteModelFamily::ArtifactOnly,
        raw => {
            return Err(PgError::DataFormat(format!(
                "invalid PG-Lite binary model family tag {raw}"
            )));
        }
    };
    let context = cursor.read_u64()? as usize;
    let residual_weight = cursor.read_f64()?;
    let bigram_rows = cursor.read_u32()? as usize;
    let residual_buckets = cursor.read_u32()? as usize;
    let source_config_fingerprint = cursor.read_short_string()?;
    let model_fingerprint = cursor.read_short_string()?;
    let bigram_len = cursor.read_u32()? as usize;
    let mut bigram_entries = Vec::with_capacity(bigram_len);
    for _ in 0..bigram_len {
        let row = cursor.read_u16()? as u32;
        let byte = cursor.read_u8()? as u32;
        let count = cursor.read_u32()?;
        bigram_entries.push([row, byte, count]);
    }
    let residual_len = cursor.read_u32()? as usize;
    let mut residual_entries = Vec::with_capacity(residual_len);
    for _ in 0..residual_len {
        let row = cursor.read_u32()?;
        let byte = cursor.read_u8()? as u32;
        let count = cursor.read_u16()? as u32;
        residual_entries.push([row, byte, count]);
    }
    cursor.finish()?;
    Ok(LiteStoredArtifact {
        kind: "pg_lite_model_artifact".to_string(),
        version,
        source_config_fingerprint,
        model_fingerprint,
        model_family,
        context,
        residual_weight,
        bigram_rows,
        residual_buckets,
        bigram_entries,
        residual_entries,
    })
}

fn push_short_string(out: &mut Vec<u8>, value: &str) -> PgResult<()> {
    let bytes = value.as_bytes();
    if bytes.len() > u16::MAX as usize {
        return Err(PgError::DataFormat(
            "PG-Lite binary string field is too long".into(),
        ));
    }
    push_u16(out, bytes.len() as u16);
    out.extend_from_slice(bytes);
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct BinaryCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BinaryCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self, magic: &[u8]) -> PgResult<()> {
        let raw = self.take(magic.len())?;
        if raw != magic {
            return Err(PgError::DataFormat(
                "invalid PG-Lite binary artifact magic".into(),
            ));
        }
        Ok(())
    }

    fn read_u8(&mut self) -> PgResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> PgResult<u16> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn read_u32(&mut self) -> PgResult<u32> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn read_u64(&mut self) -> PgResult<u64> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]))
    }

    fn read_f64(&mut self) -> PgResult<f64> {
        Ok(f64::from_le_bytes(self.read_u64()?.to_le_bytes()))
    }

    fn read_short_string(&mut self) -> PgResult<String> {
        let len = self.read_u16()? as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec())
            .map_err(|err| PgError::DataFormat(format!("invalid PG-Lite binary UTF-8: {err}")))
    }

    fn take(&mut self, len: usize) -> PgResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| PgError::DataFormat("PG-Lite binary cursor offset overflow".into()))?;
        if end > self.bytes.len() {
            return Err(PgError::DataFormat(
                "truncated PG-Lite binary artifact".into(),
            ));
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn finish(&self) -> PgResult<()> {
        if self.offset != self.bytes.len() {
            return Err(PgError::DataFormat(
                "PG-Lite binary artifact has trailing bytes".into(),
            ));
        }
        Ok(())
    }
}

fn validate_lite_spec(spec: &LiteSpec) -> PgResult<()> {
    if spec.data.format != "bytes" {
        return Err(PgError::InvalidOp(format!(
            "unsupported PG-Lite data format {}; expected bytes",
            spec.data.format
        )));
    }
    if spec.model.vocab != "byte" {
        return Err(PgError::InvalidOp(format!(
            "unsupported PG-Lite vocab {}; expected byte",
            spec.model.vocab
        )));
    }
    if spec.model.context == 0 {
        return Err(PgError::InvalidOp(
            "PG-Lite model.context must be greater than zero".into(),
        ));
    }
    if spec.model.residual_buckets == 0 {
        return Err(PgError::InvalidOp(
            "PG-Lite model.residual_buckets must be greater than zero".into(),
        ));
    }
    if spec.track == LiteTrack::ArtifactGolf && spec.model.family != LiteModelFamily::ArtifactOnly {
        return Err(PgError::InvalidOp(
            "PG-Lite artifact_golf requires model.family = \"artifact_only\"".into(),
        ));
    }
    if spec.track != LiteTrack::ArtifactGolf && spec.model.family == LiteModelFamily::ArtifactOnly {
        return Err(PgError::InvalidOp(
            "PG-Lite artifact_only model is only valid for artifact_golf".into(),
        ));
    }
    if spec.track != LiteTrack::ArtifactGolf && spec.model.artifact_path.is_some() {
        return Err(PgError::InvalidOp(
            "PG-Lite model.artifact_path is only valid for artifact_golf".into(),
        ));
    }
    Ok(())
}

fn context_bucket(bytes: &[u8], idx: usize, context: usize, buckets: usize) -> usize {
    let start = idx.saturating_sub(context.min(32));
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in &bytes[start..idx] {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    (hash as usize) % buckets.max(1)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("pg_local_{name}_{stamp}"))
    }

    fn tiny_run_spec(path: &Path) -> RunSpec {
        let mut spec = RunSpec::default();
        spec.name = "tiny_local_test".to_string();
        spec.model.num_layers = 1;
        spec.model.model_dim = 16;
        spec.model.num_heads = 2;
        spec.model.num_kv_heads = 1;
        spec.model.rope.dims = 4;
        spec.model.vocab_size = 64;
        spec.model.mlp_mult = 2.0;
        spec.model.xsa_last_n = 0;
        spec.model.value_embedding.enabled = false;
        spec.model.bigram.enabled = false;
        spec.model.caseops.byte_sidecar = false;
        spec.train.batch_tokens = 64;
        spec.train.seq_len = 16;
        spec.train.total_iterations = 2;
        spec.eval.caseops_byte_sidecar_pattern = None;
        spec.quant.lqer.enabled = false;
        spec.save(path).unwrap();
        spec
    }

    fn write_test_shard(path: &Path, values: &[u16]) {
        let mut bytes = Vec::new();
        let mut header = [0i32; 256];
        header[0] = 20240520;
        header[1] = 1;
        header[2] = values.len() as i32;
        for value in header {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn verify_options(spec: PathBuf) -> VerifyOptions {
        VerifyOptions {
            spec,
            artifact: None,
            artifact_audit: None,
            score_first_log: None,
            output: None,
        }
    }

    fn write_lite_config(dir: &Path) -> PathBuf {
        write_lite_config_for_track(dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000)
    }

    fn write_lite_config_for_track(
        dir: &Path,
        track: &str,
        family: &str,
        train_time_seconds: usize,
        eval_time_seconds: usize,
        memory_budget_bytes: usize,
    ) -> PathBuf {
        write_lite_named_config_for_track(
            dir,
            "lite.toml",
            track,
            family,
            train_time_seconds,
            eval_time_seconds,
            memory_budget_bytes,
        )
    }

    fn write_lite_named_config_for_track(
        dir: &Path,
        file_name: &str,
        track: &str,
        family: &str,
        train_time_seconds: usize,
        eval_time_seconds: usize,
        memory_budget_bytes: usize,
    ) -> PathBuf {
        let train = dir.join("train.txt");
        let val = dir.join("val.txt");
        fs::write(&train, b"abcabcabcabcabcabc").unwrap();
        fs::write(&val, b"abcabc").unwrap();
        let config = dir.join(file_name);
        fs::write(
            &config,
            format!(
                r#"
track = "{}"
artifact_budget_bytes = 1000000
train_time_seconds = {}
eval_time_seconds = {}
memory_budget_bytes = {}
score = "bpb"

[data]
train_path = "{}"
val_path = "{}"
format = "bytes"

[model]
family = "{}"
vocab = "byte"
context = 16
residual_buckets = 8
residual_weight = {}

[backend]
kind = "cpu_reference"
"#,
                track,
                train_time_seconds,
                eval_time_seconds,
                memory_budget_bytes,
                train.display(),
                val.display(),
                family,
                if family == "artifact_only" {
                    "0.0"
                } else {
                    "0.1"
                }
            ),
        )
        .unwrap();
        config
    }

    #[test]
    fn verify_passes_for_tiny_spec_without_artifact() {
        let path = temp_path("verify_spec.toml");
        tiny_run_spec(&path);
        let report = run_verify(verify_options(path)).unwrap();
        assert_eq!(report.kind, "pg_local_verify");
        assert!(report.bpb_smoke_ok);
        assert_eq!(report.eval_legality.status, "pass");
        assert!(report.eval_legality.score_before_update_order_ok);
        assert!(report.eval_legality.document_boundary_reset_ok);
        assert!(report.eval_legality.artifact_decode_eval_equivalence_ok);
        assert!(report.proposal_claims_ok);
    }

    #[test]
    fn verify_strict_loads_matching_artifact_manifest() {
        let dir = temp_path("verify_artifact");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let spec = tiny_run_spec(&spec_path);
        let artifact = dir.join("model.pgrs");
        let model = GptModel::new(spec.model.to_model_config());
        pg_quant::export::export_model_with_spec(&model, &spec.quant, "tiny_verify", &artifact)
            .unwrap();
        let mut options = verify_options(spec_path);
        options.artifact = Some(artifact);
        let report = run_verify(options).unwrap();
        assert!(report.artifact_budget_known);
        assert!(report.artifact_manifest_ok);
        assert_eq!(report.status, "pass");
    }

    #[test]
    fn verify_reports_mismatched_artifact_manifest() {
        let dir = temp_path("verify_bad_artifact");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let mut spec = tiny_run_spec(&spec_path);
        let artifact = dir.join("model.pgrs");
        let model = GptModel::new(spec.model.to_model_config());
        pg_quant::export::export_model_with_spec(&model, &spec.quant, "tiny_verify", &artifact)
            .unwrap();
        spec.quant.matrix_bits = 5;
        spec.save(&spec_path).unwrap();
        let mut options = verify_options(spec_path);
        options.artifact = Some(artifact);
        let report = run_verify(options).unwrap();
        assert!(!report.artifact_manifest_ok);
        assert_eq!(report.status, "fail");
    }

    #[test]
    fn verify_parses_caseops_sidecar_shards() {
        let dir = temp_path("sidecar");
        fs::create_dir_all(&dir).unwrap();
        let sidecar = dir.join("fineweb_val_bytes_000.bin");
        write_test_shard(&sidecar, &[1, 2, 3, 4]);
        let spec_path = dir.join("spec.toml");
        let mut spec = tiny_run_spec(&spec_path);
        spec.model.caseops.enabled = true;
        spec.model.caseops.byte_sidecar = true;
        spec.eval.caseops_byte_sidecar_pattern =
            Some(dir.join("fineweb_val_bytes_*.bin").to_string_lossy().into());
        spec.save(&spec_path).unwrap();
        let report = run_verify(verify_options(spec_path)).unwrap();
        assert!(report.caseops_sidecar_ok);
        assert!(report.data_preflight.sidecar_structural_ok);
        assert_eq!(report.data_preflight.sidecar_shards_found, 1);
        assert_eq!(report.data_preflight.sidecar_tokens_found, Some(4));
        assert_eq!(
            report.data_preflight.caseops_sidecar_pattern,
            Some(dir.join("fineweb_val_bytes_*.bin").to_string_lossy().into())
        );
    }

    #[test]
    fn verify_reports_canonical_caseops_data_requirements() {
        let dir = temp_path("canonical_data_preflight");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let mut spec = tiny_run_spec(&spec_path);
        spec.name = "frontier_2135_audit_target".to_string();
        spec.model.caseops.enabled = true;
        spec.model.caseops.byte_sidecar = true;
        spec.train.train_data_pattern = Some(dir.join("train_*.bin").to_string_lossy().into());
        spec.train.validation_data_pattern = Some(dir.join("val_*.bin").to_string_lossy().into());
        spec.eval.caseops_byte_sidecar_pattern =
            Some(dir.join("val_bytes_*.bin").to_string_lossy().into());
        spec.save(&spec_path).unwrap();

        let report = run_verify(verify_options(spec_path)).unwrap();

        assert!(!report.data_preflight.ready);
        assert!(report.data_preflight.canonical_target);
        assert_eq!(report.data_preflight.train_shards_found, 0);
        assert_eq!(report.data_preflight.train_shards_required, Some(80));
        assert_eq!(
            report.data_preflight.validation_tokens_required,
            Some(47_851_520)
        );
        assert_eq!(report.data_preflight.validation_docs_required, Some(50_000));
        assert!(!report.caseops_sidecar_ok);
        assert_eq!(report.status, "fail");
    }

    #[test]
    fn verify_accepts_artifact_audit_and_score_first_logs() {
        let dir = temp_path("verify_audit_logs");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let artifact_audit = dir.join("artifact_audit.json");
        fs::write(
            &artifact_audit,
            r#"{
                "event": "record_artifact_audit",
                "artifact_model_bytes": 10,
                "artifact_code_bytes": 5,
                "artifact_total_bytes": 15,
                "artifact_total_limit": 16000000,
                "artifact_budget_known": true,
                "artifact_budget_ok": true,
                "artifact_model_sha256": "model",
                "artifact_code_sha256": "code"
            }"#,
        )
        .unwrap();
        let score_first = dir.join("score_first.json");
        fs::write(
            &score_first,
            r#"{"score_first_legal":true,"validation_contamination":false}"#,
        )
        .unwrap();
        let mut options = verify_options(spec_path);
        options.artifact_audit = Some(artifact_audit);
        options.score_first_log = Some(score_first);
        let report = run_verify(options).unwrap();
        assert!(report.artifact_audit_ok);
        assert!(report.score_first_log_ok);
        assert_eq!(report.status, "pass");
    }

    #[test]
    fn dist_sim_matches_replicated_update() {
        let path = temp_path("dist_spec.toml");
        tiny_run_spec(&path);
        let report = run_dist_sim(DistSimOptions {
            spec: path,
            world_size: 4,
            steps: 2,
            seed: 7,
            output: None,
        })
        .unwrap();
        assert!(report.reduce_scatter_equivalent);
        assert_eq!(report.bank_reports.len(), 4);
        assert!(report.optimizer_parity_max_abs_diff <= report.tolerance);
    }

    #[test]
    fn artifact_lab_reports_multiple_quant_sweeps() {
        let path = temp_path("artifact_spec.toml");
        tiny_run_spec(&path);
        let report = run_artifact_lab(ArtifactLabOptions {
            spec: path,
            artifact: None,
            sweep: vec!["quant".to_string()],
            mini_train: None,
            mini_val: None,
            output: None,
        })
        .unwrap();
        assert_eq!(report.requested_sweeps, vec!["quant"]);
        assert_eq!(report.artifact_kind, "none");
        assert_eq!(report.sweeps.len(), 5);
        assert!(report.mixed_bit_allocations.len() >= 4);
        assert!(report.mini_bpb.is_none());
        assert!(
            report
                .sweeps
                .iter()
                .all(|sweep| sweep.sweep_kind == "quant")
        );
        assert!(
            report
                .quant_kernel_ids
                .iter()
                .any(|id| id.contains("pack_signed"))
        );
    }

    #[test]
    fn artifact_lab_recognizes_pg_lite_artifact() {
        let dir = temp_path("artifact_lab_pg_lite");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let lite_config =
            write_lite_config_for_track(&dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000);
        let lite = run_lite(LiteRunOptions {
            config: lite_config,
            output: dir.join("lite_out"),
        })
        .unwrap();

        let report = run_artifact_lab(ArtifactLabOptions {
            spec: spec_path,
            artifact: Some(PathBuf::from(&lite.artifact_path)),
            sweep: vec!["quant".to_string()],
            mini_train: None,
            mini_val: None,
            output: None,
        })
        .unwrap();

        assert_eq!(report.artifact_kind, "pg_lite");
        assert!(report.artifact_manifest_ok);
        assert!(report.artifact_budget_known);
        assert!(report.artifact_budget_ok);
        let lite_artifact = report.pg_lite_artifact.as_ref().unwrap();
        assert_eq!(lite_artifact.model_family, LiteModelFamily::NgramResidual);
        assert!(lite_artifact.bigram_entries > 0);
        assert!(lite_artifact.local_proxy_only);
    }

    #[test]
    fn artifact_lab_respects_lqer_and_compression_sweeps() {
        let path = temp_path("artifact_sweeps_spec.toml");
        tiny_run_spec(&path);
        let report = run_artifact_lab(ArtifactLabOptions {
            spec: path,
            artifact: None,
            sweep: vec!["lqer,compression".to_string()],
            mini_train: None,
            mini_val: None,
            output: None,
        })
        .unwrap();
        assert_eq!(report.requested_sweeps, vec!["compression", "lqer"]);
        assert!(report.sweeps.iter().any(|sweep| sweep.sweep_kind == "lqer"));
        assert!(
            report
                .sweeps
                .iter()
                .any(|sweep| sweep.sweep_kind == "compression")
        );
        assert!(
            report
                .sweeps
                .iter()
                .all(|sweep| sweep.sweep_kind != "quant")
        );
    }

    #[test]
    fn artifact_lab_reports_mixed_allocations_and_mini_bpb() {
        let dir = temp_path("artifact_mini_bpb");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let train = dir.join("train.txt");
        let val = dir.join("val.txt");
        fs::write(&train, b"abcabcabcabcabcabc").unwrap();
        fs::write(&val, b"abcabc").unwrap();

        let report = run_artifact_lab(ArtifactLabOptions {
            spec: spec_path,
            artifact: None,
            sweep: vec!["quant".to_string()],
            mini_train: Some(train.clone()),
            mini_val: Some(val.clone()),
            output: None,
        })
        .unwrap();

        assert!(report.mixed_bit_allocations.iter().any(|entry| {
            entry.name == "current_manifest" && entry.total_bytes_estimate > 0 && entry.proxy_only
        }));
        let mini = report.mini_bpb.as_ref().unwrap();
        assert_eq!(mini.train_path, train.display().to_string());
        assert_eq!(mini.val_path, val.display().to_string());
        assert_eq!(
            mini.score_scope,
            "artifact_lab_local_proxy_bpb_not_leaderboard"
        );
        assert!(mini.validation_bpb.is_finite());
        assert!(mini.proxy_only);
    }

    #[test]
    fn artifact_lab_requires_mini_train_and_val_together() {
        let dir = temp_path("artifact_mini_requires_pair");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let train = dir.join("train.txt");
        fs::write(&train, b"abcabc").unwrap();

        let err = run_artifact_lab(ArtifactLabOptions {
            spec: spec_path,
            artifact: None,
            sweep: vec!["quant".to_string()],
            mini_train: Some(train),
            mini_val: None,
            output: None,
        })
        .expect_err("mini BPB should require both files");
        assert!(err.to_string().contains("--mini-train"));
    }

    #[test]
    fn backend_check_reports_cpu_available() {
        let report = run_backend_check(BackendCheckOptions {
            backend: LiteBackendKind::CpuReference,
            output: None,
        })
        .unwrap();
        assert_eq!(report.status, "pass");
        assert!(report.executable_backend_available);
        assert!(report.rust_runtime_linked);
        assert!(report.kernel_contract_ok);
        assert!(report.expected_kernel_symbols.is_empty());
        assert_eq!(report.backend, LiteBackendKind::CpuReference);
    }

    #[test]
    fn backend_check_reports_metal_source_and_runtime_state() {
        let dir = temp_path("backend_check");
        fs::create_dir_all(&dir).unwrap();
        let output = dir.join("metal.json");
        let report = run_backend_check(BackendCheckOptions {
            backend: LiteBackendKind::MetalApple,
            output: Some(output.clone()),
        })
        .unwrap();
        assert_eq!(report.backend, LiteBackendKind::MetalApple);
        assert!(report.kernel_source_present);
        assert!(report.kernel_source_crc32.is_some());
        assert_eq!(
            report.expected_kernel_symbols,
            pg_lite_metal_expected_symbols()
        );
        assert_eq!(report.kernel_symbols_found, report.expected_kernel_symbols);
        assert_eq!(
            report.kernel_contracts.len(),
            report.expected_kernel_symbols.len()
        );
        assert!(report.kernel_contracts.iter().any(|contract| {
            contract.symbol == "pg_lite_ngram_loss_reduce_u16_presummed_kernel"
                && contract.optimization_note.contains("avoids materializing")
        }));
        let source = fs::read_to_string(pg_lite_metal_kernel_path()).unwrap();
        assert!(source.contains("PG_LITE_FNV_OFFSET = 0xcbf29ce484222325UL"));
        assert!(source.contains("PG_LITE_FNV_PRIME = 0x100000001b3UL"));
        assert!(source.contains("pg_lite_score_first_eval_u32_kernel"));
        assert!(report.kernel_contract_ok);
        assert_eq!(report.rust_runtime_linked, cfg!(feature = "metal_apple"));
        assert_eq!(report.executable_backend_available, report.status == "pass");
        if !report.metal_compiler_available {
            assert!(!report.metal_compile_smoke_attempted);
            assert!(!report.metal_compile_smoke_ok);
        }
        if cfg!(feature = "metal_apple") {
            assert!(
                report.status == "pass" || report.status == "fail",
                "runtime probe status must be explicit"
            );
        } else {
            assert_eq!(report.status, "fail");
        }
        assert!(output.exists());
    }

    #[test]
    fn wind_tunnel_emits_estimate_and_recommendation() {
        let path = temp_path("wind_spec.toml");
        tiny_run_spec(&path);
        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: None,
            output: None,
        })
        .unwrap();
        assert!(report.estimate_only);
        assert_eq!(report.prediction_source, "shape_model_default");
        assert!(report.train_step_ms_estimate > 0.0);
        assert!(report.expected_train_wall_seconds > 0.0);
        assert!(!report.next_recommended_experiment.is_empty());
        assert!(!report.recommendations.is_empty());
        assert!(
            report
                .risk_flags
                .iter()
                .any(|flag| flag == "estimate_only_not_record_claim")
        );
        assert!(
            report
                .operation_dag
                .iter()
                .any(|node| node.name == "output_projection_ce")
        );
        assert!(
            report
                .what_if_scenarios
                .iter()
                .any(|scenario| scenario.name == "optimizer_launch_collapse"
                    && scenario.estimate_only
                    && scenario.train_step_ms_estimate < report.train_step_ms_estimate)
        );
    }

    #[test]
    fn wind_tunnel_distinguishes_exact_boundary_from_persistent_cta() {
        let path = temp_path("wind_recurrent_spec.toml");
        let mut spec = tiny_run_spec(&path);
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 0;
        spec.model.recurrence.repeat_layers = 1;
        spec.runtime.recurrent_backward_profile = RecurrentBackwardProfile::ExactFused;
        spec.runtime.recurrent_fused_pass_boundary_backward = true;
        spec.save(&path).unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: None,
            output: None,
        })
        .unwrap();

        assert_eq!(report.recurrent_backward_profile, "exact_fused");
        assert_eq!(
            report.exact_recurrent_boundary_fusion,
            "enabled_exact_boundary_only"
        );
        assert_eq!(report.exact_recurrent_boundary_layers, vec![0]);
        assert_eq!(report.recurrent_split.exact_boundary_layers, vec![0]);
        assert_eq!(
            report.recurrent_split.exact_boundary_fusion,
            "enabled_exact_boundary_only"
        );
        assert_eq!(
            report.persistent_cta_block_backward,
            "not_implemented_unclaimed"
        );
        assert!(report.what_if_scenarios.iter().any(|scenario| {
            scenario.name == "exact_recurrent_replay_cut"
                && scenario.active_recurrent_step_ms_estimate.is_some()
                && scenario.status == "candidate_unimplemented_or_partially_implemented"
        }));
        assert!(report.what_if_scenarios.iter().any(|scenario| {
            scenario.name == "persistent_cta_recurrent_boundary"
                && scenario.status == "speculative_requires_kernel_and_h100_validation"
        }));
        assert!(report.operation_dag.iter().any(|node| {
            node.op_kind == "exact_recurrent_boundary_fusion"
                && node.status == "enabled_exact_boundary_only"
        }));
        assert!(report.operation_dag.iter().any(|node| {
            node.name == "persistent_cta_block_backward"
                && node.status == "not_implemented_unclaimed"
                && !node.record_relevant
        }));
    }

    #[test]
    fn wind_tunnel_uses_trace_total_when_present() {
        let path = temp_path("wind_trace_spec.toml");
        tiny_run_spec(&path);
        let trace = temp_path("wind_trace.json");
        fs::write(
            &trace,
            r#"{"timing_measured_ms_per_step":123.5,"bank_update_ms_per_step":9.0}"#,
        )
        .unwrap();
        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();
        assert_eq!(report.trace_total_ms, Some(123.5));
        assert_eq!(report.prediction_source, "trace_calibrated");
        assert_eq!(report.train_step_ms_estimate, 123.5);
        assert!(
            report
                .stage_estimates
                .iter()
                .any(|stage| stage.name == "unattributed_trace_overhead")
        );
    }

    #[test]
    fn wind_tunnel_parses_run_timing_json_log_records() {
        let path = temp_path("wind_trace_log_spec.toml");
        tiny_run_spec(&path);
        let trace = temp_path("wind_trace.log");
        fs::write(
            &trace,
            r#"status=running
run_timing_json={"timing_measured_ms_per_step":127.5,"timing_recurrent_active_ms_per_step":156.0,"timing_recurrent_inactive_ms_per_step":119.5,"timing_cuda_bank_update_ms_per_step":8.9,"timing_cuda_non_bank_update_ms_per_step":2.7}
"#,
        )
        .unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        assert_eq!(report.trace_total_ms, Some(127.5));
        assert_eq!(report.prediction_source, "trace_calibrated");
        assert_eq!(report.active_recurrent_trace_ms_per_step, Some(156.0));
        let bank = report
            .stage_estimates
            .iter()
            .find(|stage| stage.name == "optimizer_update")
            .unwrap();
        assert_eq!(bank.ms, 8.9);
        assert!(bank.source.contains("timing_cuda_bank_update_ms_per_step"));
        let non_bank = report
            .stage_estimates
            .iter()
            .find(|stage| stage.name == "non_bank_update")
            .unwrap();
        assert_eq!(non_bank.ms, 2.7);
    }

    #[test]
    fn wind_tunnel_parses_flat_timing_log_records() {
        let path = temp_path("wind_trace_flat_log_spec.toml");
        tiny_run_spec(&path);
        let trace = temp_path("wind_trace_flat.log");
        fs::write(
            &trace,
            r#"metrics:
timing_measured_ms_per_step=256.787
timing_cuda_backward_block_qkv_ms_per_step=21.5
timing_cuda_bank_update_ms_per_step=15.2
timing_cuda_non_bank_update_ms_per_step=2.1
"#,
        )
        .unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        assert_eq!(report.trace_total_ms, Some(256.787));
        assert_eq!(report.prediction_source, "trace_calibrated");
        let qkv = report
            .stage_estimates
            .iter()
            .find(|stage| stage.name == "qkv_backward")
            .unwrap();
        assert_eq!(qkv.ms, 21.5);
        assert!(
            qkv.source
                .contains("timing_cuda_backward_block_qkv_ms_per_step")
        );
    }

    #[test]
    fn wind_tunnel_uses_recurrent_active_trace_when_present() {
        let path = temp_path("wind_trace_recurrent_spec.toml");
        let mut spec = tiny_run_spec(&path);
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 0;
        spec.model.recurrence.repeat_layers = 1;
        spec.save(&path).unwrap();
        let trace = temp_path("wind_trace_recurrent.json");
        fs::write(
            &trace,
            r#"{
                "timing_measured_ms_per_step":130.0,
                "timing_recurrent_active_ms_per_step":148.0,
                "timing_recurrent_inactive_ms_per_step":121.0,
                "timing_cuda_backward_recurrent_pass2_ms_per_step":3.8,
                "timing_cuda_backward_recurrent_pass1_ms_per_step":3.7
            }"#,
        )
        .unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        assert_eq!(report.active_recurrent_trace_ms_per_step, Some(148.0));
        assert_eq!(report.inactive_recurrent_trace_ms_per_step, Some(121.0));
        assert_eq!(report.active_recurrent_step_ms_estimate, 148.0);
        assert_eq!(report.inactive_recurrent_step_ms_estimate, 121.0);
        assert_eq!(
            report.recurrent_split.active_source,
            "trace_field:timing_recurrent_active_ms_per_step"
        );
        assert_eq!(
            report.recurrent_split.inactive_source,
            "trace_field:timing_recurrent_inactive_ms_per_step"
        );
        assert!(report.stage_estimates.iter().any(|stage| {
            stage.name == "recurrent_pass2_backward"
                && (stage.ms - 3.8).abs() < f64::EPSILON
                && stage
                    .source
                    .contains("timing_cuda_backward_recurrent_pass2_ms_per_step")
        }));
        assert_eq!(
            report.experiment_rankings[0].scenario,
            "exact_recurrent_replay_cut"
        );
        assert_eq!(
            report.experiment_rankings[0].trace_confidence,
            "low_trace_coverage"
        );
        assert!(
            report.experiment_rankings[0]
                .recommended_validation
                .contains("active recurrent")
        );
        assert!(
            report
                .next_recommended_experiment
                .contains("exact_recurrent_replay_cut")
        );
    }

    #[test]
    fn wind_tunnel_maps_nested_trace_fields_and_reports_coverage() {
        let path = temp_path("wind_trace_mapping_spec.toml");
        tiny_run_spec(&path);
        let trace = temp_path("wind_trace_mapping.json");
        fs::write(
            &trace,
            r#"{
                "metrics": {
                    "backward_block_qkv_ms": 31.0,
                    "output_ms": 4.0
                },
                "run_timing": {
                    "timing_measured_ms_per_step": 80.0,
                    "bank_update_ms_per_step": 11.0
                }
            }"#,
        )
        .unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        assert_eq!(report.prediction_source, "trace_calibrated");
        assert_eq!(report.trace_total_ms, Some(80.0));
        assert!(report.trace_coverage.trace_supplied);
        assert!(
            report
                .trace_coverage
                .present_fields
                .iter()
                .any(|field| field == "backward_block_qkv_ms")
        );
        assert!(report.trace_coverage.coverage_ratio > 0.0);
        let qkv = report
            .stage_estimates
            .iter()
            .find(|stage| stage.name == "qkv_backward")
            .unwrap();
        assert_eq!(qkv.ms, 31.0);
        assert_eq!(qkv.source, "trace_field:backward_block_qkv_ms");
        assert!((qkv.pct_of_step - 38.75).abs() < 1e-9);
    }

    #[test]
    fn wind_tunnel_reports_top_three_bottlenecks_in_rank_order() {
        let path = temp_path("wind_top_bottlenecks_spec.toml");
        tiny_run_spec(&path);
        let trace = temp_path("wind_top_bottlenecks.json");
        fs::write(
            &trace,
            r#"{
                "timing_measured_ms_per_step":120.0,
                "backward_block_mlp_ms":45.0,
                "backward_block_qkv_ms":35.0,
                "bank_update_ms_per_step":25.0,
                "output_ms":5.0
            }"#,
        )
        .unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        let stages = report
            .top_bottlenecks
            .iter()
            .map(|bottleneck| bottleneck.stage.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            stages,
            vec!["mlp_backward", "qkv_backward", "optimizer_update"]
        );
        assert_eq!(report.top_bottleneck, "mlp_backward");
        assert_eq!(report.top_bottlenecks[0].rank, 1);
        assert!(report.recommendations[0].contains("MLP"));
    }

    #[test]
    fn wind_tunnel_reports_exact_recurrent_boundary_layers() {
        let path = temp_path("wind_exact_boundary_layers_spec.toml");
        let mut spec = tiny_run_spec(&path);
        spec.model.num_layers = 4;
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 1;
        spec.model.recurrence.repeat_layers = 2;
        spec.runtime.recurrent_backward_profile = RecurrentBackwardProfile::Full;
        spec.runtime.recurrent_fused_pass_boundary_backward = true;
        spec.save(&path).unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: None,
            output: None,
        })
        .unwrap();

        assert_eq!(
            report.exact_recurrent_boundary_fusion,
            "enabled_exact_boundary_only"
        );
        assert_eq!(report.exact_recurrent_boundary_layers, vec![1, 2]);
        assert_eq!(report.recurrent_split.exact_boundary_layers, vec![1, 2]);
        assert!(report.operation_dag.iter().any(|node| {
            node.name == "layer_1_recurrent_pass_boundary"
                && node.status == "enabled_exact_boundary_only"
                && node.record_relevant
        }));
    }

    #[test]
    fn wind_tunnel_reports_train_eval_artifact_budget_status() {
        let path = temp_path("wind_budget_status_spec.toml");
        let mut spec = tiny_run_spec(&path);
        spec.train.total_iterations = 2;
        spec.train.max_wallclock_seconds = 1.0;
        spec.quant.target_artifact_bytes = 1;
        spec.save(&path).unwrap();
        let trace = temp_path("wind_budget_status.json");
        fs::write(&trace, r#"{"timing_measured_ms_per_step":750.0}"#).unwrap();

        let report = run_wind_tunnel(WindTunnelOptions {
            spec: path,
            trace: Some(trace),
            output: None,
        })
        .unwrap();

        let train = report
            .budget_status
            .iter()
            .find(|budget| budget.name == "train")
            .unwrap();
        let eval = report
            .budget_status
            .iter()
            .find(|budget| budget.name == "eval")
            .unwrap();
        let artifact = report
            .budget_status
            .iter()
            .find(|budget| budget.name == "artifact")
            .unwrap();

        assert_eq!(train.status, "fail");
        assert_eq!(train.estimate, Some(1.5));
        assert_eq!(eval.status, "unknown_budget");
        assert_eq!(artifact.status, "fail");
        assert!(
            report
                .risk_flags
                .iter()
                .any(|flag| flag == "train_wall_budget_risk")
        );
        assert!(
            report
                .risk_flags
                .iter()
                .any(|flag| flag == "artifact_budget_risk")
        );
    }

    #[test]
    fn wind_tunnel_suite_summarizes_synthetic_traces() {
        let dir = temp_path("wind_suite");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let trace_dir = dir.join("traces");
        fs::create_dir_all(&trace_dir).unwrap();
        fs::write(
            trace_dir.join("short.json"),
            r#"{"timing_measured_ms_per_step":123.5,"bank_update_ms_per_step":9.0}"#,
        )
        .unwrap();
        fs::write(
            trace_dir.join("nested.json"),
            r#"{"metrics":{"timing_measured_ms_per_step":140.0,"output_ms":8.0}}"#,
        )
        .unwrap();
        fs::write(
            trace_dir.join("modal.log"),
            r#"timing_measured_ms_per_step=130.0
timing_cuda_bank_update_ms_per_step=10.0
"#,
        )
        .unwrap();
        let nested_trace_dir = trace_dir.join("nested");
        fs::create_dir_all(&nested_trace_dir).unwrap();
        fs::write(
            nested_trace_dir.join("deep.log"),
            r#"run_timing_json={"timing_measured_ms_per_step":125.0,"timing_cuda_bank_update_ms_per_step":7.5}"#,
        )
        .unwrap();

        let report = run_wind_tunnel_suite(WindTunnelSuiteOptions {
            spec: spec_path,
            trace_dir,
            output_dir: dir.join("out"),
        })
        .unwrap();

        assert_eq!(report.traces_found, 4);
        assert_eq!(report.traces_with_step_timing, 4);
        assert!(report.traces_with_stage_attribution >= 1);
        assert_eq!(report.status, "pass");
        assert!(report.mean_abs_baseline_step_error_ms.is_some());
        assert!(Path::new(&report.summary_json_path).exists());
        assert!(Path::new(&report.summary_markdown_path).exists());
        assert!(Path::new(&report.baseline_report_path).exists());
        assert!(
            report
                .reports
                .iter()
                .all(|entry| Path::new(&entry.report_path).exists())
        );
        assert!(
            report
                .reports
                .iter()
                .any(|entry| entry.prediction_source == "trace_calibrated")
        );
    }

    #[test]
    fn wind_tunnel_suite_handles_empty_trace_dir_as_failed_report() {
        let dir = temp_path("wind_suite_empty");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let trace_dir = dir.join("traces");
        fs::create_dir_all(&trace_dir).unwrap();

        let report = run_wind_tunnel_suite(WindTunnelSuiteOptions {
            spec: spec_path,
            trace_dir,
            output_dir: dir.join("out"),
        })
        .unwrap();

        assert_eq!(report.traces_found, 0);
        assert_eq!(report.traces_with_step_timing, 0);
        assert_eq!(report.status, "fail");
        assert!(Path::new(&report.summary_json_path).exists());
        assert!(Path::new(&report.summary_markdown_path).exists());
    }

    #[test]
    fn wind_tunnel_corpus_dedupes_finish_status_and_selects_rich_trace() {
        let dir = temp_path("wind_corpus_dedupe");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let trace_dir = dir.join("traces");
        fs::create_dir_all(&trace_dir).unwrap();
        fs::write(
            trace_dir.join("frontier_2135_case.json"),
            r#"{
                "run_name":"frontier_2135_case",
                "mode":"RecordShapedProxy",
                "timing_measured_ms_per_step":127.0,
                "timing_recurrent_active_ms_per_step":155.0,
                "timing_recurrent_inactive_ms_per_step":119.0,
                "backward_block_mlp_ms":20.0,
                "backward_block_qkv_ms":15.0,
                "backward_block_attention_sdpa_ms":10.0,
                "backward_block_attn_out_gate_xsa_ms":8.0,
                "output_ms":5.0,
                "bank_update_ms_per_step":6.0
            }"#,
        )
        .unwrap();
        fs::write(
            trace_dir.join("frontier_2135_case.finish_status.json"),
            r#"{"run_name":"frontier_2135_case","timing_measured_ms_per_step":127.0}"#,
        )
        .unwrap();

        let report = run_wind_tunnel_corpus(WindTunnelCorpusOptions {
            spec: spec_path,
            trace_dir,
            output: None,
        })
        .unwrap();

        assert_eq!(report.parseable_traces, 2);
        assert_eq!(report.deduped_runs, 1);
        assert_eq!(report.duplicate_source_files, 1);
        assert_eq!(report.timed_runs, 1);
        assert_eq!(report.stage_calibration_runs, 1);
        assert_eq!(report.best_exact_2135_ms, Some(127.0));
        let run = &report.runs[0];
        assert_eq!(run.source_paths.len(), 2);
        assert!(run.selected_trace_path.ends_with("frontier_2135_case.json"));
        assert!(run.selection_reason.contains("stage_calibration"));
        assert!(run.tags.iter().any(|tag| tag == "frontier_2135"));
    }

    #[test]
    fn wind_tunnel_calibration_and_scenario_use_deduped_corpus() {
        let dir = temp_path("wind_calibration");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let mut spec = tiny_run_spec(&spec_path);
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 0;
        spec.model.recurrence.repeat_layers = 1;
        spec.save(&spec_path).unwrap();
        let trace_dir = dir.join("traces");
        fs::create_dir_all(&trace_dir).unwrap();
        for (name, total, active, inactive) in [
            ("frontier_2135_a", 127.0, 155.0, 119.0),
            ("frontier_2135_b", 128.0, 156.0, 120.0),
        ] {
            fs::write(
                trace_dir.join(format!("{name}.json")),
                format!(
                    r#"{{
                        "run_name":"{name}",
                        "mode":"RecordShapedProxy",
                        "timing_measured_ms_per_step":{total},
                        "timing_recurrent_active_ms_per_step":{active},
                        "timing_recurrent_inactive_ms_per_step":{inactive},
                        "backward_block_qkv_ms":20.0,
                        "bank_update_ms_per_step":5.0
                    }}"#
                ),
            )
            .unwrap();
        }
        let corpus_path = dir.join("corpus.json");
        let calibration_path = dir.join("calibration.json");
        let scenario_path = dir.join("scenario.json");

        let corpus = run_wind_tunnel_corpus(WindTunnelCorpusOptions {
            spec: spec_path.clone(),
            trace_dir,
            output: Some(corpus_path.clone()),
        })
        .unwrap();
        assert_eq!(corpus.deduped_runs, 2);

        let calibration = run_wind_tunnel_calibrate(WindTunnelCalibrationOptions {
            spec: spec_path.clone(),
            trace_dir: None,
            corpus: Some(corpus_path.clone()),
            output: Some(calibration_path.clone()),
        })
        .unwrap();
        assert_eq!(calibration.status, "pass");
        assert_eq!(
            calibration
                .exact_2135_total_step_ms
                .as_ref()
                .unwrap()
                .sample_count,
            2
        );
        assert!(
            calibration
                .active_recurrent_ms
                .as_ref()
                .unwrap()
                .confidence
                .contains("medium")
        );

        let scenario = run_wind_tunnel_scenario(WindTunnelScenarioOptions {
            spec: spec_path,
            calibration: Some(calibration_path),
            trace: None,
            active_recurrent_replay_cut_ms: 10.0,
            bank_update_cut_ms: 0.0,
            graph_overhead_cut_ms: 0.0,
            target_step_ms: 120.0,
            output: Some(scenario_path.clone()),
        })
        .unwrap();
        assert!(scenario.clears_target);
        assert_eq!(scenario.predicted_step_ms, 117.0);
        assert!(Path::new(&scenario_path).exists());
    }

    #[test]
    fn wind_tunnel_reviewer_report_writes_handoff_packet() {
        let dir = temp_path("wind_reviewer_report");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        let mut spec = tiny_run_spec(&spec_path);
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 0;
        spec.model.recurrence.repeat_layers = 1;
        spec.save(&spec_path).unwrap();
        let trace_dir = dir.join("traces");
        fs::create_dir_all(&trace_dir).unwrap();
        for (name, total, active, inactive, canonical) in [
            ("frontier_2135_a", 127.0, 155.0, 119.0, true),
            ("frontier_2135_b", 128.0, 156.0, 120.0, true),
            ("frontier_2135_c", 129.0, 157.0, 121.0, false),
            ("diagnostic_allst", 117.0, 117.0, 117.0, false),
        ] {
            fs::write(
                trace_dir.join(format!("{name}.json")),
                format!(
                    r#"{{
                        "run_name":"{name}",
                        "mode":"RecordShapedProxy",
                        "canonical_caseops_dataset":{canonical},
                        "timing_measured_ms_per_step":{total},
                        "timing_recurrent_active_ms_per_step":{active},
                        "timing_recurrent_inactive_ms_per_step":{inactive},
                        "backward_block_qkv_ms":20.0,
                        "backward_block_mlp_ms":18.0,
                        "bank_update_ms_per_step":5.0
                    }}"#
                ),
            )
            .unwrap();
        }

        let output_dir = dir.join("out");
        let report = run_wind_tunnel_report(WindTunnelReviewerOptions {
            spec: spec_path,
            trace_dir,
            output_dir: output_dir.clone(),
            target_step_ms: 120.0,
            active_recurrent_replay_cut_ms: 10.0,
        })
        .unwrap();

        assert_eq!(report.status, "pass");
        assert_eq!(report.timed_runs, 4);
        assert_eq!(report.exact_2135_timed_runs, 3);
        assert_eq!(report.canonical_caseops_runs, 2);
        assert!(report.holdout_validation.mean_abs_error_ms.is_some());
        assert_eq!(
            report.holdout_validation.predictor_kind,
            "train_only_stage_residual_plus_recurrent_split"
        );
        assert_eq!(report.holdout_validation.samples.len(), 2);
        assert_eq!(
            report.holdout_validation.spearman_rank_correlation,
            Some(1.0)
        );
        assert!(
            report
                .holdout_validation
                .samples
                .iter()
                .all(|sample| sample.prediction_source.contains("active_recurrent_offset"))
        );
        assert!(report.scenario_clears_target);
        assert!(!report.top_stage_metrics.is_empty());
        for file in [
            "trace_index.json",
            "trace_corpus.json",
            "calibration_model.json",
            "scenario_exact_recurrent_replay.json",
            "suite/summary.json",
            "suite/summary.md",
            "report.json",
            "report.md",
        ] {
            assert!(output_dir.join(file).exists(), "missing {file}");
        }
        let markdown = fs::read_to_string(output_dir.join("report.md")).unwrap();
        assert!(markdown.contains("PG Wind Tunnel Reviewer Report"));
        assert!(markdown.contains("This report is local planning evidence"));
        assert!(markdown.contains("Spearman rank correlation"));
    }

    #[test]
    fn proof_bundle_writes_all_local_reports() {
        let dir = temp_path("proof_bundle");
        fs::create_dir_all(&dir).unwrap();
        let spec_path = dir.join("spec.toml");
        tiny_run_spec(&spec_path);
        let lite_config = write_lite_config(&dir);
        let output_dir = dir.join("out");

        let report = run_proof_bundle(ProofBundleOptions {
            spec: spec_path,
            artifact: None,
            trace: None,
            trace_dir: None,
            lite_config,
            lite_config_dir: None,
            output_dir: output_dir.clone(),
        })
        .unwrap();

        assert_eq!(report.status, "pass");
        for file in [
            "verify.json",
            "dist_sim.json",
            "artifact_lab.json",
            "wind_tunnel.json",
            "trace_corpus.json",
            "calibration_model.json",
            "scenario_exact_recurrent_replay.json",
            "wind_tunnel_reviewer/report.json",
            "wind_tunnel_reviewer/report.md",
            "pg_lite/report.json",
            "pg_lite/model.pglite.bin",
            "pg_lite/model.pglite.json",
            "pg_lite/artifact_manifest.json",
            "pg_lite_benchmark/summary.json",
            "pg_lite_benchmark/summary.md",
            "pg_lite_suite/summary.json",
            "pg_lite_suite/summary.md",
            "backend_cpu_reference.json",
            "backend_metal_apple.json",
            "proposal_features.json",
            "proof_bundle.json",
            "evidence_manifest.json",
            "README.md",
        ] {
            assert!(output_dir.join(file).exists(), "missing {file}");
        }
        assert_eq!(
            report.readme_path,
            output_dir.join("README.md").display().to_string()
        );
        let readme = fs::read_to_string(output_dir.join("README.md")).unwrap();
        assert!(readme.contains("PG-Local Proof Bundle"));
        assert!(readme.contains("Still Requires Remote Validation"));
        assert!(readme.contains("Record claim: `false`"));
        assert_eq!(
            report.evidence_manifest_path,
            output_dir
                .join("evidence_manifest.json")
                .display()
                .to_string()
        );
        let evidence: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(output_dir.join("evidence_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(evidence["record_claim"], false);
        assert_eq!(evidence["leaderboard_claim"], false);
        assert!(
            evidence["requires_remote_validation"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry["name"] == "full_record_train_eval_export")
        );
        assert_eq!(
            report.lite_suite_summary_path,
            output_dir
                .join("pg_lite_suite/summary.json")
                .display()
                .to_string()
        );
        assert_eq!(
            report.trace_corpus_path,
            output_dir.join("trace_corpus.json").display().to_string()
        );
        assert_eq!(
            report.calibration_model_path,
            output_dir
                .join("calibration_model.json")
                .display()
                .to_string()
        );
        assert_eq!(
            report.scenario_report_path,
            output_dir
                .join("scenario_exact_recurrent_replay.json")
                .display()
                .to_string()
        );
    }

    #[test]
    fn lite_run_scores_tiny_byte_golf() {
        let dir = temp_path("lite");
        fs::create_dir_all(&dir).unwrap();
        let train = dir.join("train.txt");
        let val = dir.join("val.txt");
        fs::write(&train, b"abcabcabcabcabcabc").unwrap();
        fs::write(&val, b"abcabc").unwrap();
        let config = dir.join("lite.toml");
        fs::write(
            &config,
            format!(
                r#"
track = "byte_golf"
artifact_budget_bytes = 1000000
train_time_seconds = 1
eval_time_seconds = 1
score = "bpb"

[data]
train_path = "{}"
val_path = "{}"
format = "bytes"

[model]
family = "ngram_residual"
vocab = "byte"
context = 16
residual_buckets = 8
residual_weight = 0.1

[backend]
kind = "cpu_reference"
"#,
                train.display(),
                val.display()
            ),
        )
        .unwrap();
        let report = run_lite(LiteRunOptions {
            config: config.clone(),
            output: dir.join("out"),
        })
        .unwrap();
        assert!(report.validation_bpb.is_finite());
        assert_eq!(
            report.validation_bpb_scope,
            "local_proxy_bpb_not_leaderboard"
        );
        assert!(report.score_first_legal);
        assert_eq!(report.score_first_update_count, 0);
        assert_eq!(report.memory_budget_status, "pass");
        assert!(dir.join("out").join("report.json").exists());
        assert!(dir.join("out").join("artifact_manifest.json").exists());
        assert!(dir.join("out").join("model.pglite.bin").exists());
        assert!(dir.join("out").join("model.pglite.json").exists());
        assert!(report.artifact_actual_bytes > 0);
        assert_eq!(
            report.artifact_actual_bytes,
            fs::metadata(dir.join("out").join("model.pglite.bin"))
                .unwrap()
                .len() as usize
        );
        let report_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("out").join("report.json")).unwrap())
                .unwrap();
        let manifest_json: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(dir.join("out").join("artifact_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            report_json["config_fingerprint"],
            manifest_json["config_fingerprint"]
        );
        assert_eq!(
            manifest_json["score_scope"],
            "local_proxy_bpb_not_leaderboard"
        );
        assert_eq!(manifest_json["artifact_path"], "model.pglite.bin");
        assert_eq!(
            manifest_json["artifact_actual_bytes"],
            report.artifact_actual_bytes
        );
        assert_eq!(manifest_json["outputs"][0]["kind"], "model_artifact_binary");
        assert_eq!(
            manifest_json["outputs"][1]["kind"],
            "model_artifact_debug_json"
        );
        assert_eq!(load_lite_spec(&config).unwrap().track, LiteTrack::ByteGolf);
        assert_eq!(report.status, "pass");
    }

    #[test]
    fn lite_init_creates_self_contained_benchmark_and_verify_artifact() {
        let dir = temp_path("lite_init");
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("corpus.txt");
        fs::write(
            &input,
            b"parameter golf lite corpus\nthis is a local byte benchmark\n",
        )
        .unwrap();
        let init_dir = dir.join("pg_lite_bench");

        let init = run_lite_init(LiteInitOptions {
            input: input.clone(),
            output_dir: init_dir.clone(),
            artifact_budget_bytes: 64_000,
            train_time_seconds: 10.0,
            eval_time_seconds: 5.0,
            memory_budget_bytes: Some(1_000_000),
            val_bytes: Some(12),
            val_fraction: 0.2,
            context: 32,
            residual_buckets: 16,
            residual_weight: 0.2,
            backend: LiteBackendKind::CpuReference,
        })
        .unwrap();

        assert_eq!(init.status, "pass");
        assert_eq!(init.input_bytes, 58);
        assert_eq!(init.val_bytes, 12);
        assert_eq!(init.train_bytes, 46);
        assert!(Path::new(&init.train_path).exists());
        assert!(Path::new(&init.val_path).exists());
        assert!(Path::new(&init.readme_path).exists());
        assert!(Path::new(&init.report_path).exists());
        assert!(
            init.configs
                .iter()
                .any(|config| config.name == "byte_golf_ngram_baseline")
        );
        assert!(
            init.configs
                .iter()
                .any(|config| config.name == "stream_golf_score_first")
        );
        assert!(
            init.configs
                .iter()
                .any(|config| config.name == "artifact_golf_empty")
        );

        let residual_config = init
            .configs
            .iter()
            .find(|config| config.name == "byte_golf_ngram_residual")
            .unwrap()
            .path
            .clone();
        let run = run_lite(LiteRunOptions {
            config: PathBuf::from(&residual_config),
            output: dir.join("run"),
        })
        .unwrap();
        assert_eq!(run.status, "pass");

        let verified = run_lite_verify_artifact(LiteVerifyArtifactOptions {
            artifact: PathBuf::from(&run.artifact_path),
            config: Some(PathBuf::from(residual_config)),
            val: None,
            output: Some(dir.join("verify_artifact.json")),
        })
        .unwrap();
        assert_eq!(verified.status, "pass");
        assert!(verified.decoded_ok);
        assert!(verified.model_reconstruct_ok);
        assert_eq!(verified.artifact_budget_ok, Some(true));
        assert_eq!(verified.source_config_matches_expected, Some(true));
        assert_eq!(verified.val_bytes, Some(12));
        assert!(verified.validation_bpb.unwrap().is_finite());
    }

    #[test]
    fn lite_verify_artifact_reports_config_mismatch() {
        let dir = temp_path("lite_verify_mismatch");
        fs::create_dir_all(&dir).unwrap();
        let residual_config =
            write_lite_config_for_track(&dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000);
        let run = run_lite(LiteRunOptions {
            config: residual_config,
            output: dir.join("run"),
        })
        .unwrap();
        let baseline_config =
            write_lite_config_for_track(&dir, "byte_golf", "byte_ngram", 1, 1, 4_000_000);

        let verified = run_lite_verify_artifact(LiteVerifyArtifactOptions {
            artifact: PathBuf::from(&run.artifact_path),
            config: Some(baseline_config),
            val: None,
            output: None,
        })
        .unwrap();
        assert_eq!(verified.status, "fail");
        assert_eq!(verified.source_config_matches_expected, Some(false));
        assert!(
            verified
                .notes
                .iter()
                .any(|note| note.contains("source_config_fingerprint"))
        );
    }

    #[test]
    fn lite_benchmark_runs_cpu_and_reports_metal_status() {
        let dir = temp_path("lite_benchmark");
        fs::create_dir_all(&dir).unwrap();
        let config =
            write_lite_config_for_track(&dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000);

        let report = run_lite_benchmark(LiteBenchmarkOptions {
            config,
            output_dir: dir.join("bench"),
            repeats: 2,
        })
        .unwrap();

        assert_eq!(report.status, "pass");
        assert_eq!(report.backends.len(), 2);
        let cpu = report
            .backends
            .iter()
            .find(|backend| backend.requested_backend == LiteBackendKind::CpuReference)
            .unwrap();
        assert_eq!(cpu.status, "pass");
        assert_eq!(cpu.run_count, 2);
        assert!(cpu.median_total_wall_seconds.is_some());
        let metal = report
            .backends
            .iter()
            .find(|backend| backend.requested_backend == LiteBackendKind::MetalApple)
            .unwrap();
        if cfg!(feature = "metal_apple") {
            assert!(
                metal.status == "pass" || metal.status == "unavailable",
                "Metal runtime status must be explicit"
            );
        } else {
            assert_eq!(metal.status, "unavailable");
        }
        assert!(Path::new(&report.summary_json_path).exists());
        assert!(Path::new(&report.summary_markdown_path).exists());
    }

    #[test]
    fn lite_artifact_golf_loads_saved_artifact_roundtrip() {
        let dir = temp_path("lite_artifact_roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let source_config =
            write_lite_config_for_track(&dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000);
        let source = run_lite(LiteRunOptions {
            config: source_config,
            output: dir.join("source"),
        })
        .unwrap();
        assert_eq!(source.status, "pass");
        let source_artifact = load_lite_stored_artifact(Path::new(&source.artifact_path)).unwrap();

        let train = dir.join("train.txt");
        let val = dir.join("val.txt");
        let artifact_config = dir.join("artifact.toml");
        fs::write(
            &artifact_config,
            format!(
                r#"
track = "artifact_golf"
artifact_budget_bytes = 1000000
train_time_seconds = 0
eval_time_seconds = 1
score = "bpb"

[data]
train_path = "{}"
val_path = "{}"
format = "bytes"

[model]
family = "artifact_only"
vocab = "byte"
context = 16
residual_buckets = 8
residual_weight = 0.0
artifact_path = "{}"

[backend]
kind = "cpu_reference"
"#,
                train.display(),
                val.display(),
                source.artifact_path
            ),
        )
        .unwrap();

        let loaded = run_lite(LiteRunOptions {
            config: artifact_config,
            output: dir.join("loaded"),
        })
        .unwrap();
        assert_eq!(loaded.track, LiteTrack::ArtifactGolf);
        assert_eq!(loaded.train_bytes_used, 0);
        assert_eq!(loaded.train_wall_seconds, 0.0);
        assert_eq!(loaded.model_family, LiteModelFamily::NgramResidual);
        assert!(loaded.validation_bpb.is_finite());
        let loaded_artifact = load_lite_stored_artifact(Path::new(&loaded.artifact_path)).unwrap();
        assert_eq!(
            source_artifact.model_fingerprint,
            loaded_artifact.model_fingerprint
        );
    }

    #[test]
    fn lite_runs_all_tracks_and_writes_stable_manifest_json() {
        for (track, family, train_budget) in [
            ("byte_golf", "ngram_residual", 1),
            ("artifact_golf", "artifact_only", 0),
            ("stream_golf", "ngram_residual", 1),
        ] {
            let dir = temp_path(track);
            fs::create_dir_all(&dir).unwrap();
            let config =
                write_lite_config_for_track(&dir, track, family, train_budget, 1, 4_000_000);

            let first = run_lite(LiteRunOptions {
                config: config.clone(),
                output: dir.join("out_a"),
            })
            .unwrap();
            let second = run_lite(LiteRunOptions {
                config,
                output: dir.join("out_b"),
            })
            .unwrap();

            assert_eq!(first.status, "pass", "{track} should pass");
            assert_eq!(second.status, "pass", "{track} should pass on repeat");
            assert_eq!(first.config_fingerprint, second.config_fingerprint);
            assert_eq!(first.track_behavior, second.track_behavior);
            assert_eq!(first.memory_budget_status, "pass");
            assert_eq!(second.memory_budget_status, "pass");
            if track == "artifact_golf" {
                assert_eq!(first.train_bytes_used, 0);
                assert_eq!(first.train_wall_seconds, 0.0);
            }
            if track == "stream_golf" {
                assert_eq!(first.track_behavior, "score_first_online_update");
                assert_eq!(first.score_first_tokens_scored, first.val_bytes);
                assert_eq!(first.score_first_update_count, first.val_bytes);
            } else {
                assert_eq!(first.score_first_tokens_scored, 0);
                assert_eq!(first.score_first_update_count, 0);
            }

            let manifest_a =
                fs::read_to_string(dir.join("out_a").join("artifact_manifest.json")).unwrap();
            let manifest_b =
                fs::read_to_string(dir.join("out_b").join("artifact_manifest.json")).unwrap();
            assert_eq!(manifest_a, manifest_b, "{track} manifest should be stable");
            let manifest_json: serde_json::Value = serde_json::from_str(&manifest_a).unwrap();
            assert_eq!(manifest_json["track"], track);
            assert_eq!(manifest_json["backend_status"], "cpu_reference_only");
            assert_eq!(manifest_json["backend_accelerated"], false);
            assert_eq!(manifest_json["execution_backend"], "cpu_reference");
            assert_eq!(manifest_json["outputs"][0]["kind"], "model_artifact_binary");
            assert_eq!(manifest_json["outputs"][0]["stable_json"], false);
            assert_eq!(
                manifest_json["outputs"][1]["kind"],
                "model_artifact_debug_json"
            );
            assert_eq!(manifest_json["outputs"][1]["stable_json"], true);
            assert_eq!(manifest_json["outputs"][2]["stable_json"], false);
            assert_eq!(manifest_json["outputs"][3]["stable_json"], true);
        }
    }

    #[test]
    fn lite_suite_runs_all_configs_and_writes_summary() {
        let dir = temp_path("lite_suite");
        fs::create_dir_all(&dir).unwrap();
        write_lite_named_config_for_track(
            &dir,
            "byte.toml",
            "byte_golf",
            "ngram_residual",
            1,
            1,
            4_000_000,
        );
        write_lite_named_config_for_track(
            &dir,
            "stream.toml",
            "stream_golf",
            "ngram_residual",
            1,
            1,
            4_000_000,
        );
        write_lite_named_config_for_track(
            &dir,
            "artifact.toml",
            "artifact_golf",
            "artifact_only",
            0,
            1,
            4_000_000,
        );

        let report = run_lite_suite(LiteSuiteOptions {
            config_dir: dir.clone(),
            output_dir: dir.join("suite_out"),
        })
        .unwrap();

        assert_eq!(report.configs_found, 3);
        assert_eq!(report.status, "pass");
        assert!(report.best_bpb.is_some());
        assert!(report.best_config.is_some());
        assert!(Path::new(&report.summary_json_path).exists());
        assert!(Path::new(&report.summary_markdown_path).exists());
        assert!(
            report
                .runs
                .iter()
                .any(|run| run.track == LiteTrack::StreamGolf
                    && run.score_first_update_count == run.score_first_tokens_scored)
        );
        for run in &report.runs {
            assert!(Path::new(&run.output_dir).join("report.json").exists());
            assert!(
                Path::new(&run.output_dir)
                    .join("artifact_manifest.json")
                    .exists()
            );
            assert!(Path::new(&run.output_dir).join("model.pglite.bin").exists());
            assert!(
                Path::new(&run.output_dir)
                    .join("model.pglite.json")
                    .exists()
            );
            assert!(run.artifact_actual_bytes > 0);
        }
    }

    #[test]
    fn lite_suite_empty_config_dir_is_failed_report() {
        let dir = temp_path("lite_suite_empty");
        fs::create_dir_all(&dir).unwrap();
        let report = run_lite_suite(LiteSuiteOptions {
            config_dir: dir.clone(),
            output_dir: dir.join("suite_out"),
        })
        .unwrap();
        assert_eq!(report.configs_found, 0);
        assert_eq!(report.status, "fail");
        assert!(report.best_bpb.is_none());
        assert!(Path::new(&report.summary_json_path).exists());
    }

    #[test]
    fn lite_bpb_is_average_loss_not_length_scaled() {
        let spec = LiteSpec::default();
        let model = LiteNgramResidual::train(b"abcabcabcabc", &spec);
        let short = model.loss_on_bytes(b"abc", false);
        let long = model.loss_on_bytes(b"abcabc", false);
        let bpb_short = compute_bpb(
            short.loss / short.tokens as f64,
            short.tokens as f64,
            short.tokens as f64,
        );
        let bpb_long = compute_bpb(
            long.loss / long.tokens as f64,
            long.tokens as f64,
            long.tokens as f64,
        );
        assert!(bpb_short.is_finite());
        assert!(bpb_long.is_finite());
        assert!(
            (bpb_long - bpb_short).abs() < 2.0,
            "BPB should stay in the same scale, short={bpb_short} long={bpb_long}"
        );
    }

    #[test]
    fn lite_fails_status_when_artifact_budget_is_missed() {
        let dir = temp_path("lite_budget");
        fs::create_dir_all(&dir).unwrap();
        let train = dir.join("train.txt");
        let val = dir.join("val.txt");
        fs::write(&train, b"abcdefghijklmnopqrstuvwxyz").unwrap();
        fs::write(&val, b"abc").unwrap();
        let config = dir.join("lite.toml");
        fs::write(
            &config,
            format!(
                r#"
track = "byte_golf"
artifact_budget_bytes = 1
train_time_seconds = 1
eval_time_seconds = 1
score = "bpb"

[data]
train_path = "{}"
val_path = "{}"
format = "bytes"

[model]
family = "byte_ngram"
vocab = "byte"
context = 16
residual_buckets = 8
residual_weight = 0.0

[backend]
kind = "cpu_reference"
"#,
                train.display(),
                val.display()
            ),
        )
        .unwrap();
        let report = run_lite(LiteRunOptions {
            config,
            output: dir.join("out"),
        })
        .unwrap();
        assert!(!report.artifact_budget_ok);
        assert_eq!(report.status, "fail");
    }

    #[test]
    fn lite_rejects_or_selects_metal_backend_without_fallback() {
        let spec = LiteSpec {
            backend: LiteBackendSpec {
                kind: LiteBackendKind::MetalApple,
                allow_cpu_fallback: false,
            },
            ..LiteSpec::default()
        };
        match select_lite_backend(&spec) {
            Ok(selection) => {
                assert_eq!(selection.requested_backend, LiteBackendKind::MetalApple);
                assert_eq!(selection.execution_backend, LiteBackendKind::MetalApple);
                assert!(selection.accelerated);
            }
            Err(err) => assert!(err.to_string().contains("metal_apple")),
        }
    }

    #[test]
    fn lite_backend_cpu_fallback_is_explicit_and_reported() {
        let dir = temp_path("lite_backend_fallback");
        fs::create_dir_all(&dir).unwrap();
        let config =
            write_lite_config_for_track(&dir, "byte_golf", "ngram_residual", 1, 1, 4_000_000);
        let mut spec = load_lite_spec(&config).unwrap();
        spec.backend.kind = LiteBackendKind::MetalApple;
        spec.backend.allow_cpu_fallback = true;
        fs::write(&config, toml::to_string_pretty(&spec).unwrap()).unwrap();

        let report = run_lite(LiteRunOptions {
            config,
            output: dir.join("out"),
        })
        .unwrap();

        assert_eq!(report.backend, LiteBackendKind::MetalApple);
        if report.execution_backend == LiteBackendKind::MetalApple {
            assert!(report.backend_accelerated);
            assert_eq!(report.backend_status, "metal_apple_runtime");
            assert!(report.backend_fallback_reason.is_none());
        } else {
            assert_eq!(report.execution_backend, LiteBackendKind::CpuReference);
            assert!(!report.backend_accelerated);
            assert_eq!(
                report.backend_status,
                "metal_apple_requested_cpu_reference_fallback"
            );
            assert!(report.backend_fallback_reason.is_some());
        }
        let manifest_json: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(dir.join("out").join("artifact_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest_json["backend"], "metal_apple");
        assert_eq!(
            manifest_json["execution_backend"],
            lite_backend_label(report.execution_backend)
        );
        assert_eq!(
            manifest_json["backend_accelerated"],
            report.backend_accelerated
        );
        if report.execution_backend == LiteBackendKind::CpuReference {
            assert!(
                manifest_json["backend_fallback_reason"]
                    .as_str()
                    .unwrap()
                    .contains("runtime")
                    || manifest_json["backend_fallback_reason"]
                        .as_str()
                        .unwrap()
                        .contains("feature")
            );
        }
    }

    #[test]
    fn lite_suite_surfaces_requested_and_execution_backend() {
        let dir = temp_path("lite_suite_backend");
        fs::create_dir_all(&dir).unwrap();
        let config = write_lite_named_config_for_track(
            &dir,
            "metal_fallback.toml",
            "byte_golf",
            "ngram_residual",
            1,
            1,
            4_000_000,
        );
        let mut spec = load_lite_spec(&config).unwrap();
        spec.backend.kind = LiteBackendKind::MetalApple;
        spec.backend.allow_cpu_fallback = true;
        fs::write(&config, toml::to_string_pretty(&spec).unwrap()).unwrap();

        let report = run_lite_suite(LiteSuiteOptions {
            config_dir: dir.clone(),
            output_dir: dir.join("suite_out"),
        })
        .unwrap();

        assert_eq!(report.status, "pass");
        assert_eq!(report.runs.len(), 1);
        assert_eq!(
            report.runs[0].requested_backend,
            LiteBackendKind::MetalApple
        );
        if report.runs[0].execution_backend == LiteBackendKind::MetalApple {
            assert!(report.runs[0].backend_accelerated);
        } else {
            assert_eq!(
                report.runs[0].execution_backend,
                LiteBackendKind::CpuReference
            );
            assert!(!report.runs[0].backend_accelerated);
        }
    }

    #[test]
    fn proposal_report_keeps_future_work_unclaimed() {
        let report = proposal_feature_report();
        assert_eq!(report.persistent_cta_block_backward, "not_implemented");
        if cfg!(feature = "metal_apple") {
            assert_eq!(
                report.pg_lite_metal_backend,
                "runtime_linked_requires_local_backend_check"
            );
        } else {
            assert_eq!(
                report.pg_lite_metal_backend,
                "source_boundary_checked_feature_gated"
            );
        }
        assert_eq!(
            report.xsa_inside_sdpa,
            "local_parity_tested_not_record_active"
        );
        assert_eq!(report.full_train_step_graph, "not_implemented");
    }

    #[test]
    fn strict_artifact_mismatch_is_reported() {
        let path = temp_path("missing_artifact_spec.toml");
        tiny_run_spec(&path);
        let missing = temp_path("missing.pgrs");
        let mut options = verify_options(path);
        options.artifact = Some(missing);
        let err = run_verify(options).expect_err("missing artifact should fail before reporting");
        assert!(err.to_string().contains("IO error") || err.to_string().contains("No such"));
    }

    #[test]
    fn score_first_stream_updates_after_scoring() {
        let spec = LiteSpec::default();
        let model = LiteNgramResidual::train(b"aaaaabbbbb", &spec);
        let plain = model.loss_on_bytes(b"abababab", false);
        let score_first = model.loss_on_bytes_score_first(b"abababab");
        assert!(score_first.loss.is_finite());
        assert_eq!(score_first.score_first_tokens_scored, 8);
        assert_eq!(score_first.score_first_update_count, 8);
        assert!(score_first.loss <= plain.loss + 1e-9);
    }

    #[cfg(feature = "metal_apple")]
    #[test]
    fn metal_lite_train_and_eval_match_cpu_reference_when_available() {
        let mut spec = LiteSpec::default();
        spec.track = LiteTrack::ByteGolf;
        spec.model.family = LiteModelFamily::NgramResidual;
        spec.model.context = 8;
        spec.model.residual_buckets = 16;
        spec.model.residual_weight = 0.25;
        let train = b"abracadabra abracadabra abracadabra";
        let val = b"cadabra abracadabra";
        let cpu = LiteNgramResidual::train(train, &spec);
        let cpu_eval = cpu.loss_on_bytes(val, false);
        let metal = match run_metal_lite_train_eval(train, val, &spec) {
            Ok(metal) => metal,
            Err(err) => {
                eprintln!("skipping Metal parity test because runtime is unavailable: {err}");
                return;
            }
        };
        assert_eq!(metal.model.bigram_counts, cpu.bigram_counts);
        assert_eq!(metal.model.residual_counts, cpu.residual_counts);
        assert_eq!(metal.eval.tokens, cpu_eval.tokens);
        assert!(
            (metal.eval.loss - cpu_eval.loss).abs() < 1e-3,
            "metal loss {} cpu loss {}",
            metal.eval.loss,
            cpu_eval.loss
        );
    }

    #[cfg(feature = "metal_apple")]
    #[test]
    fn metal_lite_score_first_matches_cpu_reference_when_available() {
        let mut spec = LiteSpec::default();
        spec.model.context = 8;
        spec.model.residual_buckets = 16;
        spec.model.residual_weight = 0.2;
        let model = LiteNgramResidual::train(b"aaaaabbbbbaaaaabbbbb", &spec);
        let val = b"abababababab";
        let cpu_eval = model.loss_on_bytes_score_first(val);
        let metal_eval = match run_metal_lite_score_first_eval(&model, val) {
            Ok(eval) => eval,
            Err(err) => {
                eprintln!(
                    "skipping Metal score-first parity test because runtime is unavailable: {err}"
                );
                return;
            }
        };
        assert_eq!(metal_eval.tokens, cpu_eval.tokens);
        assert_eq!(metal_eval.score_first_tokens_scored, val.len());
        assert_eq!(metal_eval.score_first_update_count, val.len());
        assert!(
            (metal_eval.loss - cpu_eval.loss).abs() < 1e-3,
            "metal loss {} cpu loss {}",
            metal_eval.loss,
            cpu_eval.loss
        );
    }

    #[test]
    fn bpb_byte_accounting_smoke_matches_byte_vocab() {
        let loss = 8.0 * std::f64::consts::LN_2;
        let bpb = compute_bpb(loss, 8.0, 8.0);
        assert!((bpb - 8.0).abs() < 1e-9);
    }
}
