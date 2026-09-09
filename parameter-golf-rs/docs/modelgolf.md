# ModelGolf Planner

ModelGolf is the constraint-native planning layer for the Rust Parameter Golf
stack. It treats a language-model artifact as a compiled object:

```text
model spec + hardware + memory + latency + quality + legality constraints
  -> artifact plan + cache plan + delta plan + train plan + kernel contracts
```

The implementation is local and deterministic. It does not claim final H100
timing, final BPB, or production kernel performance. The current math audit is
tracked in [`modelgolf_math_audit.md`](modelgolf_math_audit.md).

## Commands

```bash
cargo run -q -p pg-local -- modelgolf plan \
  --spec specs/frontier_2135_audit_target.toml \
  --hardware m2-air-24gb \
  --runtime llama.cpp-metal \
  --context 8192 \
  --memory-budget-bytes 24000000000 \
  --artifact-budget-bytes 16000000 \
  --quality-budget-ppl-pct 3 \
  --proof-artifact records/local/modelgolf_proof.pgrs \
  --quality-calibration records/local/modelgolf_quality_calibration.json \
  --release-evidence records/local/modelgolf_release_evidence.json \
  --output records/local/modelgolf.json
```

Section-specific reports use the same options:

```bash
cargo run -q -p pg-local -- modelgolf pack --spec specs/frontier_2135_audit_target.toml --proof-artifact records/local/modelgolf_pack.pgrs --quality-calibration records/local/modelgolf_quality_calibration.json
cargo run -q -p pg-local -- modelgolf pack-experiment --spec specs/frontier_2135_audit_target.toml --artifact-budget-bytes 16000000 --output records/local/modelgolf_pack_experiment.json
cargo run -q -p pg-local -- modelgolf pack-source-report --spec specs/frontier_2135_audit_target.toml --release-artifact records/local/modelgolf_pack.pgrs --evidence-source records/local/modelgolf_pack_eval_raw.json --validation-dataset-id heldout-caseops-v1 --validation-command "pg-local verify ..." --heldout-bpb 1.23 --baseline-bpb 1.22 --decode-tokens-per-second 85 --hardware m2-air-24gb --runtime llama.cpp-metal --output records/local/modelgolf_pack_eval.json
cargo run -q -p pg-local -- modelgolf lqer-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_lqer_eval_raw.json --calibration-dataset-id heldout-lqer-calibration-v1 --production-svd-validated true --calibrated-tensor-sensitivity true --equal-byte-bpb-delta -0.01 --hardware m2-air-24gb --runtime llama.cpp-metal --output records/local/modelgolf_lqer_eval.json
cargo run -q -p pg-local -- modelgolf cache-plan --spec specs/frontier_2135_audit_target.toml --context 131072 --batch 8
cargo run -q -p pg-local -- modelgolf cache-experiment --spec specs/frontier_2135_audit_target.toml --context 131072 --batch 8 --output records/local/modelgolf_cache_experiment.json
cargo run -q -p pg-local -- modelgolf cache-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_cache_eval_raw.json --context 131072 --batch 8 --memory-budget-bytes 24000000000 --backend-id metal-cachegolf-v1 --kernel-id cachegolf-fused-dequant-attention --long-context-dataset-id long-context-heldout-v1 --fused-runtime true --parity-pass true --long-context-bpb-delta-pct 1.0 --speedup-x 1.25 --hardware m2-air-24gb --runtime llama.cpp-metal --output records/local/modelgolf_cache_eval.json
cargo run -q -p pg-local -- modelgolf delta-plan --spec specs/frontier_2135_audit_target.toml --delta-budget-bytes 1048576
cargo run -q -p pg-local -- modelgolf delta-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_delta_eval_raw.json --delta-budget-bytes 1048576 --domain-dataset-id private-domain-v1 --trained-delta-bytes 982016 --legality-pass true --score-first-trace-or-review true --equal-byte-domain-bpb-delta -0.02 --hardware m2-air-24gb --runtime pg-local --output records/local/modelgolf_delta_eval.json
cargo run -q -p pg-local -- modelgolf train-plan --spec specs/frontier_2135_audit_target.toml
cargo run -q -p pg-local -- modelgolf train-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_train_eval_raw.json --training-run-id train-artifact-aware-v1 --backend-id cuda-train-v1 --gpu-backend-integrated true --proxy-calibrated true --post-export-bpb-delta-vs-posthoc -0.01 --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_train_eval.json
cargo run -q -p pg-local -- modelgolf resource-cost-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_resource_cost_raw.json --measurement-run-id train-artifact-aware-v1 --power-meter-id rack-pdu-h100x8-v1 --wall-time-seconds 590 --average-power-watts 5200 --energy-joules 3068000 --telemetry-validated true --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_resource_cost.json
cargo run -q -p pg-local -- modelgolf optimizer-comm-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_optimizer_comm_raw.json --distributed-backend-id nccl-sharded-muon-v1 --reduce-scatter-parity-pass true --all-gather-parity-pass true --optimizer-update-parity-pass true --nccl-trace-validated true --overlap-validated true --measured-comm-time-ms 8.5 --measured-step-time-ms 127.0 --communication-speedup-x 1.05 --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_optimizer_comm.json
cargo run -q -p pg-local -- modelgolf kernel-plan --spec specs/frontier_2135_audit_target.toml
cargo run -q -p pg-local -- modelgolf kernel-experiment --spec specs/frontier_2135_audit_target.toml --output records/local/modelgolf_kernel_experiment.json
cargo run -q -p pg-local -- modelgolf kernel-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_kernel_eval_raw.json --backend-id cuda-kernelforge-v1 --generated-kernel-ids exact-tiled-ce-fwd,exact-tiled-ce-bwd --generated-kernels true --parity-pass true --speedup-x 1.10 --memory-reduction-x 2.0 --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_kernel_eval.json
cargo run -q -p pg-local -- modelgolf wind-plan --spec specs/frontier_2135_audit_target.toml
cargo run -q -p pg-local -- modelgolf wind-experiment --spec specs/frontier_2135_audit_target.toml --output records/local/modelgolf_wind_experiment.json
cargo run -q -p pg-local -- modelgolf wind-source-report --spec specs/frontier_2135_audit_target.toml --evidence-source records/local/modelgolf_wind_eval_raw.json --trace-corpus-id profiler-traces-v1 --calibration-report-id wind-calibration-v1 --fresh-profiler-traces true --external-timing-validated true --holdout-spearman 0.80 --mean-abs-pct-error 10.0 --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_wind_eval.json
cargo run -q -p pg-local -- modelgolf scale-plan --spec specs/frontier_2135_audit_target.toml --hardware h100x8 --runtime pg-train-cuda --output records/local/modelgolf_scale.json
cargo run -q -p pg-local -- modelgolf release-check --spec specs/frontier_2135_audit_target.toml --release-evidence records/local/modelgolf_release_evidence.json --output records/local/modelgolf_release_check.json
```

`--quality-calibration` accepts either a raw array of points or an object with a
`points` array:

```json
{
  "points": [
    {
      "group": "tok_emb",
      "bits": 4,
      "lqer": false,
      "measured_quality_loss": 0.018
    },
    {
      "role": "mlp_down",
      "bits": 5,
      "lqer": true,
      "estimated_quality_loss": 0.010,
      "measured_quality_loss": 0.007
    }
  ]
}
```

Group points derive their baseline estimate from the current artifact IR. Role
points must include `estimated_quality_loss` because they are not tied to one
compiled tensor group. The report emits fit factors and before/after fit error;
release claims still need held-out BPB/perplexity.

`pack-source-report` is the Pack evidence collection surface for release
binding. It strict-reloads `--release-artifact`, computes artifact bytes and
SHA-256, validates finite positive held-out BPB/baseline/decode metrics,
recomputes `relative_bpb_increase_pct`, enforces `--quality-budget-ppl-pct`
when supplied, and emits a `modelgolf_release_source_report` with
`pillar: "pack"`. Every `*-source-report` also requires `--evidence-source`,
a non-empty `modelgolf_raw_evidence` JSON packet. The packet must have
`kind: "modelgolf_raw_evidence"`, the matching `pillar`, and a `claims` object
whose measured fields match the source-report claims; the command hashes it into
`claims.raw_evidence` only after semantic validation. The command binds
caller-supplied held-out measurements to the current spec fingerprint and
hardware/runtime labels; it does not manufacture the held-out dataset or make
local smoke BPB count as release evidence.

`cache-source-report` is the CacheGolf evidence collection surface for release
binding. It selects the KV policy for the current `--context`, `--batch`, and
`--memory-budget-bytes`, verifies the local dequant-attention parity and error
bound proof, requires caller-supplied `--fused-runtime true`,
`--parity-pass true`, finite `--long-context-bpb-delta-pct`, and
`--speedup-x >= 1.0`, enforces `--quality-budget-ppl-pct` when supplied, and
emits a `modelgolf_release_source_report` with `pillar: "cache"`. The command
binds measured fused-backend claims to the selected K/V bits, layout, block
size, context, batch, and hardware/runtime labels; it does not run the external
fused backend itself.

`lqer-source-report`, `delta-source-report`, `train-source-report`,
`resource-cost-source-report`, `optimizer-comm-source-report`,
`kernel-source-report`, and `wind-source-report`
are the remaining evidence
collection surfaces for release binding. They enforce the same thresholds as
`release-check`: production SVD/calibration and equal-byte quality for LQER,
selected delta names/bytes plus legality and domain BPB for DeltaGolf, backend
integration/proxy calibration/post-export comparison for TrainGolf, measured
wall-time/power/energy telemetry for the resource cost contract, distributed
optimizer parity/NCCL/overlap/timing for Optimizer/Comm, generated kernel
IDs/tile/parity/speed/memory for KernelForge, and fresh traces plus holdout
correlation/error for Wind Tunnel. These commands bind measured claims to the
current spec fingerprint and hardware/runtime labels; they do not run the
external training, backend, power-meter, legal review, or profiler workflows by
themselves. The LQER report also binds `selected_lqer_group_names` to the
current Pack plan so evidence cannot be silently reused across a different
selected LQER assignment, and it refuses release evidence when the current Pack
contract selected no LQER proof groups.

`--release-evidence` accepts measured release evidence. Missing fields fail
closed. The evidence must be bound to the current spec fingerprint,
hardware/runtime labels, source reports, selected cache policy, selected delta
names, generated kernel IDs, and final artifact digest. Example shape:

```json
{
  "kind": "modelgolf_release_evidence",
  "spec_name": "frontier_2135_audit_target",
  "spec_fingerprint": "<ExecutionPlan.variant_fingerprint>",
  "hardware": "m2-air-24gb",
  "runtime": "llama.cpp-metal",
  "evidence_id": "release-candidate-2026-06-19",
  "generated_at": "2026-06-19T00:00:00Z",
  "source_reports": [
    {
      "pillar": "resource_cost",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_resource_cost.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "pack",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_pack_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "cache",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_cache_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "lqer",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_lqer_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "delta",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_delta_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "train",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_train_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "optimizer_comm",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_optimizer_comm.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "kernel_forge",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_kernel_eval.json",
      "sha256": "sha256:<hex>"
    },
    {
      "pillar": "wind_tunnel",
      "kind": "modelgolf_release_source_report",
      "path": "modelgolf_wind_eval.json",
      "sha256": "sha256:<hex>"
    }
  ],
  "resource_cost": {
    "measurement_run_id": "artifact-aware-train-001",
    "power_meter_id": "rack-pdu-h100x8-v1",
    "wall_time_seconds": 590.0,
    "average_power_watts": 5200.0,
    "energy_joules": 3068000.0,
    "telemetry_validated": true,
    "wall_time_budget_seconds": 600.0,
    "energy_budget_joules": 3120000.0,
    "wall_time_budget_fit": true,
    "energy_budget_fit": true,
    "energy_consistency_error_pct": 0.0
  },
  "pack": {
    "artifact_path": "modelgolf_release.pgrs",
    "artifact_sha256": "sha256:<hex>",
    "validation_dataset_id": "heldout-caseops-v1",
    "validation_command": "pg-local verify --spec ... --artifact ...",
    "artifact_bytes": 15000000,
    "strict_reload_pass": true,
    "heldout_bpb": 1.23,
    "baseline_bpb": 1.22,
    "relative_bpb_increase_pct": 0.82,
    "decode_tokens_per_second": 85.0
  },
  "lqer": {
    "calibration_dataset_id": "lqer-calibration-v1",
    "selected_lqer_group_names": ["mlp_down_bank", "tok_emb"],
    "production_svd_validated": true,
    "calibrated_tensor_sensitivity": true,
    "equal_byte_bpb_delta": -0.01
  },
  "cache": {
    "backend_id": "metal-cachegolf-v1",
    "kernel_id": "cachegolf-fused-dequant-attention",
    "long_context_dataset_id": "long-context-heldout-v1",
    "k_bits": 4,
    "v_bits": 6,
    "block_size_tokens": 128,
    "layout": "paged",
    "context_tokens": 8192,
    "batch_sequences": 1,
    "fused_runtime": true,
    "parity_pass": true,
    "memory_budget_fit": true,
    "long_context_bpb_delta_pct": 1.0,
    "speedup_x": 1.25
  },
  "delta": {
    "domain_dataset_id": "private-domain-heldout-v1",
    "selected_delta_names": ["lora_qo_rank4", "output_low_rank_rank4"],
    "trained_delta_bytes": 1048576,
    "legality_pass": true,
    "score_first_trace_or_review": true,
    "equal_byte_domain_bpb_delta": -0.02
  },
  "train": {
    "training_run_id": "artifact-aware-train-001",
    "backend_id": "cuda-distributed-h100",
    "gpu_backend_integrated": true,
    "proxy_calibrated": true,
    "post_export_bpb_delta_vs_posthoc": -0.01
  },
  "optimizer_comm": {
    "distributed_backend_id": "nccl-sharded-muon-v1",
    "world_size": 8,
    "optimizer_sharded": true,
    "nccl_overlap_mode": "side_stream_bucketed",
    "sharded_total_wire_bytes_per_rank": 12345678,
    "owned_optimizer_state_reduction_x": 8.0,
    "reduce_scatter_parity_pass": true,
    "all_gather_parity_pass": true,
    "optimizer_update_parity_pass": true,
    "nccl_trace_validated": true,
    "overlap_validated": true,
    "measured_comm_time_ms": 8.5,
    "measured_step_time_ms": 127.0,
    "communication_speedup_x": 1.05
  },
  "kernel_forge": {
    "backend_id": "metal-kernelforge-v1",
    "generated_kernel_ids": ["exact_softcapped_tiled_ce"],
    "tile_t": 512,
    "generated_kernels": true,
    "parity_pass": true,
    "speedup_x": 1.10,
    "memory_reduction_x": 2.0
  },
  "wind_tunnel": {
    "trace_corpus_id": "fresh-profiler-traces-2026-06-19",
    "calibration_report_id": "wind-calibration-2026-06-19",
    "fresh_profiler_traces": true,
    "external_timing_validated": true,
    "holdout_spearman": 0.80,
    "mean_abs_pct_error": 10.0
  }
}
```

These fields must come from measured artifacts, validation runs, backend traces,
or formal reviews. Each `source_reports` entry resolves relative to the release
evidence file, must match its SHA-256 digest, and must point at JSON shaped as:

`modelgolf release-evidence --source-report-dir ...` is not a blind bundler. It
scans only valid `modelgolf_release_source_report` JSON files, requires every
release pillar, revalidates each source report's SHA-256/spec/runtime binding,
and revalidates each report's hashed `modelgolf_raw_evidence` packet before
emitting the assembled `modelgolf_release_evidence` JSON.

```json
{
  "kind": "modelgolf_release_source_report",
  "pillar": "pack",
  "spec_name": "frontier_2135_audit_target",
  "spec_fingerprint": "<ExecutionPlan.variant_fingerprint>",
  "hardware": "m2-air-24gb",
  "runtime": "llama.cpp-metal",
  "claims": {
    "raw_evidence": {
      "path": "records/local/pack_eval_raw.json",
      "sha256": "sha256:<hex>",
      "bytes": 12345,
      "semantic_kind": "modelgolf_raw_evidence",
      "validated_claim_count": 9
    },
    "strict_reload_pass": true,
    "artifact_sha256": "sha256:<hex>"
  }
}
```

The gate requires source reports for `resource_cost`, `pack`, `lqer`, `cache`,
`delta`, `train`, `optimizer_comm`, `kernel_forge`, and `wind_tunnel`; every summary claim above must
match the corresponding digested report claim. Each source report must also
point at a non-empty raw evidence JSON packet with matching SHA-256, byte count,
`kind`, `pillar`, and measured claim values. The release gate recomputes the
relative BPB increase from `heldout_bpb` and `baseline_bpb`, verifies the
artifact SHA-256, and rejects evidence that does not match the currently
selected cache policy, selected LQER groups, selected delta names,
optimizer/communication plan, kernel tile, resource-cost telemetry budget fit,
spec fingerprint, hardware, runtime, or raw evidence packet claims. Synthetic
values are useful only for testing the gate.
The `*-source-report` commands generate the digested source report shape
consumed by this gate for all nine release pillars: `resource_cost`, `pack`,
`lqer`, `cache`, `delta`, `train`, `optimizer_comm`, `kernel_forge`, and
`wind_tunnel`.

## Report Surface

`modelgolf plan` emits:

- `resource_contract`: hardware/runtime/memory/latency/quality/adaptation
  constraints plus the spec-derived training-time budget and a hardware-label
  nominal-power energy proxy. The energy fields are planner proxies, not
  measured wall-power release evidence; release requires the separate
  `resource_cost` source report.
- `cost_model`: constraint cost model over training time and energy. It reports
  train tokens per budget second, artifact bytes per budget second, parameter
  elements per proxy joule, optional latency-budget tokens/second, and the
  evidence boundary for the nominal-power heuristic.
- `artifact_ir`: typed tensor groups, roles, shapes, current quant bits, and
  local sensitivity surrogates.
- `pack`: ModelGolf Pack mixed-precision planner. This uses an exact sparse
  multiple-choice knapsack over integer byte estimates and ranks LQER candidates
  by predicted quality-per-byte. LQER candidate rows report both requested rank
  and effective rank `min(r, rows, cols)`, and quality correction is limited to
  the effective rank. The exporter selects `lqer.top_k` groups by rank-r
  captured residual energy per actual requested-rank artifact byte, with role
  weighting. Selected LQER options emit deterministic
  residual-correction proofs with rank-r error reduction and activation/logit
  bound reduction. Pack also emits `quality_comparison`, an equal-byte local
  proxy table comparing uniform Q4, best uniform no-LQER, exact mixed no-LQER,
  selected mixed+LQER, and a deterministic LQER control/fallback using
  synthetic residual and activation-bound metrics. When `--proof-artifact` is
  supplied, it also exports a
  deterministic local artifact, strict-reloads it against the `QuantSpec`, and
  smoke-scores finite pre/post-reload losses. The proof is a local format/reload
  check, not trained-artifact byte or BPB evidence. When `--quality-calibration`
  is supplied, Pack consumes measured quality-loss JSON points and scales
  group/role bit-loss estimates before running the exact DP. The
  `pack-experiment` subcommand runs the plan's concrete LQER++ experiment table
  on a bounded deterministic small model: uniform Q4, mixed Q4/Q6 no-LQER,
  mixed+LQER top-3, and deterministic random-LQER control, with actual
  export/reload bytes, local BPB proxy, decode-speed smoke timing, and selected
  LQER groups from artifact metadata.
- `cache`: CacheGolf KV-cache planner plus local packed K/V reference format.
  This enumerates K/V bits, block sizes, and paged/contiguous layouts using the
  attention perturbation bound; `pg-kernels` implements per-channel K,
  per-token V, contiguous token-major payloads, paged whole-block payloads with
  page-table bytes, and on-the-fly dequant-attention parity. The report
  includes a deterministic selected-policy proof with packed runtime bytes,
  reconstruction error, packed-vs-full attention error, dequant-attention
  parity, and bound coverage.
  It also emits residual-sketch, sink/recent eviction, and long-context memory
  scaling plans with explicit planner-only evidence boundaries. The
  `cache-experiment` subcommand runs the plan's concrete K/V bit-grid protocol:
  25 K-bit/V-bit rows, observed attention-output error, perturbation-bound
  coverage, explicit-dequant parity, a best-under-contract-budget selection,
  and long-context memory extrapolation for that measured grid row. The default
  contract is 35% of requested FP16 cache bytes when no explicit
  `--memory-budget-bytes` is supplied.
- `delta`: DeltaGolf byte-constrained allocation across LoRA, low-rank output
  correction, n-gram residuals, bias corrections, and KV adapters, with a
  local weighted low-rank Kronecker-curvature reference for adapter factors.
  The report includes selected low-rank proofs that materialize deterministic
  weighted-SVD deltas, compare weighted error against zero/lower-rank deltas,
  and record the local evidence boundary. It also emits
  `score_first_legality_audit`, a static metadata legality check for selected
  score-first/offline-trained artifact deltas, plus `domain_evaluation`, a
  deterministic equal-byte proxy table over no-delta, selected, best-single,
  low-rank-only, and static-control plans.
- `train`: TrainGolf artifact-aware training objective, CPU
  quantization-distance regularizer reference, `pg-model` gradient integration
  seam, opt-in `pg-train` CPU train-loop scheduling, local regularizer/export
  gap proof, and export-gap bound.
- `optimizer_comm`: sharded optimizer and communication compiler report. It
  records the requested distributed optimizer backend, world size, NCCL overlap
  mode, graph/shadow-refresh contracts, ring-collective byte estimates for
  replicated all-reduce versus sharded reduce-scatter/all-gather, and a
  deterministic local proof that reduce-scatter + shard-local update +
  all-gather equals replicated update for shard-separable optimizers. This is
  local algebraic evidence, not NCCL timing, overlap validation, or production
  distributed Muon parity.
- `kernel_forge`: KernelForge fusion contracts, exact tiled CE memory math,
  CPU full-logit parity reference, report-level tiled CE loss/`dH`/`dW` proof,
  and backend parity obligations. The `kernel-experiment` subcommand runs the
  plan's exact tiled CE experiment: full-logit reference vs tiled logit-free CE,
  loss/`dH`/`dW` parity, CE scratch bytes, scratch reduction, and local CPU
  timing. CE scratch accounting compares persistent `M x V` logits/grad-logits
  to tile logits plus row stats, excluding live hidden/lm-head inputs shared by
  both paths.
- `wind_tunnel`: Pareto summary tying the pack/cache/delta/kernel plans to
  estimated run value and required evidence. The reviewer packet also includes
  train-only holdout predictions with per-run error and Spearman rank
  correlation when timed traces are available. The `wind-experiment`
  subcommand implements the plan's local Experiment 5 harness: it generates 24
  ModelGolf candidate plans spanning Pack bits/LQER, CacheGolf K/V bits, and
  DeltaGolf byte budgets; ranks every candidate by a cheap planner score;
  export/reloads and smoke-scores the top 8 candidates; then reports
  Spearman rank correlation, mean absolute rank error, top-3 overlap, and
  candidate-level cheap-vs-full proxy metrics. The fuller proxy uses the
  bounded export/reload BPB delta for local quality evidence, but scores
  resource pressure from the candidate's deployment-scale artifact/cache byte
  estimates so tiny proof artifacts cannot make over-budget plans look good.
  The report marks
  `calibration_pass=false` and `local_wind_experiment_needs_calibration` when
  the local Spearman correlation is below 0.50. This is still local proxy
  evidence, not fresh profiler trace or full held-out runtime evidence.
- `scale_golf`: ScaleGolf track matrix for PG-Lite, ParameterGolf, DeltaGolf,
  LongContext, and larger-cluster ScaleGolf. Every track row uses the same
  resource contract, model IR, artifact compiler, runtime planner, evaluator,
  and cost model vocabulary, then lists the modules and measured evidence
  needed before that track can be claimed release-ready. This is a planning
  matrix, not measured large-cluster quality-per-dollar evidence.
- `release_readiness`: fail-closed checklist for release claims. It marks local
  planner gates separately from blocking external evidence such as held-out
  BPB/perplexity, fused backend timing, trained domain deltas, post-export
  quality comparisons, generated kernel parity/perf, and fresh profiler traces.
  The `release-check` subcommand emits only this object. When
  `--release-evidence` is supplied, the checklist validates the measured JSON
  and can mark individual blockers satisfied.

## Evidence Boundary

These reports are planning evidence. Before using a plan as a research or
product claim, collect the missing evidence listed in `platform_status`:

- final trained-artifact byte proof and held-out full validation
  BPB/perplexity for Pack/LQER plans
- large-matrix randomized/exact SVD validation and calibrated tensor
  sensitivity for production LQER++, plus measured equal-byte quality
  comparisons on pretrained small models and real validation data
- CUDA/Metal fused dequant-attention kernels, full long-context quality, and
  timing for CacheGolf
- measured full-update/curvature estimation, runtime score-first ordering
  traces or formal competition/legal review, trained domain deltas, and real
  equal-byte private/domain BPB improvement for DeltaGolf
- proxy calibration, GPU/backend integration, and post-export quality
  comparisons for artifact-aware training
- CUDA/Metal lowering, backend parity, and backend timing for generated
  KernelForge kernels
- fresh calibrated profiler traces and external full train/eval timing
  validation for Wind Tunnel; the local `wind-experiment` rank-correlation
  report is a pre-release screening harness only
- per-track measured quality, runtime, energy, or dollar evidence before making
  ScaleGolf cross-scale claims

The planner is intended to make no-op fields visible. Every module consumes and
emits typed metadata with bytes, quality, runtime, and legality status.
