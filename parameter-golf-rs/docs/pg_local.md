# PG-Local, PG-Lite, and Wind Tunnel

`pg-local` is the local proof surface for the Rust/CUDA Parameter Golf stack.
It does not claim leaderboard validity. It checks correctness, artifact
metadata, eval legality, distributed optimizer math, and local benchmark
behavior before another paid H100 run.

## Command Discovery

```bash
cargo run -q -p pg-local -- --help
cargo run -q -p pg-local -- help verify
cargo run -q -p pg-local -- help backend-check
cargo run -q -p pg-local -- help wind-tunnel
cargo run -q -p pg-local -- help wind-tunnel-suite
cargo run -q -p pg-local -- help trace-index
cargo run -q -p pg-local -- help modelgolf
cargo run -q -p pg-local -- help lite
cargo run -q -p pg-local -- help lite init
cargo run -q -p pg-local -- help lite suite
cargo run -q -p pg-local -- help lite benchmark
cargo run -q -p pg-local -- help lite verify-artifact
```

Reports are JSON. Commands with `--output <json>` write that file; without
`--output`, they print JSON to stdout. `lite run` always writes
`<output-dir>/report.json`; `lite suite` writes one directory per config plus
`summary.json` and `summary.md`. `proof-bundle` writes a directory of component
reports, backend readiness probes, PG-Lite suite output, and `proof_bundle.json`.
It also writes `evidence_manifest.json`, which is the reviewer-facing index of
component statuses, caveats, and remaining remote-validation requirements.
`README.md` is generated next to it as the human-readable handoff.

## Common Commands

```bash
cargo run -q -p pg-local -- verify \
  --spec specs/frontier_2135_audit_target.toml \
  --output records/local/verify.json

cargo run -q -p pg-local -- dist-sim \
  --spec specs/frontier_2135_audit_target.toml \
  --world-size 8 \
  --steps 3 \
  --seed 1 \
  --output records/local/dist_sim.json

cargo run -q -p pg-local -- artifact-lab \
  --spec specs/frontier_2135_audit_target.toml \
  --sweep quant,lqer,compression \
  --mini-train specs/pg_lite/sample_train.txt \
  --mini-val specs/pg_lite/sample_val.txt \
  --output records/local/artifact_lab.json

cargo run -q -p pg-local -- modelgolf plan \
  --spec specs/frontier_2135_audit_target.toml \
  --hardware local_planner \
  --runtime pg-local \
  --context 8192 \
  --artifact-budget-bytes 16000000 \
  --proof-artifact records/local/modelgolf_proof.pgrs \
  --quality-calibration records/local/modelgolf_quality_calibration.json \
  --release-evidence records/local/modelgolf_release_evidence.json \
  --output records/local/modelgolf.json

cargo run -q -p pg-local -- backend-check \
  --backend metal_apple \
  --output records/local/backend_metal_apple.json

cargo run -q -p pg-local -- wind-tunnel \
  --spec specs/frontier_2135_audit_target.toml \
  --trace records/h100/latest.json \
  --output records/local/wind_tunnel.json

cargo run -q -p pg-local -- wind-tunnel-suite \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir records/h100 \
  --output-dir records/local/wind_suite

cargo run -q -p pg-local -- wind-tunnel corpus \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records \
  --output records/wind_tunnel/calibration_corpus.json

cargo run -q -p pg-local -- wind-tunnel calibrate \
  --spec specs/frontier_2135_audit_target.toml \
  --corpus records/wind_tunnel/calibration_corpus.json \
  --output records/wind_tunnel/calibration_model.json

cargo run -q -p pg-local -- wind-tunnel scenario \
  --spec specs/frontier_2135_audit_target.toml \
  --calibration records/wind_tunnel/calibration_model.json \
  --active-recurrent-replay-cut-ms 10 \
  --target-step-ms 120 \
  --output records/wind_tunnel/scenario_exact_recurrent_replay.json

cargo run -q -p pg-local -- wind-tunnel report \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records \
  --output-dir records/wind_tunnel/reviewer_report \
  --active-recurrent-replay-cut-ms 10 \
  --target-step-ms 120

cargo run -q -p pg-local -- trace-index \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records \
  --output records/wind_tunnel/all_records_trace_index.json

cargo run -q -p pg-local -- lite init \
  --input specs/pg_lite/sample_train.txt \
  --output records/pg_lite/init_demo \
  --artifact-budget-bytes 1000000 \
  --val-fraction 0.20

cargo run -q -p pg-local -- lite run \
  --config specs/pg_lite/byte_golf_1mb_300s.toml \
  --output records/pg_lite/run_001

cargo run -q -p pg-local -- lite suite \
  --config-dir specs/pg_lite \
  --output records/pg_lite/suite

cargo run -q -p pg-local -- lite benchmark \
  --config specs/pg_lite/byte_golf_1mb_300s_metal_probe.toml \
  --output records/pg_lite/backend_benchmark \
  --repeats 3

cargo run -q -p pg-local -- lite verify-artifact \
  --artifact records/pg_lite/run_001/model.pglite.bin \
  --config specs/pg_lite/byte_golf_1mb_300s.toml \
  --output records/pg_lite/run_001/verify_artifact.json
```

The checked-in PG-Lite configs use the small sample corpus in `specs/pg_lite/`
so the command works from a fresh checkout. Replace those paths for real local
experiments. The repo does not require checked-in H100 traces; Wind Tunnel suite
handles empty or non-timing trace folders as failed calibration reports instead
of silently inventing evidence.

Use `--trace-dir ../records` when Wind Tunnel should inspect the historical H100
logs and finish-status files outside the Rust workspace. Generated Wind Tunnel
outputs are filtered out of later scans so reviewer reports do not self-calibrate
on their own JSON.

## Command Outputs

### `verify`

`verify` loads a normal run spec and emits `kind = "pg_local_verify"`.
Important fields:

- `artifact_manifest_ok`, `artifact_budget_known`, `artifact_budget_ok`: strict
  artifact checks when `--artifact` is supplied; skipped artifact evidence is
  explicit when no artifact is supplied.
- `caseops_sidecar_ok`, `bpb_smoke_ok`, `score_first_legal`: local eval legality
  and scoring checks.
- `eval_legality`: deterministic local proof for BPB byte accounting,
  score-before-update ordering, no-future-token access, document-boundary reset
  behavior, and PG-Lite artifact decode/eval equivalence.
- `data_preflight`: structured data readiness details for the configured
  train/validation/CaseOps patterns, including shard counts, token counts when
  shard headers are readable, #2135 canonical requirements, and notes explaining
  missing fields.
- `artifact_audit_ok`, `score_first_log_ok`: optional log/audit checks when
  `--artifact-audit` or `--score-first-log` are supplied.
- `proposal_claims_ok` and `proposal_features`: non-overclaiming audit fields.
- `status`: `pass` only when all required local checks pass.

This is local evidence. It is not final artifact proof unless a final artifact
and record-run audit are supplied.

### `proof-bundle`

`proof-bundle` runs:

- `verify`
- `dist-sim`
- `artifact-lab`
- `modelgolf plan`
- `wind-tunnel`
- `wind-tunnel report`
- `lite run`
- `lite benchmark`
- `lite suite`
- `backend-check --backend cpu_reference`
- `backend-check --backend metal_apple`

It writes a local evidence directory with `verify.json`, `dist_sim.json`,
`artifact_lab.json`, `modelgolf.json`, `wind_tunnel.json`, `pg_lite/report.json`,
`trace_corpus.json`, `calibration_model.json`,
`scenario_exact_recurrent_replay.json`, `pg_lite/model.pglite.bin`,
`wind_tunnel_reviewer/report.json`, `pg_lite_benchmark/summary.json`,
`pg_lite_suite/summary.json`, `backend_cpu_reference.json`,
`backend_metal_apple.json`, `proposal_features.json`, and
`proof_bundle.json`, plus `evidence_manifest.json` and `README.md`.

`evidence_manifest.json` always sets `record_claim=false` and
`leaderboard_claim=false`. It lists component statuses, generated files,
locally supported claims, missing remote evidence, blocking reasons, and caveats
so an external reviewer does not have to infer which reports are decisive.
`README.md` renders the same evidence into Markdown tables for quick handoff.

The bundle is intentionally non-record evidence. A passing bundle does not prove
H100 timing, final BPB, final compressed artifact size, or leaderboard standing.
The Metal backend is implemented behind the optional `metal_apple` Cargo
feature. A proof bundle generated without that feature still treats Metal as a
source/contract check plus explicitly labeled CPU fallback. A feature-enabled
backend check additionally probes the Rust Metal runtime and reports whether a
system default Metal device is visible. On this machine, after installing
Apple's Metal Toolchain component, the feature-enabled binary can find `xcrun
metal`, compile the checked-in `.metal` source to a temporary `.metallib`, link
the Rust runtime, and execute the Metal PG-Lite path.
When a #2135-shaped spec is used without canonical data mounted, the bundle is
expected to fail and `verify.json.data_preflight` plus
`evidence_manifest.json.blocking_reasons` explain the missing data evidence.

### `dist-sim`

`dist-sim` emits `kind = "pg_local_dist_sim"` and checks deterministic local
distributed optimizer math:

- parameter bank shapes
- reduce-scatter and all-gather equivalence flags
- shard-local Muon parity max absolute difference and tolerance
- approximate reduce-scatter/all-gather bytes per step
- `status`

This is not a network benchmark and does not measure NCCL or multi-GPU runtime.

### `artifact-lab`

`artifact-lab` emits `kind = "pg_local_artifact_lab"` and reports:

- optional strict artifact manifest status and artifact byte budget status
- `artifact_kind`, which is `parameter_golf_record` for strict official
  artifacts, `pg_lite` for local `model.pglite.bin` or debug
  `model.pglite.json` files, `none` when no artifact is supplied, and
  `unknown` for unrecognized inputs
- decoded PG-Lite artifact metadata when a local `model.pglite.bin` or
  `model.pglite.json` is supplied
- quant layout manifest CRC32 and quant kernel IDs
- per-group packed weight, scale, and LQER byte estimates
- mixed bit-allocation byte estimates such as `all_int4_floor`,
  `matrix6_mlp4_embed8`, and `embed8_mlp5_gate4`
- synthetic `quant`, `lqer`, and `compression` sweep records
- optional `mini_bpb` when both `--mini-train` and `--mini-val` byte files are
  supplied
- `proxy_only = true` on sweep records
- `status`

Sweep reconstruction, mixed bit-allocation estimates, and mini-BPB numbers are
local proxies. They should be used to choose experiments, not to claim final
artifact size or validation BPB.

### `modelgolf`

`modelgolf plan` emits `kind = "modelgolf_constraint_native_platform_plan"`.
It is the constraint-native planner described in
[`docs/modelgolf.md`](modelgolf.md). The report includes the resource contract,
typed artifact IR, ModelGolf Pack mixed-precision/LQER plan, CacheGolf KV-cache
plan, DeltaGolf byte-constrained adapter plan, TrainGolf artifact-aware
objective plus CPU gradient/train-loop integration, KernelForge fusion
contracts, and a Wind Tunnel Pareto summary.

The CacheGolf section includes `selected_policy_proof`, a deterministic local
proof for the chosen K/V cache policy. It records single-layer packed bytes,
K/V reconstruction errors, packed-vs-full attention error, explicit-dequant
attention parity, and whether the perturbation bound covers the observed
attention error. This is local CPU proof evidence, not fused CUDA/Metal timing
or full long-context BPB evidence.

The CacheGolf section also includes:

- `residual_sketch_policy`: planner-only residual sketch settings, trigger
  bound, protected recent window, rank/bits, byte estimate, and estimated bound
  reduction.
- `eviction_policy`: sink-token plus recent-window protection, evictable token
  count, and estimated bytes that would need eviction to meet the cache budget.
- `long_context_eval`: deterministic memory scaling rows for smaller,
  requested, and larger contexts, including selected-policy bytes,
  residual-sketch bytes, FP16 reduction, target-budget fit, and max modeled
  context under budget.

These fields make the long-context cache compiler choices visible, but they are
still byte/error-bound planning evidence rather than long-context BPB,
perplexity, or fused-runtime timing.

The Pack section includes `lqer_proofs` when selected options carry LQER bytes.
Each proof builds a deterministic quantization residual, materializes a rank-r
low-rank correction, and reports residual Frobenius reduction plus reduction of
the conservative activation/logit CE bound. This is local residual math
evidence, not calibrated tensor sensitivity or production-scale SVD validation.

The Pack section also includes `quality_comparison`, a deterministic equal-byte
proxy table for the Pack experiment protocol: uniform Q4, best uniform no-LQER
under the selected byte budget, exact mixed no-LQER under the selected byte
budget, selected mixed+LQER, and a deterministic LQER control/fallback. Rows
report bytes, surrogate quality loss, synthetic weighted residual MSE, and
activation/CE-bound proxy. This is not held-out BPB/perplexity or decode-speed
evidence.

`modelgolf pack-experiment` runs the plan's concrete LQER++ experiment table on
a bounded deterministic small model derived from the supplied spec. It exports,
strict-reloads, and scores uniform Q4, mixed Q4/Q6 without LQER, mixed+LQER
top-3, and deterministic random-LQER control artifacts. Rows include compressed
artifact bytes, local BPB proxy, local decode-speed smoke timing, and actual
selected LQER groups read back from artifact metadata. This closes the local
harness for Experiment 1; it is still not a pretrained small-model or held-out
validation result.

`modelgolf cache-experiment` runs the plan's concrete CacheGolf Experiment 3
protocol on a bounded deterministic CPU fixture. It sweeps the 25-row K/V bit
grid, quantizes K per channel and V per token, measures attention-output error,
checks explicit-dequant parity, compares the perturbation bound to observed
error, selects the lowest-error row that fits the active cache budget from the
resource contract, and emits long-context memory rows for that measured policy.
When no memory budget is supplied, the planner default remains 35% of requested
FP16 cache bytes. This closes the local harness for Experiment 3; it is still
not fused CUDA/Metal timing or long-context BPB/perplexity evidence.

The DeltaGolf section includes `selected_low_rank_proofs` when selected deltas
use rank-bearing families. Each proof builds a deterministic full update and
positive diagonal curvature, runs the weighted low-rank CPU reference,
materializes the factors, and compares weighted error against zero/lower-rank
baselines. It also includes `score_first_legality_audit`, a static metadata
audit for selected score-first/offline-trained artifact deltas, and
`domain_evaluation`, a deterministic equal-byte proxy table comparing no-delta,
selected, best-single, low-rank-only, and static-control plans. This is local
math/planner evidence; it is not a trained domain delta, runtime score-first
trace, formal competition/legal review, or private-corpus BPB result.

The KernelForge section includes `exact_tiled_ce.local_proof`, a deterministic
CPU parity proof for exact tiled output CE. It compares tiled loss,
`d_hidden`, and `d_weight` against full-logit CE plus explicit chain-rule
gradient accumulation. This is not generated CUDA/Metal kernel parity or
backend performance evidence.

`modelgolf kernel-experiment` runs the plan's concrete exact tiled CE protocol
on a bounded deterministic CPU fixture. It reports full-logit reference versus
tiled logit-free CE rows, loss/`dH`/`dW` parity, CE scratch bytes, scratch
reduction, and local CPU timing. This proves the local Experiment 2 harness, not
generated CUDA/Metal lowering or production speed.

The TrainGolf section includes `local_proof`, a deterministic CPU proof for the
artifact-aware regularizer and smooth export-gap bound. It checks the
stop-gradient gradient formula, fixed-grid distance reduction after one
regularizer step, regularized objective composition, and bound coverage on a
quadratic export-gap fixture. This is not GPU/backend integration or
post-export quality evidence.

When `--proof-artifact <path>` is supplied, the Pack report also writes an
actual deterministic local quant artifact through `pg-quant`, strict-reloads it
against the requested `QuantSpec`, and records byte count plus finite pre/post
reload smoke losses. This is local format/reload evidence; deterministic
fixture bytes are not final trained-artifact bytes or trained-model BPB
evidence.

When `--quality-calibration <json>` is supplied, the Pack report consumes
measured group/role quality-loss points, scales the role/bit/LQER surrogate
losses before exact DP selection, and emits fit factors plus before/after fit
error. This is calibration plumbing; release claims still need held-out
BPB/perplexity on the final trained artifact.

`modelgolf release-check` emits `kind = "modelgolf_release_readiness"`, a
fail-closed checklist that separates local deterministic gates from remaining
release evidence. Without `--release-evidence <json>`, it reports
`release_ready = false` until all blocking requirements have measured evidence:
final Pack bytes and held-out quality, production LQER calibration, fused
CacheGolf runtime and long-context eval, trained/legal DeltaGolf evidence,
TrainGolf post-export comparisons, KernelForge backend parity/perf, and fresh
Wind Tunnel profiler validation. With `--release-evidence`, it validates those
measured fields and marks only the satisfied pillars non-blocking. The evidence
file must bind to the current spec fingerprint and hardware/runtime labels,
then provide SHA-256 verified JSON source reports for every release pillar. The
summary evidence must match those digested source-report claims, the final
artifact SHA-256, selected cache policy, selected delta names, and generated
kernel IDs; Pack quality uses a recomputed BPB delta from held-out and baseline
BPB.

Section-specific subcommands emit only one part of the report:

- `modelgolf pack`
- `modelgolf pack-experiment`
- `modelgolf cache-plan`
- `modelgolf cache-experiment`
- `modelgolf delta-plan`
- `modelgolf train-plan`
- `modelgolf kernel-plan`
- `modelgolf kernel-experiment`
- `modelgolf wind-plan`
- `modelgolf release-check`

All ModelGolf outputs are deterministic local planning evidence. They do not
establish H100 timing, final validation BPB, decode speed, or production kernel
performance without the evidence listed in `platform_status`.

### `backend-check`

`backend-check` emits `kind = "pg_local_backend_check"` and checks local
backend readiness without running a benchmark.

For `--backend metal_apple`, the report includes:

- checked-in kernel source path:
  `crates/pg-local/kernels/pg_lite_ngram.metal`
- whether that source exists
- source CRC32, expected kernel symbols, found kernel symbols,
  per-kernel `kernel_contracts`, and `kernel_contract_ok`
- whether `xcrun --find metal` can locate Apple’s Metal compiler
- `metal_compiler_error` when Xcode/SDK licensing or toolchain selection blocks
  the compiler probe
- whether the checked-in `.metal` source compiled to a temporary `.metallib`
  (`metal_compile_smoke_*`)
- whether the `metal_apple` Cargo feature is enabled
- whether the Rust Metal runtime is linked
- `executable_backend_available`

Default Rust v1 status is expected to be `fail` for `metal_apple` unless the
binary is built with `--features metal_apple` and the host exposes a usable
Metal device. The checked-in source and feature-gated executor are local
acceleration surfaces, but a failing backend check must be treated as no Metal
execution/perf evidence.

The checked-in Metal source and Rust executor currently define the PG-Lite
n-gram/residual kernel contract:

- dense add-one-smoothed table initialization for bigram and residual counts
- parallel atomic training into u32 count tables
- saturating residual u32-to-u16 compaction for compact artifacts/eval
- bigram and residual row-sum precompute kernels
- presummed u16/u32 loss kernels
- fused presummed loss+block-reduction kernels that avoid materializing
  `token_losses`
- legacy non-presummed loss kernels for bring-up parity
- a sequential score-first eval/update kernel for order-dependent stream-golf
  legality checks

The Metal hash uses the same 64-bit FNV-1a constants as the CPU reference
`context_bucket` path. When `--features metal_apple` is enabled and a Metal
device is visible, `lite run` can execute ByteGolf training and fixed eval
through Metal. ArtifactGolf fixed eval and StreamGolf score-first eval also
have runtime entrypoints; the StreamGolf kernel remains sequential by design
because score-before-update ordering is semantically serial.

### `lite benchmark`

`lite benchmark` runs the same PG-Lite config with requested `cpu_reference` and
`metal_apple` backends. CPU fallback is disabled for benchmark runs, so a Metal
toolchain/runtime issue appears as an unavailable Metal backend instead of a
quiet CPU run. The report includes median total/train/eval wall time, local BPB,
artifact bytes, execution backend labels, acceleration counts, and CPU-vs-Metal
speedup when both backends run.

This command is the local PG-Lite performance proof surface. It still reports
local proxy BPB only.

### `lite verify-artifact`

`lite verify-artifact` strict-loads a `model.pglite.bin` or debug
`model.pglite.json`, reconstructs the local model, verifies the model
fingerprint, and reports artifact bytes plus CRC32. When `--config` is supplied,
it also checks:

- the artifact source config fingerprint
- the configured artifact byte budget
- local proxy BPB on either `--val` or the config validation split

This closes the PG-Lite artifact lifecycle: a local run is only complete when
the artifact can be independently decoded and scored.

### `wind-tunnel`

`wind-tunnel` emits `kind = "pg_local_wind_tunnel"` and always sets
`estimate_only = true`.

Core fields:

- `trace_path` and `trace_total_ms` when a trace JSON is supplied.
- `prediction_source` and `trace_coverage`: whether the estimate is trace
  calibrated or defaulted, expected trace fields, mapped/defaulted stages, and
  coverage ratios.

Subcommands:

- `wind-tunnel corpus`: recursively scans a historical records tree through the
  same parser as `trace-index`, dedupes paired `.json` and
  `.finish_status.json` artifacts into one run, scores richer artifacts higher,
  and writes `kind = "pg_local_wind_tunnel_corpus"`.
- `wind-tunnel calibrate`: consumes either `--trace-dir` or a saved corpus JSON
  and writes `kind = "pg_local_wind_tunnel_calibration"` with total-step,
  exact/#2135, active/inactive recurrent, recurrent-gap, and stage metrics plus
  confidence labels.
- `wind-tunnel scenario`: consumes a calibration JSON or a single trace and
  evaluates explicit cuts such as
  `--active-recurrent-replay-cut-ms`, `--bank-update-cut-ms`, and
  `--graph-overhead-cut-ms` against a `--target-step-ms`.
- `wind-tunnel report`: writes a reviewer-facing packet with trace index,
  deduped corpus, calibration model, recurrent-cut scenario, suite summary,
  train-only holdout predictions, rank-correlation sanity checks, JSON report,
  and Markdown report.

The corpus/calibration/scenario chain is the preferred local planning path:
raw historical evidence becomes a deduped run set, then a confidence-labeled
calibration model, then a concrete decision report for the next paid H100 run.
It is still estimate-only; clearing a scenario target is not validation.

`wind-tunnel report` is the best single command for handoff. It writes:

- `trace_index.json`: all parseable and unparseable trace files.
- `trace_corpus.json`: one selected evidence record per deduplicated run.
- `calibration_model.json`: total, exact #2135, active recurrent, inactive
  recurrent, and stage metrics.
- `scenario_exact_recurrent_replay.json`: estimate-only projection for the
  requested active recurrent replay cut.
- `suite/summary.json` and `suite/summary.md`: per-trace shape-model comparison.
- `report.json` and `report.md`: compact external-review summary.

The holdout check is intentionally local: it splits timed traces
deterministically, fits only on the train split, predicts each held-out run from
class medians, non-total stage fields, and active/inactive recurrent timing
offsets, then reports per-run prediction error and Spearman rank correlation.
This is a calibration sanity check, not an H100 emulator.
- `stage_estimates`: per-stage millisecond estimates, percent of step, and
  source.
- `train_step_ms_estimate`, `active_recurrent_step_ms_estimate`,
  `inactive_recurrent_step_ms_estimate`.
- `expected_train_wall_seconds`, `budget_status`, and `risk_flags`.
- `recurrent_split`: active/inactive recurrent timing, source, and exact
  boundary-fusion layer details.
- `expected_train_steps_in_600s`, `eval_time_ms_estimate`,
  `artifact_bytes_estimate`.
- `top_bottleneck`, `top_bottlenecks`, `next_recommended_experiment`, and
  `recommendations`.
- `what_if_scenarios`: estimate-only deltas for candidate work such as exact
  recurrent replay cuts, persistent-CTA recurrent boundary kernels, optimizer
  launch collapse, CE retests, and full train-step graph capture. These are
  planning scenarios only, not measured speedups.
- `experiment_rankings`: the same scenario set ranked by estimated step-time
  reduction, train-budget impact, trace coverage, and whether the scenario is
  speculative. Each row includes the validation run that would be required
  before promoting the change.
- `operation_dag`: spec-derived operation nodes with rough FLOP and HBM-byte
  estimates plus implementation/claim status.
- `persistent_cta_block_backward = "not_implemented_unclaimed"` unless the code
  changes and the audit changes with it.

Wind Tunnel is a planning estimate. It is not an H100 emulator, CUDA profiler,
BPB validator, or record-readiness certificate.

### `wind-tunnel-suite`

`wind-tunnel-suite` runs the Wind Tunnel over JSON and log traces in a trace
directory. It scans `*.json`, `*.jsonl`, `*.log`, `*.txt`, and `*.out`
recursively, and it understands top-level JSON timing fields, nested
`json_events.run_timing_json`, `run_timing_json=...` log lines, and flat
`timing_*=...` logs. It writes:

- `baseline_shape_model.json`: the no-trace shape-model estimate.
- `<trace-stem>.wind_tunnel.json`: one estimate report per trace.
- `summary.json`: trace coverage, useful nonzero stage coverage, trace total,
  measured-step trace count, calibration role, baseline absolute error, top
  bottleneck, and risk flags.
- `summary.md`: a compact review table for handoff.

The suite is a calibration and triage tool. It can show how wrong the local
shape model is against prior H100 traces, but it does not replace a measured
H100 run.

Calibration roles are deliberately conservative:

- `stage_calibration`: enough nonzero stage fields are present for stage-level
  attribution.
- `partial_stage_calibration`: total timing is useful, but many stage fields are
  missing or zero because instrumentation was disabled by the run profile.
- `total_calibration`: total step time is useful, but no stage attribution is
  available.
- `shape_model_only`: metadata-only input or no trace timing.

Zero-valued stage fields are reported separately. They usually mean a graph
profile or run mode did not collect that stage, not that the stage was free.

### `trace-index`

`trace-index` recursively scans a historical records tree and emits
`kind = "pg_local_trace_index"`.

It is broader than `wind-tunnel-suite`: instead of writing one model estimate
per trace, it creates a reviewer-facing inventory of what the trace corpus can
prove. It extracts:

- total measured step time
- active/inactive recurrent timing
- timing steps and completed steps
- mode, record profile, recurrent profile, graph profile, world size, and
  sequence length when available
- canonical CaseOps flags, train shard counts, validation token counts
- BF16/F32 bridge counters and host-batch counters when present
- nonzero stage fields, zero stage fields, signal quality, and calibration role
- tags such as `frontier_2135`, `exact_2135`, `canonical_caseops`,
  `record_shaped_proxy`, and `active_recurrent_timed`

Use this when answering whether the previous Modal runs contain useful evidence.
The checked-in all-records index currently finds the historical exact #2135
floor around `127.1 ms/step`, with active recurrent timing still around
`155-156 ms/active step`.

### `lite init`

`lite init` turns any byte corpus into a self-contained PG-Lite benchmark
directory. It writes:

- `data/train.bytes` and `data/val.bytes`
- `configs/byte_golf_ngram_residual.toml`
- `configs/byte_golf_ngram_baseline.toml`
- `configs/stream_golf_score_first.toml`
- `configs/artifact_golf_empty.toml`
- `configs/byte_golf_metal_probe.toml` when the requested primary backend is
  not already Metal
- `init_report.json`
- `README.md`

The split is deterministic: validation is the final `--val-bytes` bytes, or
`--val-fraction` of the input clamped so both train and validation are non-empty.
Generated configs use absolute data paths so they can be run from any working
directory.

Typical local lifecycle:

```bash
pg-local lite init --input corpus.txt --output pg_lite_local
pg-local lite run \
  --config pg_lite_local/configs/byte_golf_ngram_residual.toml \
  --output pg_lite_local/run_residual
pg-local lite verify-artifact \
  --artifact pg_lite_local/run_residual/model.pglite.bin \
  --config pg_lite_local/configs/byte_golf_ngram_residual.toml \
  --output pg_lite_local/run_residual/verify_artifact.json
pg-local lite suite --config-dir pg_lite_local/configs --output pg_lite_local/suite
```

### `lite run`

`lite run` reads a PG-Lite TOML config and writes `<output-dir>/report.json`
with `kind = "pg_lite_run"`.

Supported v1 config boundaries:

- `score = "bpb"`
- `[data] format = "bytes"`
- `[model] vocab = "byte"`
- `[backend] kind = "cpu_reference"` for portable execution.
- `[backend] kind = "metal_apple"` requires a binary built with
  `--features metal_apple` and a visible Metal device. If unavailable, it fails
  by default; setting `allow_cpu_fallback = true` runs the CPU reference path
  and reports the requested backend, execution backend, fallback reason, and
  `backend_accelerated = false`.
- `[backend] kind = "mlx_prototype"` is a parsed future interface only in Rust
  v1 and requires explicit CPU fallback.
- tracks: `byte_golf`, `stream_golf`, `artifact_golf`
- model families: `byte_ngram`, `ngram_residual`, `artifact_only`

Report fields include `config_fingerprint`, track behavior, backend status,
requested/execution backend, backend fallback reason,
train/validation byte counts, bytes actually used for training, train/eval wall
seconds, time budgets, `artifact_manifest_path`, `artifact_bytes_estimate`,
`artifact_actual_bytes`, `artifact_path`, artifact/time/memory budget booleans,
`validation_bpb`, `validation_bpb_scope`, score-first counters,
`proxy_only = true`, and `status`.

The output directory also contains `model.pglite.bin`, a compact deterministic
PG-Lite model artifact, `model.pglite.json`, a stable debug representation of
the same model, plus `artifact_manifest.json`. The manifest uses actual binary
artifact bytes for the local budget gate. These are local reproducibility
artifacts, not final Parameter Golf record artifact proof.

`stream_golf` uses score-first adaptation during validation. `artifact_golf`
requires `model.family = "artifact_only"` and does not train on the train
corpus. If `[model].artifact_path` is set, `artifact_golf` loads that saved
`model.pglite.bin` or debug `model.pglite.json` and evaluates it; otherwise it
scores the empty artifact-only baseline.

PG-Lite BPB is a small local proxy benchmark. It must not be compared directly
to official FineWeb/H100 leaderboard BPB.

### `lite suite`

`lite suite` runs every `*.toml` config in a PG-Lite config directory. It
writes one output directory per config plus:

- `summary.json`: stable machine-readable run summaries, best local proxy BPB,
  budget status, and score-first counters.
- `summary.md`: a reviewer-friendly table.

Use this as the default local PG-Lite regression command before changing the
CPU reference benchmark, artifact estimator, or score-first path.

## Checked-In PG-Lite Configs

- `specs/pg_lite/byte_golf_256kb_60s.toml`: tiny byte-golf smoke run with a
  256 KB artifact budget.
- `specs/pg_lite/byte_golf_1mb_300s.toml`: larger byte-golf local reference.
- `specs/pg_lite/byte_golf_1mb_300s_metal_probe.toml`: same local reference
  with `metal_apple` requested and explicit CPU fallback for environments
  without the feature/runtime.
- `specs/pg_lite/byte_golf_ngram_baseline.toml`: byte ngram baseline without
  residual buckets affecting the score.
- `specs/pg_lite/stream_golf_1mb_score_first.toml`: stream-golf score-first
  validation path.
- `specs/pg_lite/artifact_golf_1mb.toml`: artifact-only path with no training
  budget.

`records/pg_lite/init_demo/` is a generated example from `lite init`. It proves
the fresh-corpus lifecycle: deterministic split, generated configs, single run,
suite run, artifact decode, budget check, and local proxy BPB verification.

## Non-Overclaiming Boundaries

- `verify`, `artifact-lab`, `dist-sim`, and `lite run` are local evidence.
- `wind-tunnel` is estimate-only and must be reported as such.
- `wind-tunnel-suite` compares estimates to trace files; it is calibration, not
  validation.
- PG-Lite BPB is a local proxy benchmark and must not be compared directly to
  the official FineWeb/H100 leaderboard.
- A PG-Lite `metal_apple` run with CPU fallback is not Metal acceleration. It is
  a backend-contract smoke that keeps the report honest when the feature,
  compiler, or visible device is missing.
- `backend-check` reports local readiness only. A present `.metal` source file
  is not compiled or executed evidence unless `rust_runtime_linked=true` and
  `executable_backend_available=true`.
- PG-Lite `artifact_actual_bytes` is an actual local `model.pglite.bin` byte
  count, but it is not the official compressed Rust/CUDA record artifact proof.
- Synthetic artifact-lab sweeps are proxies, not final compression proof.
- `dist-sim` validates local math, not network throughput.
- Persistent-CTA block backward and full train-step CUDA graph remain
  unimplemented unless their audit states change.
- True XSA-inside-SDPA has local reference/parity surfaces only; it must not
  be claimed as record-active until `pg-model` routes XSA layers through that
  path and H100 parity/perf is measured.
