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
cargo run -q -p pg-local -- help lite
cargo run -q -p pg-local -- help lite suite
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

cargo run -q -p pg-local -- trace-index \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records \
  --output records/wind_tunnel/all_records_trace_index.json

cargo run -q -p pg-local -- lite run \
  --config specs/pg_lite/byte_golf_1mb_300s.toml \
  --output records/pg_lite/run_001

cargo run -q -p pg-local -- lite suite \
  --config-dir specs/pg_lite \
  --output records/pg_lite/suite
```

The checked-in PG-Lite configs use the small sample corpus in `specs/pg_lite/`
so the command works from a fresh checkout. Replace those paths for real local
experiments. The repo does not require checked-in H100 traces; Wind Tunnel suite
handles empty or non-timing trace folders as failed calibration reports instead
of silently inventing evidence.

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
- `wind-tunnel`
- `lite run`
- `lite suite`
- `backend-check --backend cpu_reference`
- `backend-check --backend metal_apple`

It writes a local evidence directory with `verify.json`, `dist_sim.json`,
`artifact_lab.json`, `wind_tunnel.json`, `pg_lite/report.json`,
`pg_lite/model.pglite.bin`, `pg_lite_suite/summary.json`, `backend_cpu_reference.json`,
`backend_metal_apple.json`, `proposal_features.json`, and
`proof_bundle.json`, plus `evidence_manifest.json` and `README.md`.

`evidence_manifest.json` always sets `record_claim=false` and
`leaderboard_claim=false`. It lists component statuses, generated files,
locally supported claims, missing remote evidence, blocking reasons, and caveats
so an external reviewer does not have to infer which reports are decisive.
`README.md` renders the same evidence into Markdown tables for quick handoff.

The bundle is intentionally non-record evidence. A passing bundle does not prove
H100 timing, final BPB, final compressed artifact size, or leaderboard standing.
The Metal backend check is expected to be non-executable until a Rust Metal
runtime is linked; a proof bundle only requires that this state is reported
truthfully and that the source boundary exists.
On this machine, updating the Xcode Command Line Tools to 16.2 was not enough
to provide `xcrun metal` or `xcrun metallib`; a full Xcode toolchain is still
required before `metal_compile_smoke_*` can pass.
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

### `backend-check`

`backend-check` emits `kind = "pg_local_backend_check"` and checks local
backend readiness without running a benchmark.

For `--backend metal_apple`, the report includes:

- checked-in kernel source path:
  `crates/pg-local/kernels/pg_lite_ngram.metal`
- whether that source exists
- source CRC32, expected kernel symbols, found kernel symbols, and
  `kernel_contract_ok`
- whether `xcrun --find metal` can locate Apple’s Metal compiler
- whether the checked-in `.metal` source compiled to a temporary `.metallib`
  (`metal_compile_smoke_*`)
- whether the `metal_apple` Cargo feature is enabled
- whether the Rust Metal runtime is linked
- `executable_backend_available`

Current Rust v1 status is expected to be `fail` for `metal_apple` unless and
until a real Metal runtime is linked. The checked-in source is a kernel boundary
for the n-gram evaluator; it is not execution evidence by itself.

### `wind-tunnel`

`wind-tunnel` emits `kind = "pg_local_wind_tunnel"` and always sets
`estimate_only = true`.

Core fields:

- `trace_path` and `trace_total_ms` when a trace JSON is supplied.
- `prediction_source` and `trace_coverage`: whether the estimate is trace
  calibrated or defaulted, expected trace fields, mapped/defaulted stages, and
  coverage ratios.
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

### `lite run`

`lite run` reads a PG-Lite TOML config and writes `<output-dir>/report.json`
with `kind = "pg_lite_run"`.

Supported v1 config boundaries:

- `score = "bpb"`
- `[data] format = "bytes"`
- `[model] vocab = "byte"`
- `[backend] kind = "cpu_reference"` for actual execution.
- `[backend] kind = "metal_apple"` or `"mlx_prototype"` may be parsed only as
  explicit interface probes. They fail by default; setting
  `allow_cpu_fallback = true` runs the CPU reference path and reports the
  requested backend, execution backend, fallback reason, and
  `backend_accelerated = false`.
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
- `specs/pg_lite/byte_golf_ngram_baseline.toml`: byte ngram baseline without
  residual buckets affecting the score.
- `specs/pg_lite/stream_golf_1mb_score_first.toml`: stream-golf score-first
  validation path.
- `specs/pg_lite/artifact_golf_1mb.toml`: artifact-only path with no training
  budget.

## Non-Overclaiming Boundaries

- `verify`, `artifact-lab`, `dist-sim`, and `lite run` are local evidence.
- `wind-tunnel` is estimate-only and must be reported as such.
- `wind-tunnel-suite` compares estimates to trace files; it is calibration, not
  validation.
- PG-Lite BPB is a local proxy benchmark and must not be compared directly to
  the official FineWeb/H100 leaderboard.
- A PG-Lite `metal_apple` run with CPU fallback is not Metal acceleration. It is
  a backend-contract smoke that keeps the report honest until real Metal kernels
  are wired and locally verified.
- `backend-check` reports local readiness only. A present `.metal` source file
  is not a compiled or executed Metal backend.
- PG-Lite `artifact_actual_bytes` is an actual local `model.pglite.bin` byte
  count, but it is not the official compressed Rust/CUDA record artifact proof.
- Synthetic artifact-lab sweeps are proxies, not final compression proof.
- `dist-sim` validates local math, not network throughput.
- Persistent-CTA block backward and full train-step CUDA graph remain
  unimplemented unless their audit states change.
- True XSA-inside-SDPA has local reference/parity surfaces only; it must not
  be claimed as record-active until `pg-model` routes XSA layers through that
  path and H100 parity/perf is measured.
