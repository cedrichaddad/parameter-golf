# Wind Tunnel Local Calibration Records

This folder contains checked-in, local Wind Tunnel reports. These reports are
planning and triage evidence only. They do not claim H100 validation, record
readiness, or BPB competitiveness.

## Current Suite

`frontier_2135_modal_results_suite/` was generated from historical Modal result
files in:

```text
../records/track_non_record_16mb/2026-04-30_RustCudaSystems/modal_results
```

Regenerate it from `parameter-golf-rs/` with:

```bash
cargo run -q -p pg-local -- wind-tunnel-suite \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records/track_non_record_16mb/2026-04-30_RustCudaSystems/modal_results \
  --output-dir records/wind_tunnel/frontier_2135_modal_results_suite
```

## Summary

- Traces found: 14
- Traces with measured step timing: 10
- Traces with stage attribution: 6
- Total-only traces: 4
- Shape-only traces: 4
- Shape-model baseline: 134.200 ms/step
- Mean absolute baseline error against timed traces: 6.537 ms/step

The best #2135 exact/proxy totals in this historical set remain around
127.1-128.6 ms/step. The stage-calibration trace is
`frontier_2135_recurrent_stage_probe_caseops_v2.json` at 141.531 ms/step. Most
fast graph-timed runs are only partial stage-calibration traces because the
graph profile emitted many zero-valued stage fields. Treat those zeros as
disabled instrumentation, not proof that those stages cost nothing.

## How To Read Calibration Roles

- `stage_calibration`: enough nonzero stage fields exist to use the trace for
  stage-level attribution.
- `partial_stage_calibration`: total step time is useful, but nonzero stage
  coverage is sparse.
- `total_calibration`: total step time is useful; stage attribution is absent.
- `shape_model_only`: metadata-only trace or no timing fields.

For current #2135 planning, use the timed traces for total-step calibration and
use only `stage_calibration` traces for bottleneck attribution.

## All-Records Trace Index

`all_records_trace_index.json` was generated from the broader historical
records tree:

```bash
cargo run -q -p pg-local -- trace-index \
  --spec specs/frontier_2135_audit_target.toml \
  --trace-dir ../records \
  --output records/wind_tunnel/all_records_trace_index.json
```

Current index summary:

- Trace-like files scanned: 109
- Parseable trace artifacts: 46
- Traces with step timing: 13
- Stage-calibration traces: 1
- Partial-stage-calibration traces: 6
- Total-calibration traces: 6
- Canonical CaseOps traces: 4
- Best measured step time in the index: 127.109924 ms/step
- Fastest exact/#2135-tagged trace: 127.109924 ms/step
- Best active recurrent timing: 155.576088 ms/active step

This answers the main evidence question: the historical Modal runs are useful
for total-step calibration and for confirming the exact active-recurrent
blocker, but only one trace in the current local corpus has enough nonzero stage
fields for stage-level attribution. Most fast graph runs are total or partial
calibration traces.

## Deduped Corpus And Calibration Model

The decision-grade Wind Tunnel flow is:

```bash
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
```

Current corpus summary:

- Trace-like files scanned: 109
- Parseable trace artifacts: 46
- Deduped runs: 44
- Duplicate source artifacts folded into richer runs: 2
- Timed runs: 12
- Exact/#2135 timed runs in calibration: 10
- Rich stage-calibration runs: 1
- Best exact/#2135 total: 127.109924 ms/step
- Best active recurrent timing: 155.576088 ms/active step

Current calibration summary:

- Exact/#2135 total-step estimate: 127.993650 ms median, 127.109924 ms best,
  128.631212 ms p90, high multi-trace confidence
- Active recurrent estimate: 156.310609 ms median, 155.576088 ms best,
  157.502122 ms p90, high multi-trace confidence
- Stage attribution confidence: low single-trace

Current scenario summary:

- Scenario: exact active recurrent replay cut of 10 ms
- Base: 127.109924 ms/step from best exact/#2135 calibration
- Predicted step: 117.109924 ms/step
- Target: 120 ms/step
- Status: clears target as an estimate only

Interpretation: the old H100 runs are now useful enough to decide that the next
paid run should test an exact recurrent replay cut, but they are not rich enough
to claim stage-level attribution beyond the one stage-probe trace.

## Reviewer Report

`reviewer_report/` is the current one-command external handoff packet. It
contains `trace_index.json`, `trace_corpus.json`, `calibration_model.json`,
`scenario_exact_recurrent_replay.json`, `suite/summary.json`, `report.json`,
and `report.md`.

Current reviewer-report summary:

- Trace-like files scanned: 109
- Parseable trace artifacts: 46
- Deduped runs: 44
- Timed runs: 12
- Exact/#2135 timed runs: 10
- Canonical CaseOps traces: 4
- Best exact/#2135 total: 127.109924 ms/step
- Best active recurrent timing: 155.576088 ms/active step
- Scenario with a 10 ms exact recurrent replay cut predicts 117.109924 ms/step
- Holdout check: 6 train samples, 6 holdout samples, 25.856 ms/step mean
  absolute error

The holdout error is deliberately conservative and reinforces the main caveat:
Wind Tunnel is useful for planning the next H100 experiment, not for claiming a
validated speed result.
