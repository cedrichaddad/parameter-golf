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
