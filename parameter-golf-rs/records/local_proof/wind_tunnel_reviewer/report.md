# PG Wind Tunnel Reviewer Report

This report is local planning evidence. It indexes historical traces, deduplicates runs, calibrates a simple stage model, and estimates the next recurrent-replay cut. It is not an H100 validation run.

## Summary

- Spec: `specs/frontier_2135_audit_target.toml`
- Trace dir: `../records`
- Status: `pass`
- Traces found: `109`
- Parseable traces: `46`
- Deduped runs: `44`
- Timed runs: `12`
- Exact #2135 timed runs: `10`
- Stage calibration runs: `1`
- Canonical CaseOps runs: `4`
- Best exact #2135 timed trace: `127.110 ms/step`
- Best active recurrent trace: `155.576 ms/active step`
- Calibration confidence: `exact_total=high_multi_trace; active_recurrent=high_multi_trace; stage_attribution=low_single_trace`
- Scenario prediction after requested cut: `117.110 ms/step`
- Scenario clears target: `true`
- Recommended next experiment: run the cheapest exact #2135 H100 A/B that exercises active recurrence; require median/p90 and bridge/host-copy counters before promotion; next trace to collect: collect graph-disabled exact #2135 stage timing with active recurrent windows and nonzero qkv/mlp/attention/output/optimizer fields

## Holdout Check

- Status: `pass`
- Train samples: `6`
- Holdout samples: `6`
- Median train baseline: `127.786 ms/step`
- Mean absolute holdout error: `25.856 ms/step`
- Mean absolute holdout error: `10.57%`

## Top Calibrated Stages

| Stage | Estimate ms | Samples | Confidence | Source fields |
|---|---:|---:|---|---|
| `qkv_backward` | 22.024 | 1 | `low_single_trace` | `timing_cuda_backward_block_qkv_ms_per_step` |
| `mlp_backward` | 20.445 | 1 | `low_single_trace` | `timing_cuda_backward_block_mlp_ms_per_step` |
| `attention_backward` | 14.155 | 1 | `low_single_trace` | `timing_cuda_backward_block_attention_sdpa_ms_per_step` |
| `optimizer_update` | 8.970 | 7 | `high_multi_trace` | `timing_cuda_bank_update_ms_per_step` |
| `gate_xsa_backward` | 6.869 | 1 | `low_single_trace` | `timing_cuda_backward_block_attn_out_gate_xsa_ms_per_step` |
| `output_ce` | 5.814 | 1 | `low_single_trace` | `timing_cuda_backward_output_ms_per_step` |
| `non_bank_update` | 2.741 | 7 | `high_multi_trace` | `timing_cuda_non_bank_update_ms_per_step` |

## Scenario

- Base step: `127.110 ms/step` from `calibration:exact_2135_total_step_ms:best_ms`
- Requested active recurrent replay cut: `10.000 ms`
- Target: `120.000 ms/step`
- Predicted step: `117.110 ms/step`
- Remaining gap: `0.000 ms`
- Risk: `medium_low`

## Corpus And Suite

- Index status: `pass`
- Corpus status: `pass`
- Calibration status: `pass`
- Suite status: `pass`
- Suite mean absolute baseline error: `25.239 ms/step`

## Warnings

- fewer than two rich stage-calibration traces; use stage estimates as low-confidence attribution

## Generated Files

- Trace index: `records/local_proof/wind_tunnel_reviewer/trace_index.json`
- Corpus: `records/local_proof/wind_tunnel_reviewer/trace_corpus.json`
- Calibration: `records/local_proof/wind_tunnel_reviewer/calibration_model.json`
- Scenario: `records/local_proof/wind_tunnel_reviewer/scenario_exact_recurrent_replay.json`
- Suite summary: `records/local_proof/wind_tunnel_reviewer/suite/summary.json`
- JSON report: `records/local_proof/wind_tunnel_reviewer/report.json`
