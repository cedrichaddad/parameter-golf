# PG-Local Proof Bundle

This directory is a local, non-record evidence packet for the Rust/CUDA Parameter Golf project. It is designed for review and planning before an H100 record run.

## Status

- Proof bundle status: `fail`
- Record claim: `false`
- Leaderboard claim: `false`

## Component Reports

| Component | Status | Report |
|---|---:|---|
| `verify` | `fail` | `records/local_proof/verify.json` |
| `dist_sim` | `pass` | `records/local_proof/dist_sim.json` |
| `artifact_lab` | `pass` | `records/local_proof/artifact_lab.json` |
| `wind_tunnel` | `pass` | `records/local_proof/wind_tunnel.json` |
| `wind_tunnel_calibration` | `pass` | `records/local_proof/calibration_model.json` |
| `wind_tunnel_scenario` | `pass` | `records/local_proof/scenario_exact_recurrent_replay.json` |
| `wind_tunnel_reviewer` | `pass` | `records/local_proof/wind_tunnel_reviewer/report.json` |
| `pg_lite` | `pass` | `records/local_proof/pg_lite/report.json` |
| `pg_lite_benchmark` | `pass` | `records/local_proof/pg_lite_benchmark/summary.json` |
| `pg_lite_suite` | `pass` | `records/local_proof/pg_lite_suite/summary.json` |
| `backend_cpu_reference` | `pass` | `records/local_proof/backend_cpu_reference.json` |
| `backend_metal_apple` | `runtime_available` | `records/local_proof/backend_metal_apple.json` |

## Local Evidence

| Claim | Status | Evidence |
|---|---:|---|
| `proposal_claim_honesty` | `pass` | `records/local_proof/verify.json` |
| `record_data_preflight` | `fail` | `records/local_proof/verify.json` |
| `local_eval_legality` | `pass` | `records/local_proof/verify.json` |
| `distributed_optimizer_math` | `pass` | `records/local_proof/dist_sim.json` |
| `quant_layout_compiler` | `local_parity_tested` | `records/local_proof/artifact_lab.json` |
| `pg_lite_cpu_reference` | `pass` | `records/local_proof/pg_lite_suite/summary.json` |
| `pg_lite_metal_backend` | `local_validated` | `records/local_proof/pg_lite_benchmark/summary.json` |
| `wind_tunnel_estimate` | `estimate_only` | `records/local_proof/wind_tunnel.json` |
| `wind_tunnel_reviewer_packet` | `pass` | `records/local_proof/wind_tunnel_reviewer/report.json` |
| `wind_tunnel_trace_corpus` | `pass` | `records/local_proof/trace_corpus.json` |
| `wind_tunnel_next_cut_scenario` | `scenario_clears_target_estimate_only` | `records/local_proof/scenario_exact_recurrent_replay.json` |

## Still Requires Remote Validation

| Requirement | Status | Evidence |
|---|---:|---|
| `full_record_train_eval_export` | `missing` | n/a |
| `post_export_full_validation_bpb` | `missing` | n/a |
| `artifact_budget_from_final_record_run` | `missing` | `records/local_proof/verify.json` |
| `exact_2135_h100_step_time` | `missing` | n/a |
| `persistent_cta_block_backward` | `not_implemented` | `records/local_proof/proposal_features.json` |
| `record_active_xsa_inside_sdpa` | `local_parity_tested_not_record_active` | `records/local_proof/proposal_features.json` |

## Blocking Reasons

- verify status is fail; notes: artifact not supplied; artifact checks are skipped; CaseOps byte sidecar is required or configured but not locally valid
- final artifact bytes were not supplied to verify
- canonical CaseOps byte sidecar is not locally valid
- data preflight: train shard count mismatch: found 0 required 80
- data preflight: validation token count mismatch: found Some(0) required 47851520
- data preflight: validation doc count is not available from local shard headers
- data preflight: CaseOps sidecar pattern did not resolve to readable token shards: Some("/data/datasets/fineweb10B_sp8192/fineweb_val_bytes_*.bin")

## Caveats

- This package is local evidence only; it is not a Parameter Golf leaderboard claim.
- Wind Tunnel values are estimates unless backed by supplied H100 timing traces.
- PG-Lite BPB and artifact-lab mini-BPB are local proxy scores, not FineWeb validation BPB.
- A PG-Lite Metal claim requires backend-check to report executable_backend_available=true; CPU fallback remains a contract smoke only.

## File Index

| Kind | Required | Exists | Path |
|---|---:|---:|---|
| `verify_report` | `true` | `true` | `records/local_proof/verify.json` |
| `distributed_sim_report` | `true` | `true` | `records/local_proof/dist_sim.json` |
| `artifact_lab_report` | `true` | `true` | `records/local_proof/artifact_lab.json` |
| `wind_tunnel_report` | `true` | `true` | `records/local_proof/wind_tunnel.json` |
| `wind_tunnel_trace_corpus` | `true` | `true` | `records/local_proof/trace_corpus.json` |
| `wind_tunnel_calibration_model` | `true` | `true` | `records/local_proof/calibration_model.json` |
| `wind_tunnel_scenario_report` | `true` | `true` | `records/local_proof/scenario_exact_recurrent_replay.json` |
| `wind_tunnel_reviewer_report` | `true` | `true` | `records/local_proof/wind_tunnel_reviewer/report.json` |
| `pg_lite_run_report` | `true` | `true` | `records/local_proof/pg_lite/report.json` |
| `pg_lite_backend_benchmark` | `true` | `true` | `records/local_proof/pg_lite_benchmark/summary.json` |
| `pg_lite_model_artifact` | `true` | `true` | `records/local_proof/pg_lite/model.pglite.bin` |
| `pg_lite_suite_summary` | `true` | `true` | `records/local_proof/pg_lite_suite/summary.json` |
| `backend_check` | `true` | `true` | `records/local_proof/backend_cpu_reference.json` |
| `backend_check` | `true` | `true` | `records/local_proof/backend_metal_apple.json` |
| `proposal_feature_report` | `true` | `true` | `records/local_proof/proposal_features.json` |
| `proof_bundle_report` | `true` | `true` | `records/local_proof/proof_bundle.json` |
| `human_readme` | `true` | `true` | `records/local_proof/README.md` |
| `evidence_manifest` | `true` | `true` | `records/local_proof/evidence_manifest.json` |

## Entry Points

- Bundle directory: `records/local_proof`
- Proof bundle JSON: `records/local_proof/proof_bundle.json`
- Evidence manifest: `records/local_proof/evidence_manifest.json`
- Machine summary: `records/local_proof`

This README is generated by `pg-local proof-bundle`; edit the source command or inputs, then regenerate rather than hand-editing this file.
