#!/usr/bin/env bash
set -euo pipefail

# Run from parameter-golf-rs/.

# Historical clean record-shaped timing run.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v86_throughput_clean.pgrs \
  --result-json /output/frontier_v86_throughput_clean.json \
  --frontier-throughput-record-profile

# Current #2135-shaped speed-probe timing run. Use the direct function entrypoint
# so the job does not get lost in local-entrypoint spawn semantics.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst_pass1cache_v7.pgrs --result-json /output/frontier_2135_hybridst_pass1cache_timing_v7.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --force-cargo-clean"

# Pre-fix <=120 ms diagnostic profile. The old timing proof was 119.36 ms/step,
# but code review found the pass1-ST path skipped a whole recurrent layer
# backward. Rerun only as a corrected semantics guardrail, not as a clean claim.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst1_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst1_export_v1.pgrs --result-json /output/frontier_2135_hybridst1_export_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --export-record-shaped-artifact --force-cargo-clean"

# Leaderboard-clean #2135 audit timing candidate. This keeps full recurrent
# backward and enables the combined QKV/RoPE tail plus exact recurrent boundary
# fusion from the TOML. Force-clean v3 measured 137.735 ms/step with zero
# measured BF16 bridge wrapper launches. It did not close the active-recurrent
# gap: active 148.852 ms/step, inactive 120.963 ms/step. The mounted Modal
# validation shard is not canonical PR #2135 data (40,547,886 tokens rather
# than 47,851,520), so full record mode must not be run against that dataset.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
	  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_boundary_fusion_v3.pgrs --result-json /output/frontier_2135_audit_boundary_fusion_v3.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --force-cargo-clean"

# Exact recurrent-boundary reducer cleanup. This replaces the fused pass-1
# MLP-scale per-element atomics with the normal chunked BF16 scale reducer and
# aliases pass-1 grad_x_after_attn to grad_mid to avoid a duplicate full-tensor
# write. It is correctness-preserving but not decisive: full 600s proxy measured
# 137.756 ms/step, active 149.565 ms/active-step, inactive 120.047
# ms/inactive-step, bridge launches 0. Treat as a cleanup, not the 120 ms cut.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_boundary_chunked_alias_v1.pgrs --result-json /output/frontier_2135_audit_boundary_chunked_alias_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --record-timing-skip-steps 64 --force-cargo-clean"

# One-layer pass1-ST speed comparator on the same tree. This is still not a
# leaderboard-clean algorithm, and it also does not hit 120 ms: 130.762 ms/step,
# active 138.660 ms/active-step, inactive 117.758 ms/inactive-step.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst1_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst1_latest_probe_v1.pgrs --result-json /output/frontier_2135_hybridst1_latest_probe_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --record-timing-skip-steps 64 --force-cargo-clean"

# A/B guardrail for the old exact profile without recurrent boundary fusion.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_no_boundary_fusion_v1.pgrs --result-json /output/frontier_2135_audit_no_boundary_fusion_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --disable-recurrent-boundary-fusion --force-cargo-clean"

# Stage attribution for the exact clean #2135 audit profile. Latest
# instrumentation mode measured 144.546 ms/step over 2,136 timed steps.
# Dominant substages: forward replay 42.054, QKV 21.080, MLP 20.305, SDPA
# backward 13.712, attn-out/gate/XSA 12.567, QKV norm/resid 9.911 ms/step.
# Recurrent pass2/pass1 each cost ~3.81 ms/all timed step, or ~18 ms/active
# recurrent step in this late-activation probe.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_recurrent_stage_probe_v2.pgrs --result-json /output/frontier_2135_audit_recurrent_stage_probe_v2.json --frontier-throughput-stage-profile --record-shaped-proxy-max-steps 2200 --force-cargo-clean"

# Exact graph-side GEMM capture A/B. It remained leaderboard-clean but did not
# win reliably: the older run measured 136.91 ms/step, while the latest explicit
# run measured 136.78 ms/step versus the current best exact clean 133.30.
# Keep as an A/B flag, not as the audit-target default.
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_graph_side_explicit_v4.pgrs --result-json /output/frontier_2135_audit_graph_side_explicit_v4.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --enable-graph-side-gemm-capture"

# Exact QKV norm/resid reducer A/B. Chunked compact measured 137.44 ms/step
# and split compact regressed to 156.91 ms/step; direct compact remains default.
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_chunked_qkv_norm_v1.pgrs --result-json /output/frontier_2135_audit_chunked_qkv_norm_timing_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --bf16-backward-chain-qkv-norm-resid-reducer chunked_compact --qkv-norm-resid-bwd-rows-per-chunk 1024"
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_split_qkv_norm_v1.pgrs --result-json /output/frontier_2135_audit_split_qkv_norm_timing_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --bf16-backward-chain-qkv-norm-resid-reducer split_compact --qkv-norm-resid-bwd-rows-per-chunk 1024"

# Latest active-recurrence cut sweep. These are speed probes, not leaderboard
# claims. Exact activation flow with pass-1 bank grads dropped still measured
# 124.52 ms/step; dropping all recurrent bank grads measured 124.31 ms/step.
# Hybrid pass1-ST across all three recurrent layers measured 123.65 ms/step.
# The only <=120 run so far is the all-ST diagnostic with combined QKV/RoPE tail
# and standalone shard clipping. Short probe: 116.945 ms/step over 2,136 timed
# steps. Full 4,994-step proxy: 113.477 ms/step over 4,930 timed steps, 576.64s
# wall-clock, zero BF16/F32 bridge launches, zero hot-path host batch work. It
# is useful as a systems floor, but not BPB/record evidence.
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_full_skip_pass1bank_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_skip_pass1bank_boundary_short_v1.pgrs --result-json /output/frontier_2135_skip_pass1bank_boundary_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_flowgrad_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_flowgrad_boundary_short_v1.pgrs --result-json /output/frontier_2135_flowgrad_boundary_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst3_short_v1.pgrs --result-json /output/frontier_2135_hybridst3_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_allst_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_allst_combinedtail_short_v2.pgrs --result-json /output/frontier_2135_allst_combinedtail_short_v2.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_allst_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_allst_combinedtail_full_v1.pgrs --result-json /output/frontier_2135_allst_combinedtail_full_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --record-timing-skip-steps 64 --force-cargo-clean"

# Full record mode intentionally rejects speed-probe profiles unless the
# algorithm gaps are closed. This command is retained as a guardrail check:
# it fails before training with leaderboard_algorithm_ready=false.
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst1_budget_target.toml --mode record --backend cuda-distributed --artifact /output/frontier_2135_hybridst1_full_record_v1.pgrs --result-json /output/frontier_2135_hybridst1_full_record_v1.json --frontier-graph-record-profile"

# Negative runtime sweep: grouped-KV SparseAttnGate/XSA backward plus fused
# global clip and parallel local sharded-Muon updates measured 137.83 ms/step,
# slightly worse than the clean audit baseline. Keep this to avoid repeating it.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_audit_runtime_sweep_v1.pgrs --result-json /output/frontier_2135_audit_runtime_sweep_timing_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --enable-sparse-xsa-grouped-kv-bwd --enable-sharded-muon-fused-global-clip --enable-sharded-muon-parallel-local"

# Proxy artifact BPB sanity only. This is intentionally not leaderboard-quality
# evidence because record-shaped proxy disables training loss; use a real record
# run for final BPB.
/tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait eval \
  --ttt-audit \
  --assert-score-no-mutation \
  --spec /specs/frontier_2135_hybridst1_budget_target.toml \
  --artifact /output/frontier_2135_hybridst1_export_v1.pgrs \
  --max-tokens 262144 \
  --eval-gpu-world-size 1 \
  --result-json /output/eval_frontier2135_hybridst1_262k_v1.json \
  --force-cargo-clean

# Regressed compact-u16 upload A/B.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v87_u16_shift.pgrs \
  --result-json /output/frontier_v87_u16_shift.json \
  --frontier-throughput-record-profile \
  --enable-shifted-u16-batch-upload

# Artifact export proof command. This did not complete before submission due
# to Modal connectivity failure.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v90_export_probe.pgrs \
  --result-json /output/frontier_v90_export_probe.json \
  --frontier-throughput-record-profile \
  --disable-shifted-u16-batch-upload \
  --export-record-shaped-artifact \
  --submission-code-bytes 1

# Local validation commands.
cargo check -q --features cuda -p pg-train -p pg-eval -p pg-data -p pg-kernels
cargo test -q -p pg-data
cargo test -q --features cuda -p pg-eval
cargo test -q --features cuda -p pg-train
python3 -m py_compile deploy/run_detached.py deploy/build_submission.py

# Frontier #2135 canonical data preflight. This must pass before any full
# mode=record train/eval/export can be considered compliant. Latest clean
# preflight reaches the real data audit and fails for the data-mount reason:
# train_shards=80 and val_docs=50,000 are correct, but val_tokens=40,541,268
# versus the canonical PR #2135 requirement of 47,851,520.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::preflight_caseops_string \
  --args "preflight-caseops --spec /specs/frontier_2135_audit_target.toml --result-json /output/frontier_2135_caseops_preflight_latest.json"

# Data-seed guard. The run_detached image now installs huggingface_hub and the
# seeder checks token counts instead of accepting any existing SP8192 files. The
# official willdepueoai/parameter-golf repo does not publish fineweb10B_sp8192 at
# this path, so the seeder falls back to Austin362667/fineweb10B_sp8192 for proxy
# timing only, prunes to 80 train shards, regenerates CaseOps sidecars, and then
# fails closed because val_tokens=40,541,268 != 47,851,520.
/tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py::seed_data

# Updated preflight after the seed-data repair. This archives the fail-closed
# canonical data blocker against the restored 80-shard proxy mount.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::preflight_caseops_string \
  --args "preflight-caseops --spec /specs/frontier_2135_audit_target.toml --result-json /output/frontier_2135_caseops_preflight_after_seed_v1.json"

# Latest exact-gradient #2135 active-recurrence evidence after promoting
# recurrent_backward_profile=exact_fused and fusing the non-parallel MLP-norm
# add into RMSNorm backward. This is still over the 120 ms target:
# timing_measured_ms_per_step=127.578, active=156.311, inactive=119.845.
/tmp/pg-modal-venv312/bin/modal run --detach \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_exactfused_accum_short_v1.pgrs --result-json /output/frontier_2135_exactfused_accum_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"

# Exact-gradient #2135 after routing the attention-output projection dX through
# BF16 directly into the compact SparseAttnGate/XSA backward consumer. The new
# audit field proposal_sparse_xsa_attn_proj_dx_bf16_fusion_active=true and bridge
# launches remain zero. This is correct but still not decisive:
# timing_measured_ms_per_step=127.110, active=155.655, inactive=119.427.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_sparsexsa_bf16dx_short_v1.pgrs --result-json /output/frontier_2135_sparsexsa_bf16dx_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --force-cargo-clean"

# Graph-side GEMM capture remains a negative A/B on top of the BF16 XSA dX cut:
# timing_measured_ms_per_step=127.994, worse than the clean 127.110 short run.
# Keep graph_side_gemm_capture=false in the audit spec.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_audit_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_sparsexsa_bf16dx_graphside_short_v1.pgrs --result-json /output/frontier_2135_sparsexsa_bf16dx_graphside_short_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 2200 --record-timing-skip-steps 64 --enable-graph-side-gemm-capture"
