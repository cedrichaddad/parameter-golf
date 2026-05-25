#!/usr/bin/env bash
set -euo pipefail

# Credit-efficient BPB diagnostic:
# - full fast train schedule from the all-ST quality probe spec
# - official-quality PR #2135 export bits (6/6/7), not the budget int5/int4 export
# - capped validation so the run answers "is BPB still catastrophically bad?"
# - persistent Cargo target cache reuse to avoid paying for cargo clean
#
# This is intentionally not leaderboard evidence. The spec is marked
# allow_unsupported_variants=true and record_profile=frontier2135_speed_probe.
#
# If the remote target cache predates the current enum/spec surface, first run
# the no-GPU preflight below after the Modal billing limit is available:
#
#   ../.venv/bin/modal run deploy/run_detached.py --modal-wait preflight-caseops \
#     --force-cargo-clean \
#     --spec /specs/frontier_2135_eval_compat_target.toml \
#     --result-json /output/frontier2135_quality_probe_preflight.json

cd "$(dirname "$0")/.."

extra_build_flag="--reuse-cargo-cache"
if [[ "${PG_BPB_PROBE_FORCE_CLEAN:-0}" == "1" ]]; then
  extra_build_flag="--force-cargo-clean"
fi

PG_WAIT=1 ../.venv/bin/modal run deploy/run_detached.py --modal-wait --multi \
  "${extra_build_flag}" \
  --frontier-graph-record-profile \
  --spec /specs/frontier_2135_eval_compat_target.toml \
  --mode record \
  --backend cuda-distributed \
  --eval-max-tokens "${PG_BPB_PROBE_EVAL_TOKENS:-262144}" \
  --result-json /output/frontier2135_allst_quality_probe_bpb.json
