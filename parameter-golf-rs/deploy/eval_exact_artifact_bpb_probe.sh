#!/usr/bin/env bash
set -euo pipefail

# Credit-efficient BPB diagnostic for an already-exported exact-gradient artifact.
# This does not retrain. It uses the eval-only compatibility spec so pg-eval
# builds with older runtime enum support can still score the artifact.
#
# Defaults to a one-H100, capped-token diagnostic. Set PG_EXACT_EVAL_TOKENS=0
# for full validation once credits/capacity are available.

cd "$(dirname "$0")/.."

artifact="${PG_EXACT_ARTIFACT:-/output/frontier_2135_audit_quality_debug_v1.pgrs}"
result_json="${PG_EXACT_EVAL_RESULT_JSON:-/output/frontier_2135_exact_artifact_eval_compat_262k_v1.json}"
eval_tokens="${PG_EXACT_EVAL_TOKENS:-262144}"
eval_gpus="${PG_EXACT_EVAL_GPUS:-1}"

extra_build_flag="--reuse-cargo-cache"
if [[ "${PG_EXACT_EVAL_FORCE_CLEAN:-0}" == "1" ]]; then
  extra_build_flag="--force-cargo-clean"
fi

args=(
  --modal-wait
  eval
  "${extra_build_flag}"
  --spec /specs/frontier_2135_exact_artifact_eval_compat.toml
  --artifact "${artifact}"
  --eval-gpu-world-size "${eval_gpus}"
  --result-json "${result_json}"
)

if [[ "${eval_tokens}" != "0" ]]; then
  args+=(--max-tokens "${eval_tokens}")
fi

PG_WAIT=1 ../.venv/bin/modal run deploy/run_detached.py "${args[@]}"
