# Non-record submission: Rust/CUDA record-shaped systems stack

This is a non-record systems submission, not a leaderboard record claim.

## Summary

This PR adds and documents a Rust/CUDA Parameter Golf stack that can execute the real record-shaped 8xH100 training surface:

```text
world_size             = 8
seq_len                = 2048
global_batch_tokens     = 786432
local_batch_sequences   = 48 per rank
attention_backend       = cuDNN frontend BF16 SDPA
optimizer target        = sharded Parallel Muon
```

The best clean measured record-shaped proxy result is:

```text
timing_measured_ms_per_step = 256.787
```

This is not competitive with the current leaderboard target of roughly <=130 ms/step, and I am not claiming a full-validation BPB.

## Why submit it

The stack closes the original catastrophic systems gap from a 91-second/step real record run to a measured 256.8 ms/step record-shaped runtime, while preserving explicit audit fields for record semantics, attention backend, distributed optimizer, output CE path, SmearGate legality, and TTT legality.

It also documents negative results that should be useful:

- Tiled output CE was slower because it repeated output GEMMs.
- BF16 attention backward tail regressed until the downstream BF16 QKV gradient path is complete.
- Compact u16 input upload regressed because H2D was already sub-millisecond.
- Fast TF32 regressed on this workload.

## What is included

```text
records/track_non_record_16mb/2026-04-30_RustCudaSystems/
  README.md
  TECHNICAL_REPORT.md
  PR_BODY.md
  submission.json
  specs/frontier_1855_merged_target.toml
  scripts/exact_modal_commands.sh
  logs/
  artifacts/artifact_budget.json
```

## Validation

Local validation passed:

```text
cargo check -q --features cuda -p pg-train -p pg-eval -p pg-data -p pg-kernels
cargo test -q -p pg-data
cargo test -q --features cuda -p pg-eval
cargo test -q --features cuda -p pg-train
python3 -m py_compile deploy/run_detached.py deploy/build_submission.py
```

## Remaining blockers

The remaining blockers for a real leaderboard submission are:

- Complete BF16 backward activation graph.
- Add bucketed backward/NCCL overlap.
- Replace chunked BF16 CE cache with production fused projection + softcapped CE/backward.
- Implement a fully GPU-resident sampler.
- Prove full legal distributed eval/TTT under 600 seconds.
- Produce final artifact/code-byte proof.
- Run 3 full validation seeds.

