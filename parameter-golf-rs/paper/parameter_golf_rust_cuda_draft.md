# Systems-Level Language Model Compression Under 16 MB and 600 Seconds

Cedric Haddad

## Abstract

We study whether a systems-level Rust+CUDA training stack can make qualitatively different design points viable in OpenAI's Parameter Golf setting: a language model must fit within a strict 16,000,000 byte submission artifact and train/evaluate under fixed 600 second budgets on 8xH100. The implementation replaces the Python/PyTorch training loop with an explicit Rust workspace, preallocated CUDA memory, cuDNN BF16 SDPA, cuBLASLt GEMMs, NCCL distributed training, graph-captured backward/update regions, device-resident sampling, and spec-owned audit gates. The best currently substantiated leaderboard-clean systems profile measures 133.295 ms/step in a record-shaped 8xH100 proxy run. A pre-fix speed probe reached 119.36 ms/step while exporting a 15,640,347 byte artifact, but code review showed that its "pass1 straight-through" setting skipped a whole recurrent-layer backward; that result is now treated as diagnostic only. The paper therefore separates measured systems throughput from unvalidated quality claims and identifies the remaining exact recurrent-backward work required to beat the current audited Parameter Golf frontier.

## 1. Motivation

Parameter Golf makes framework and kernel overhead first-order. The training budget is short enough that saving tens of milliseconds per step can materially change the number of optimizer updates available inside 600 seconds. The top public frontier has moved beyond the original proposal's 1.12 BPB context: the current target is the PR #2135 lineage at about 1.05651 BPB, using clean canonical CaseOps data, GPTQ calibration batches 32, AsymLogit, token-only n-gram tilt, and PR #2135 TTT settings.

This project asks two questions:

1. Can a Rust+CUDA stack reproduce or exceed the current frontier under the same artifact and wall-clock constraints?
2. Which improvements are attributable to systems throughput, and which require new algorithmic or quantization choices?

## 2. Implementation Stack

The implementation is a Rust workspace with separate crates for tensor/runtime primitives, CUDA kernels, model execution, optimization, quantization, evaluation, and compatibility tooling. The record path uses:

- cuDNN BF16 SDPA forward/backward with prepacked BF16 attention tensors.
- BF16 direct-compact backward chain driven by TOML/RunSpec instead of environment-only flags.
- Device-resident token-ring batch scheduling with zero per-step host batch construction after warmup.
- Chunked BF16 output cross-entropy cache rather than full-logit materialization.
- Sharded Parallel Muon with graph-captured local update and pre-norm regions.
- Spec-owned runtime fingerprinting for BF16 chain, CUDA graph profile, token-ring data mode, NCCL overlap mode, and recurrent-backward profile.
- Artifact audit events carrying model bytes, code bytes, total bytes, SHA-256 hashes, and strict decimal 16,000,000 byte budget checks.

## 3. Proposal Feature Status

The current code implements several proposal-aligned systems ideas:

| Proposal item | Current state |
| --- | --- |
| Rust workspace and CUDA execution stack | Implemented |
| Explicit backward without PyTorch autograd | Implemented |
| Device-resident record data path | Implemented for record-shaped proxy timing |
| CUDA graph capture | Implemented for the no-scalar-loss backward region and sharded Muon local/pre-norm regions; full train-step graph capture is not implemented |
| BF16 hot backward chain | Spec-owned and audit-visible |
| Quantization proc-macro compiler | Partially implemented as a constant-layout macro plus compiled layout audit gates; a real layout compiler is not complete |
| BigramHash embedding merge | Forward and backward merge implemented and spec-gated; H100 parity/perf evidence still pending |
| SparseAttnGate+XSA BF16 fusion | Implemented adjacent to attention with warp-head/grouped-KV variants |
| Fused QK/RoPE/q-gain and combined QKV/RoPE tail | Implemented; combined tail is enabled in the #2135 audit candidate |
| Exact recurrent pass boundary fusion | Implemented narrowly and now spec-gated in the #2135 audit candidate; it is not the persistent-CTA block backward |
| Spec-owned QKV norm/resid reducer profile | Implemented; direct/split/chunked compact reducers are audit-visible and fingerprinted |
| True register-resident XSA inside SDPA | Not implemented |
| Persistent-CTA per-block backward megakernel | Not implemented |
| Full compliant BPB validation | Not complete |
| Winning-stack reproduction/beat | Not demonstrated |

## 4. Measured Systems Results

The strongest leaderboard-clean timing evidence so far is from `frontier_2135_audit_target.toml` before the strict recurrent-boundary-fusion default was enabled:

| Metric | Value |
| --- | ---: |
| mode | record-shaped-proxy |
| backend | cuda-distributed |
| profile | `--frontier-graph-record-profile` |
| measured step time | 137.016774 ms/step |
| train step time | 136.963445 ms/step |
| steps completed | 4,303 |
| wall clock | 600.096314 s |
| CUDA backward graph region | 121.958926 ms/step |
| sharded bank update | 10.341131 ms/step |
| non-bank update | 2.258918 ms/step |
| host batch flatten calls | 0 |
| host-to-device batch bytes | 0 |
| device-to-host scalar reads | 0 |

The strongest pre-fix speed-probe timing evidence was from `frontier_2135_hybridst1_budget_target.toml`:

| Metric | Value |
| --- | ---: |
| mode | record-shaped-proxy |
| backend | cuda-distributed |
| profile | `--frontier-graph-record-profile` |
| measured step time | 119.359958 ms/step |
| train step time | 119.328145 ms/step |
| steps completed | 4,952 |
| wall clock | 600.113156 s |
| CUDA backward | 108.169779 ms/step |
| sharded bank update | 8.276393 ms/step |
| non-bank update | 1.284700 ms/step |
| host batch flatten calls | 0 |
| host-to-device batch bytes | 0 |
| device-to-host scalar reads | 0 |
| compressed model artifact | 12,963,935 bytes |
| code bytes | 2,676,412 bytes |
| total artifact budget | 15,640,347 / 16,000,000 bytes |

This is no longer a valid systems milestone. Code review found that `recurrent_backward_profile = "pass1_straight_through"` with `recurrent_straight_through_layers = 1` was entering a whole-layer straight-through branch before the pass1/pass2 recurrent logic. The implementation now scopes pass1 straight-through to pass1 only and preserves pass2/backward for the selected layer; the old 119.36 ms number remains useful only as a diagnostic upper bound for how much recurrent replay can cost.

More quality-plausible profiles are slower:

| Profile | Recurrent treatment | Measured result |
| --- | --- | ---: |
| `frontier_2135_audit_target.toml` strict boundary-fusion v3 | full recurrent backward, exact boundary fusion enabled, force-clean rebuild | 137.734750 ms/step |
| `frontier_2135_audit_target.toml` graph profile, latest clean run | full recurrent backward, leaderboard-clean audit profile, combined QKV/RoPE tail enabled | 137.016774 ms/step |
| `frontier_2135_audit_target.toml` | full recurrent backward, leaderboard-clean audit profile, combined QKV/RoPE tail enabled | 137.384528 ms/step |
| `frontier_2135_audit_target.toml` graph-side GEMM capture | full recurrent backward plus graph-side dW overlap capture | 136.906285 ms/step |
| `frontier_2135_audit_target.toml` runtime sweep | full recurrent backward plus grouped-KV XSA, fused Muon global clip, parallel local Muon | 137.829721 ms/step |
| `frontier_2135_audit_target.toml` chunked compact QKV norm/resid reducer | full recurrent backward, exact alternative reducer | 137.439895 ms/step |
| `frontier_2135_audit_target.toml` split compact QKV norm/resid reducer | full recurrent backward, exact alternative reducer | 156.909543 ms/step |
| `frontier_2135_pass1_budget_target.toml` | pass-1 straight-through, no layer skip | 125.877317 ms/step |
| `frontier_2135_flowgrad_budget_target.toml` | full recurrent activation gradients, skip recurrent bank gradients | 135.572154 ms/step |
| `frontier_2135_hybridst1_budget_target.toml` pre-fix | unintentionally skipped one whole recurrent layer backward | 119.359958 ms/step |

The clean audit profile is therefore still about 17-18 ms/step above the 120 ms target. The strict boundary-fusion v3 run also exposed a data-readiness blocker: the mounted Modal validation shard had 40,547,886 tokens, not the PR #2135 canonical 47,851,520 tokens, so the audit now marks it non-canonical. The runtime sweep did not help: grouped-KV XSA and fused/parallelized sharded Muon shifted optimizer timing but left the measured step time essentially unchanged. Graph-side GEMM capture is exact but did not win reliably: the older run measured 136.91 ms/step, and a fresh explicit proxy run measured 136.78 ms/step while increasing optimizer-side timing. Alternative exact QKV norm/resid reducers did not close the gap; the direct compact reducer remains the best validated clean reducer. Backward NCCL bucket overlap with two-layer buckets emitted per-step bucket telemetry but hit the 3600s Modal timeout before finishing, so it remains disabled.

A dedicated stage-timing run on the clean audit target completed 2,200 steps and measured 144.546027 ms/step in instrumentation mode. The stage attribution was:

| Stage | ms/step |
| --- | ---: |
| full graph-equivalent backward region | 123.597 |
| forward cache replay | 42.054 |
| MLP backward | 20.305 |
| QKV backward total | 21.080 |
| QKV RoPE/q-gain tail | 7.465 |
| QKV norm/resid reducer | 9.911 |
| attention SDPA backward | 13.712 |
| attention output/gate/XSA backward | 12.567 |
| output CE/backward | 7.225 |
| exact recurrent pass2 backward replay | 3.807 averaged over all timed steps |
| exact recurrent pass1 backward replay | 3.812 averaged over all timed steps |
| sharded bank update | 15.198 |
| non-bank update | 3.430 |

The refreshed graph run gives the cleanest active/inactive split: inactive recurrence is 120.53 ms/step, while active recurrence is 147.87 ms/step. In the stage probe, recurrence starts late enough that the recurrent pass counters average only 3.81 ms/step over all timed steps; normalized over active recurrent steps they are about 18 ms/pass. The open timing question is no longer whether dormant runtime flags can close the gap; the missing work is exact recurrent backward fusion, a persistent-CTA block backward implementation that materially reduces active recurrent block cost, or BPB validation that a recurrent straight-through approximation is acceptable as a new algorithmic variant.

## 5. Quality Status

The current BPB evidence is intentionally not treated as meaningful. A 262k-token proxy artifact evaluation produced poor BPB because the artifact came from the record-shaped timing path rather than a full compliant train/eval/export run. The remaining quality gate is:

1. Run a full record train/eval/export path with canonical validation data and CaseOps sidecar bytes.
2. Validate score-first TTT, token-only n-gram tilt, AsymLogit, and GPTQ calibration batches 32 in the audit logs.
3. Run at least three seeds.
4. Compare against the PR #2135 frontier and report mean/std BPB.

An attempted full `record` run on the sub-120 `hybridst1` speed probe was correctly rejected before training because record mode now requires `leaderboard_algorithm_ready=true`, and `frontier_2135_speed_probe` profiles are diagnostic-only. This prevents a recurrent straight-through timing profile from being presented as a leaderboard-clean record.

## 6. Ablation Plan

The paper should report ablations on the same spec family and data path:

| Ablation | Purpose |
| --- | --- |
| BF16 direct-compact off/on | isolate hot backward precision chain |
| host batch path vs full-schedule token ring | isolate data orchestration removal |
| eager vs CUDA graph | isolate launch/orchestration reduction |
| sharded Muon local graph off/on | isolate optimizer launch collapse |
| combined QKV/RoPE tail off/on | isolate proposal-aligned QKV tail fusion |
| direct vs chunked/split compact QKV norm/resid reducer | document exact reducer choice and failed alternatives |
| recurrent full vs pass1-ST vs hybrid-ST1 | measure speed/quality tradeoff |
| #2014 BigramHash merge off/on | isolate embedding-memory fusion |
| compiled quant layout variants | isolate quantization search once BPB is valid |

## 7. Remaining Work Before Final Claims

The project is not 100% complete until the following are closed:

- Full leaderboard-clean train/eval/export run succeeds under budget.
- BPB is validated on full validation data and at least three seeds.
- A sub-120 profile is either made leaderboard-clean or its recurrent approximation is shown to preserve BPB under the corrected pass1-ST semantics.
- Artifact budget audit is demonstrated in the same full record run, not only in proxy export.
- NCCL overlap is either measured with CUDA event windows and enabled, or explicitly disabled as non-winning.
- Persistent-CTA backward megakernel is implemented and measured if the paper claims it.
- True XSA-inside-SDPA is implemented and measured if the paper claims it.
- The quantization proc-macro story is expanded from compiled layout declarations to generated pack/dequant/STE kernels, or the paper narrows the claim to compiled layout specialization.

## 8. Current Conclusion

The Rust+CUDA stack has demonstrated a leaderboard-clean exact profile at 133.295 ms/step with zero hot-path host batch construction, and a pre-fix diagnostic profile below 120 ms/step that exposed the cost of recurrent replay. It has not yet demonstrated a corrected sub-120 leaderboard-clean profile or a winning BPB. The principal remaining risk is algorithmic and architectural: the exact recurrent path remains above the current 120 ms/step systems target even after graph capture, graph-side GEMM, QKV reducer sweeps, and a negative NCCL bucket-overlap probe. The next decisive systems experiment must be a true exact recurrent backward fusion or persistent-CTA block backward that reduces active recurrent block cost; the next decisive quality experiment is full BPB for the fastest defensible profile.
