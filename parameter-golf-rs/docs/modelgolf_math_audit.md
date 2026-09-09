# ModelGolf Math Audit

This audits the mathematical claims in the ModelGolf plan against the current
Rust implementation. The goal is to keep product claims tied to exact
assumptions and to make approximation boundaries explicit.

## 1. Mixed-Precision Pack

Claim: with additive quality loss across tensor groups, selecting one precision
option per group under a byte budget is a multiple-choice knapsack problem.

Status: correct under the additive surrogate assumption.

Implementation: `pg-local modelgolf pack` now uses an exact sparse dynamic
program over integer byte estimates. Dominated states are pruned only when a
lower-or-equal-byte state has lower loss, which preserves the optimum, and a
brute-force regression checks the sparse DP against exhaustive enumeration on a
small multiple-choice instance. The planner can also ingest measured
quality-loss calibration points through `--quality-calibration`; those points
scale group/role bit-loss estimates before the exact DP runs and are reported
with before/after fit error. LQER candidate reporting now distinguishes
requested rank from mathematically effective rank `min(r, m, n)`, disables
candidates when LQER is disabled, and limits the quality correction to that
effective rank. The Pack report also emits `quality_comparison`, a
deterministic equal-byte local proxy table covering uniform Q4, best uniform
no-LQER, exact mixed no-LQER, selected mixed+LQER, and a deterministic LQER
control/fallback. The proxy uses synthetic tensor fixtures, rowwise quantization
residuals, identity-curvature low-rank corrections where LQER is present,
weighted residual MSE, and the same activation/logit CE-bound style used by the
LQER local proofs.
`pg-local modelgolf pack-experiment` now implements the plan's concrete LQER++
table on a bounded deterministic small model. It exports, strict-reloads, and
scores uniform Q4, mixed Q4/Q6 without LQER, mixed+LQER top-3, and a
deterministic random-LQER control, recording artifact bytes, local BPB proxy,
decode-speed smoke timing, and the selected LQER groups embedded in artifact
metadata.

Remaining evidence: the quality loss model is still a surrogate unless the
calibration points come from representative evals. Release claims need held-out
calibration against perplexity/BPB or task loss on pretrained small models and
the final trained artifact.

## 2. LQER Low-Rank Residuals

Claim: the best rank-r residual correction under Frobenius loss is the
truncated SVD of the quantization residual.

Status: correct by Eckart-Young-Mirsky.

Implementation: small residual matrices use an exact symmetric Jacobi
eigendecomposition of `E^T E` or `E E^T` and emit truncated SVD factors. Large
matrices still use deterministic power-iteration deflation for practical export
cost. `lqer.top_k` is now enforced by the artifact exporter: candidate groups
are scored by rank-r captured residual energy per actual requested-rank
artifact byte, with role weighting. This avoids over-crediting high-rank
residuals or undercharging tiny tensors whose artifact stores padded factors.
Selected group names are written to artifact metadata, and the loader requires
LQER tensors only for those selected groups. When a requested artifact rank is
larger than a small tensor's available rank, the exporter pads unused factor
components with zeros so the artifact shape and loader metadata remain
consistent while the mathematical correction remains rank-limited. `pg-local
modelgolf pack` now emits `lqer_proofs` for selected LQER options. These proofs
build deterministic quantization residuals, compute identity-curvature low-rank
factors, materialize the correction, and report residual Frobenius reduction
plus the conservative activation/logit CE bound reduction.

Remaining evidence: production LQER++ needs calibrated tensor sensitivity,
larger-matrix exact/randomized SVD validation, and measured equal-byte
BPB/perplexity comparisons on pretrained small models and real validation data.

## 3. Exact Tiled Cross-Entropy

Claim: tiled max/sum over disjoint vocabulary tiles computes the same
mathematical logsumexp CE as full logits, and gradients can be accumulated per
tile without persistent `M x V` logits.

Status: correct mathematically, assuming the same softcap transform and
deterministic arithmetic contract. Floating-point bitwise equality is not
guaranteed unless operation order is matched.

Implementation: `pg-kernels::cross_entropy` includes a CPU reference for exact
tiled output projection plus softcapped CE. It streams vocabulary tiles, avoids
persistent `M x V` logits/grad-logits, and tests loss, `dH`, and `dW` against
the full-logit reference. `pg-local modelgolf kernel-plan` now also emits
`exact_tiled_ce.local_proof`, a deterministic CPU parity proof that compares
tiled CE loss, `d_hidden`, and `d_weight` against full-logit CE plus explicit
chain-rule gradient accumulation. `pg-local modelgolf kernel-experiment` runs
the plan's concrete Experiment 2 harness: full-logit CE vs tiled logit-free CE,
loss/`dH`/`dW` parity, CE scratch bytes, scratch reduction, and local CPU
timing on a bounded deterministic fixture. Scratch reduction is accounted as
persistent `M x V` logits/grad-logits versus tiled logits plus per-row max/sum
stats; it does not count live hidden or lm-head tensors that both paths require.

Remaining evidence: lower the same contract to CUDA/Metal kernels, then test
backend loss, `dH`, and `dW` parity plus memory/runtime behavior.

## 4. CacheGolf KV Error Bound

Claim: for single-query attention, quantized K/V output error is bounded by
value error plus a key-error term scaled by query norm and value norm.

Status: directionally correct. The softmax Lipschitz constant used in the plan
is a conservative bound from `l_inf` score perturbation to `l_1` probability
perturbation.

Implementation: current CacheGolf enumerates K/V bit policies, block sizes, and
layouts using this bound as a planner score. `pg-kernels::cachegolf` now
implements the local packed cache format: K is symmetric per-channel quantized,
V is symmetric per-token quantized, contiguous layout uses token-major payloads,
and paged layout allocates whole token pages with page-major/channel-major
payload order plus explicit page-table bytes. The CPU reference attention path
dequantizes on the fly without materializing a full F32 KV cache. Tests compare
the fused dequant-attention reference against attention over explicit
dequantized K/V and check that the perturbation bound covers observed output
error on deterministic fixtures. `pg-local modelgolf cache-plan` now also emits
a selected-policy proof: it quantizes deterministic K/V with the chosen policy,
records actual packed single-layer runtime bytes and reconstruction errors,
compares packed-cache attention against full-F32 attention, checks packed
attention against explicit-dequant attention, and reports whether the
perturbation bound covers the observed output error. The report also emits
deterministic planner outputs for residual sketches, sink/recent eviction, and
long-context memory scaling. Those rows use the same selected byte model and
attention-error bound to show whether the requested and larger contexts fit the
target cache budget.
`pg-local modelgolf cache-experiment` now implements the plan's concrete
Experiment 3 harness: it runs a 25-row K/V bit-grid, measures observed
attention-output error, verifies explicit-dequant parity, checks perturbation
bound coverage for every row, selects the lowest-error row that fits 35% of
FP16 cache bytes for the requested context/batch, and emits deterministic
long-context memory extrapolations for that measured policy.

Correction made during implementation: the planner byte model now charges V
scales per token rather than per block, matching the stated per-token value
quantization policy.

Remaining evidence: lower the same format to fused CUDA/Metal
dequant-attention kernels, then measure bound tightness and full long-context
quality/timing on representative workloads.

## 5. DeltaGolf Weighted Low-Rank Deltas

Claim: under a Kronecker-factorized curvature approximation, the optimal
rank-r delta is the truncated SVD of the whitened full update.

Status: correct when curvature factors are positive definite and the local
quadratic approximation is valid.

Implementation: current DeltaGolf uses exact sparse byte allocation over
available delta families, with a brute-force regression that checks the sparse
DP against exhaustive multiple-choice enumeration on a small budgeted instance.
`pg-kernels::deltagolf` now implements the weighted low-rank CPU reference for
diagonal Kronecker curvature factors: it whitens a full update by positive
input/output curvature diagonals, computes the exact truncated SVD, unwhitens
into LoRA-style `A`/`B` factors, and tests exact rank-limited reconstruction,
identity-curvature top-component behavior, rank-monotone weighted error, and
rejection of non-positive curvature.
`pg-local modelgolf delta-plan` now emits selected low-rank local proofs for
rank-bearing selected deltas: deterministic synthetic full updates and positive
curvature diagonals are passed through the same weighted low-rank path, the
factorized delta is materialized, and weighted error is checked against zero and
lower-rank baselines. The same report now emits a static
`score_first_legality_audit` over selected delta metadata and
`domain_evaluation`, a deterministic equal-byte proxy table comparing no-delta,
selected, best-single, low-rank-only, and static-control plans by declared
domain gain and local low-rank weighted-error proxies.

Remaining evidence: estimate full updates and curvature proxies from real
domain data, train LoRA/adapter deltas under the byte contract, record runtime
score-first ordering traces or obtain formal competition/legal review where
needed, and show real equal-byte domain BPB/perplexity improvements.

## 6. Artifact-Aware Training

Claim: if `L` is smooth, then the export gap is bounded by
`||grad L(W)|| * ||Q(W)-W|| + L_s ||Q(W)-W||^2 / 2`.

Status: correct by the smoothness upper bound plus Cauchy-Schwarz.

Implementation: current TrainGolf reports the objective and schedule.
`pg-kernels::traingolf` now implements a CPU reference for stop-gradient
quantization-distance regularization against a symmetric per-block quantization
projection, exposes the exact regularizer gradient `2*lambda*(W-Q(W))`, and
checks the smooth export-gap bound on deterministic quadratic fixtures.
`pg-model::GradBuffers::add_artifact_regularization` adds those gradients to
the CPU model gradient buffers with tensor/byte-contract length checks.
`pg-train` wires this into the non-CUDA training loop behind explicit
`[train] artifact_regularization_*` knobs, after microbatch gradient averaging
and before gradient clipping/optimizer updates. `pg-local modelgolf train-plan`
now emits `local_proof`, which verifies on a deterministic fixture that the
regularizer gradient equals the stop-gradient distance objective, one fixed-grid
gradient step reduces quantization distance, the regularized objective composes
correctly, and the smooth export-gap bound covers an observed quadratic export
gap.

Remaining evidence: calibrate the smoothness/gradient proxies on real runs,
add production GPU/backend integration, and compare post-export quality against
post-hoc quantization.

## 7. Sharded Optimizer Equivalence

Claim: reduce-scatter gradients, shard-local update, and all-gather parameters
equals replicated all-reduce update when the update is shard-separable.

Status: correct under shard separability and identical arithmetic/order.

Implementation: `pg-local dist-sim` validates deterministic local math for the
current Muon-style bank updates. `pg-local modelgolf plan` now exposes an
`optimizer_comm` report with ring-collective byte estimates, graph/shadow
contract flags, and a deterministic local reduce-scatter/local-update/all-gather
equivalence proof for shard-separable updates.

Remaining evidence: NCCL/runtime timing, overlap validation, and exact
production distributed update parity.

## 8. KernelForge Fusion Equivalence

Claim: a fused forward/backward pair is equivalent to the unfused chain if the
forward equals function composition and backward implements the composed
Jacobian transpose.

Status: correct by the chain rule.

Implementation: currently contracts and existing reference tests for some local
fusions. The exact tiled CE contract now has report-level memory accounting,
local CPU proof parity, and a bounded experiment table. General spec-to-kernel
generation remains future work.

Remaining evidence: generated kernels plus forward/backward parity and backend
performance tests.

## 9. Training-Time and Energy Cost Model

Claim: ModelGolf can reason about training-time and energy constraints as part
of the same resource contract as bytes, memory, latency, and quality.

Status: correct only as a planning proxy until backed by measured power/runtime
telemetry. The implemented energy value is a nominal-power heuristic multiplied
by the spec's wall-clock training budget; it is not wall energy.

Implementation: `pg-local modelgolf plan` now includes training-time and
nominal-power proxy fields in `resource_contract`, plus a `cost_model` report
with train tokens per budget second, artifact bytes per budget second,
parameter elements per proxy joule, optional latency-budget tokens/second, and
an explicit evidence boundary.

Remaining evidence: measured wall power, thermal behavior, datacenter PUE when
relevant, and profiler-correlated runtime traces on the target hardware.

## 10. Wind Tunnel Candidate Ranking

Claim: cheap planner metrics can rank expensive ModelGolf candidate runs well
enough to decide which full evaluations are worth spending.

Status: a rank-correlation claim is only valid after comparing predicted scores
against held-out measured runs. Spearman correlation is the right first metric
because the planner's job is candidate ordering, not calibrated absolute loss.

Implementation: `pg-local modelgolf wind-experiment` now implements the plan's
local Experiment 5 harness. It generates 24 deterministic ModelGolf candidates
covering Pack bit choices/LQER settings, CacheGolf K/V bit choices, and
DeltaGolf byte budgets. It ranks all candidates by a cheap score built from the
Pack additive quality surrogate, CacheGolf error/memory proxy, and DeltaGolf
predicted gain. It then export/reloads and smoke-scores the top 8 candidates
using the bounded local artifact path, records actual artifact bytes and BPB
proxy, computes a fuller proxy score, and reports Spearman rank correlation,
mean absolute rank error, top-3 overlap, and per-candidate rank errors. The
fuller proxy intentionally uses bounded export/reload for the local BPB delta
but deployment-scale estimated artifact/cache ratios for budget penalties;
otherwise the toy proof artifact would incorrectly reward frontier candidates
that miss the deployment byte contract. Regression tests require 24 generated
candidates, 8 full proxy evaluations, a finite rank-correlation report, and a
direct check that deployment budget overflow worsens the full proxy score even
when bounded artifact bytes are identical. The local report is marked ready
only when Spearman is at least 0.50; otherwise it explicitly reports that Wind
Tunnel ranking needs calibration.

Remaining evidence: replace the bounded export/reload smoke proxy with fresh
profiler traces and held-out full train/eval measurements, then validate
Spearman rank correlation and absolute timing/quality error on unseen candidate
runs before using Wind Tunnel recommendations as release claims.

## 11. ScaleGolf Track Composition

Claim: PG-Lite, ParameterGolf, DeltaGolf, LongContext, and larger-cluster
ScaleGolf can be handled by one platform when every track is expressed through
the same resource contract, model IR, artifact compiler, runtime planner,
evaluator, and cost model.

Status: correct as a software architecture invariant, not as a measured
quality-per-dollar theorem. The invariant proves shared planning vocabulary
across tracks; it does not prove any track is externally optimal or
release-ready.

Implementation: `pg-local modelgolf scale-plan` emits a `scalegolf_track_planner`
report with five rows matching the plan's tracks. Each row records target,
primary constraint, hardware scope, budgets, required modules, local readiness,
remote evidence, and a deterministic quality/resource proxy. The report also
emits an invariant block that checks all rows use the six shared compiler
surfaces. Regression coverage requires all five track names and all six surface
invariants in the full ModelGolf report.

Remaining evidence: per-track measured artifact, quality, runtime,
energy/dollar, distributed communication, kernel, and profiler evidence before
using ScaleGolf as a release claim.

## 12. Pack release source binding

Status: implemented for Pack. `pg-local modelgolf pack-source-report` creates
the `modelgolf_release_source_report` consumed by `release-check`. It verifies
the artifact bytes and SHA-256, strict-reloads the artifact against the current
quant spec, validates finite positive held-out BPB, baseline BPB, and decode
throughput inputs, recomputes

[
\Delta_\mathrm{BPB}\% = 100 \cdot \frac{\mathrm{heldout\_bpb} -
\mathrm{baseline\_bpb}}{\mathrm{baseline\_bpb}},
]

and enforces the quality budget when one is provided. The report is then bound
to the current spec fingerprint and hardware/runtime labels. A regression test
exports a deterministic `.pgrs` artifact, generates the Pack source report, and
passes it through the same release-check SHA/source-report validator used by
release evidence.

Remaining evidence: this command binds and validates measured Pack evidence,
but it does not create the external held-out validation run. Release still
requires real held-out BPB/perplexity and decode-speed measurements from the
target runtime.

## 13. CacheGolf release source binding

Status: implemented for CacheGolf. `pg-local modelgolf cache-source-report`
creates the `modelgolf_release_source_report` consumed by `release-check`. It
reuses the selected CacheGolf policy for the active context/batch/memory
contract, checks that the policy fits the memory target, verifies the local
dequant-attention parity and perturbation-bound proof, and binds external
claims to:

[
(k_\mathrm{bits}, v_\mathrm{bits}, \mathrm{block}, \mathrm{layout},
\mathrm{context}, \mathrm{batch}).
]

The command requires `fused_runtime=true`, `parity_pass=true`, finite
long-context BPB delta, and `speedup_x >= 1`. It also enforces the quality
budget on the long-context BPB delta when a budget is supplied. A regression
test generates the Cache source report and passes it through the same
release-check SHA/source-report validator used by release evidence.

Remaining evidence: this command binds and validates measured CacheGolf
evidence, but it does not execute the external fused dequant-attention backend.
Release still requires target-runtime long-context quality and timing runs.

## 14. Remaining release source bindings

Status: implemented for LQER++, DeltaGolf, TrainGolf, resource/cost evidence,
Optimizer/Comm, KernelForge, and Wind Tunnel. The corresponding commands are:

- `pg-local modelgolf lqer-source-report`
- `pg-local modelgolf delta-source-report`
- `pg-local modelgolf train-source-report`
- `pg-local modelgolf resource-cost-source-report`
- `pg-local modelgolf optimizer-comm-source-report`
- `pg-local modelgolf kernel-source-report`
- `pg-local modelgolf wind-source-report`

Each command creates a `modelgolf_release_source_report` consumed by
`release-check`, binds it to the current spec fingerprint and hardware/runtime
labels, validates a required `--evidence-source` `modelgolf_raw_evidence` JSON
packet, hashes that packet into `claims.raw_evidence`, and validates the same
threshold conditions used by the release gate. `modelgolf release-evidence`
assembles only valid release source reports, requires every release pillar, and
revalidates each report plus its hashed raw evidence packet before emitting the
release evidence JSON. `release-check` re-hashes both the source report and its
raw evidence file, then revalidates the raw packet's `kind`, `pillar`, and
per-pillar measured claim values before accepting the claims.

The LQER report requires production SVD validation, calibrated tensor
sensitivity, a non-positive equal-byte BPB delta, a current Pack plan with
LQER candidates, at least one selected finite LQER proof, and the selected LQER
group names from the current Pack plan.
The Delta report binds the current selected delta names,
requires trained bytes within the selected delta budget, local legality audit
pass, score-first trace or review, and non-positive equal-byte domain BPB
delta. The Train report requires a training run id, backend id, GPU/backend
integration, calibrated proxy, local export-gap proof pass, and non-positive
post-export BPB delta versus post-hoc quantization. The resource/cost report
requires a measurement run id, power meter id, positive wall time, average
power, total joules, telemetry validation, wall-time budget fit, energy budget
fit, and at most 5% consistency error between total joules and wall time times
average power. The Optimizer/Comm report binds the distributed backend id,
world size, sharding mode, overlap mode, planned wire bytes, and owned optimizer
state reduction to the current optimizer/communication plan; it requires
reduce-scatter parity, all-gather parity, optimizer update parity, validated
NCCL traces, overlap validation, finite measured communication/step timing, and
communication speedup at least 1. The KernelForge report binds generated kernel
IDs to the current exact tiled CE tile size and requires generated kernels,
backend parity, speedup, memory reduction, and local exact CE proof pass. The
Wind Tunnel report requires fresh traces, external timing validation, holdout
Spearman at least 0.50, and mean absolute percent error at most 25.

Remaining evidence: these commands bind and validate measured claims, but they
do not perform the external measurements. Release still requires the actual
production-scale SVD/calibration, domain delta training, backend training,
measured wall-power/energy telemetry, distributed optimizer/NCCL runs,
generated kernel runs, and fresh profiler/full-train validation artifacts.
