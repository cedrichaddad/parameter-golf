# Proposal Feature Status

This is the handoff table for local proposal work. It intentionally separates implementation from H100 validation and final BPB evidence.

| Feature | Status | Local Evidence | H100/BPB Evidence Needed |
|---|---:|---|---|
| Quant layout compiler | Implemented and locally tested | `pg-quant` manifest/CRC/kernel ID tests; strict loader tests | CUDA pack/dequant microbench |
| Artifact-lab bit allocation | Local proxy | `pg-local artifact-lab` reports mixed bit-allocation byte estimates and optional mini-BPB from explicit byte fixtures | Final export byte proof and post-export full-validation BPB |
| Compile-time CUDA arch metadata | Implemented and compile-checked | `PG_COMPILED_CUDA_ARCHES`, `PG_COMPILED_ARCH_PROFILE`, `cfg(pg_cuda_arch_sm90)` compile with `--features cuda` | Build log from final H100 image |
| Quant CUDA pack/dequant entrypoints | Implemented as optional symbols | `quant_pack.cu` compiles under CUDA feature check | Perf and parity microbench on H100 |
| Strict artifact manifest validation | Implemented and locally tested | malformed JSON, missing CRC, QuantSpec mismatch, strict export reload smoke | Final record export proof |
| BigramHash fused merge | Implemented and locally tested | CPU fused forward/backward reference, finite-difference tests, CUDA symbol stability test | CUDA parity/perf on H100; BPB if enabled in a variant |
| #2135 BigramHash claim | Scoped out | Audit keeps Bigram disabled for #2135 unless explicitly configured | None unless a new Bigram variant is proposed |
| Adjacent SparseAttnGate/XSA fusion | Implemented and scoped | CPU composite parity and audit `xsa_fusion_kind` tests | H100 perf/parity for active profile |
| True XSA-inside-SDPA | Local reference only, not record-active | CPU semantic forward/backward parity and optional naive F32 CUDA reference wrapper; audit still reports `true_xsa_in_attention_fusion=false` | Wire into `pg-model` for XSA layers, then CUDA parity/perf on H100 |
| Exact recurrent boundary fusion | Implemented in runtime surface | Audit distinguishes exact boundary fusion from ST speed probes | H100 exact active-step timing |
| Persistent-CTA block backward | Not implemented | Audit reports inactive/unvalidated | Exact-gradient kernel and H100 active-window improvement |
| Full train-step CUDA graph | Not implemented | Audit keeps claim false | Graph capture including data materialization, loss, backward, updates |
| Final record BPB | Not validated | None locally | Full train/eval/export and 3 seeds on canonical CaseOps |
| Artifact budget | Locally hardened, not final | Strict metadata and byte-estimate tests | Final compressed model/code byte proof from record run |
| PG-Lite | Local proxy only | `pg-local lite run` reports `proxy_only=true`, local BPB, wall time, actual compact `model.pglite.bin` artifact bytes plus debug JSON, loadable artifact manifests, and budget flags; `pg-local lite suite` runs all checked-in configs and writes summary JSON/Markdown | None; PG-Lite is not the official leaderboard benchmark |
| PG-Lite Metal backend | Interface plus source boundary | `pg-local` parses `metal_apple`, rejects it by default, only allows explicitly labeled CPU fallback with requested/execution backend fields, and `pg-local backend-check` verifies the checked-in `pg_lite_ngram.metal` source plus local compiler/runtime availability. The source now includes loss, u32-residual loss, training, and reduction kernel symbols, and backend-check will compile the source to a temporary `.metallib` when the Apple Metal compiler is installed. | Install Metal compiler/runtime, link real Rust Metal executor, then local parity/perf evidence |
| Wind Tunnel | Estimate only | `pg-local wind-tunnel` reports `estimate_only=true`, stage estimates, bottleneck hints, and a spec-derived operation DAG; `pg-local wind-tunnel-suite` now ingests JSON and log traces, distinguishes total-only traces from nonzero stage-attribution traces, and writes checked-in calibration summaries under `records/wind_tunnel/` | Fresh H100 profiler trace and full train/eval timing |
| Historical H100 trace index | Local evidence inventory | `pg-local trace-index` scans the broader records tree, extracts timing/audit fields from JSON/log/markdown artifacts, and writes `records/wind_tunnel/all_records_trace_index.json`; current index finds 109 trace-like files, 13 timed traces, best exact/#2135 timing at 127.109924 ms/step, and best active recurrent timing at 155.576088 ms/active step | New traces still required for final validation and richer stage attribution |

## Handoff Notes

- Keep `frontier_2135_audit_target.toml` as the primary competitive target.
- Keep speed probes and straight-through recurrence labeled as diagnostic until full BPB validates them.
- Do not claim record-active true XSA-inside-SDPA, persistent-CTA block backward, full train-step graph, record readiness, or leaderboard win from local tests alone.
- Do not cite Wind Tunnel estimates as measured H100 timing.
- Use `pg-local lite suite` and `pg-local wind-tunnel-suite` as local regression
  surfaces before spending Modal credits, but keep both labeled as local proxy
  evidence.
- Use `pg-local proof-bundle` for handoff packets; it now includes PG-Lite suite
  summaries and backend readiness probes in addition to the single-run checks.
- Treat `evidence_manifest.json` as the top-level handoff index; it explicitly
  separates local evidence from missing H100/BPB/artifact proof.
- Do not compare PG-Lite proxy BPB directly against official FineWeb/H100 leaderboard BPB.
- Do not call a `metal_apple` CPU-fallback PG-Lite run a Metal result.
- The next paid runs should be H100 validation only after artifact budget and local checks are green.
