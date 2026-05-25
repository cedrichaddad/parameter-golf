# Source Notes

Deck: Making 600 Seconds Matter

Audience: technical generalists. Goal: explain why the Parameter Golf systems work is interesting, what was actually implemented, which timing gains are substantiated, and why current BPB results are diagnostic rather than competitive.

Primary local sources:
- `/Users/cedrichaddad/parameter-golf/parameter-golf-rs/paper/week5_blog_post.md`
- `/Users/cedrichaddad/parameter-golf/parameter-golf-rs/paper/parameter_golf_rust_cuda_draft.md`
- `/Users/cedrichaddad/parameter-golf/records/track_non_record_16mb/2026-04-30_RustCudaSystems/TECHNICAL_REPORT.md`
- `/private/tmp/frontier_2135_audit_boundary_fusion_v3.json`
- `/private/tmp/frontier_2135_exactfused_accum_short_v1.json`
- `/private/tmp/frontier_2135_sparsexsa_bf16dx_short_v1.json`
- `/private/tmp/pg-output-sample/frontier2135_hybridst3_inmemory_quality_v3.json`
- `/private/tmp/pg-output-sample/eval_full_allst_dist8_lorafix6_trace_v1.json`

External references for the torch.compile boundary slide:
- https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html
- https://docs.pytorch.org/docs/2.12/generated/torch.compile.html
- https://docs.pytorch.org/docs/stable/library.html

Important claim discipline:
- The deck may claim BF16 direct-compact hot-path enforcement, zero measured BF16/F32 bridge launches, device-resident sampler, no-loss backward graph capture, adjacent SparseAttnGate/XSA kernels, spec-gated BigramHash merge/fused backward, exact recurrent boundary fusion, and quant layout metadata/fingerprints.
- The deck must not claim persistent-CTA whole-block backward, true XSA-inside-SDPA, full train-step CUDA graph, generated quant compiler completion, record-ready status, or leaderboard win.
- BPB values are included as negative/diagnostic evidence, not quality evidence.

