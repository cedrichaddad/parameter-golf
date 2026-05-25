# QA Scorecard

Deck: `making-600-seconds-matter.pptx`

Render/export:
- Build command completed successfully.
- Exported PPTX bytes: 69,562.
- Rendered 10 PNG previews.
- Generated contact sheet.

Layout QA:
- `check_layout_quality.mjs --warn-only --min-gap 10`
- Result: 0 errors, 25 warnings.
- Remaining warnings are tight-text, conservative bottom-padding, or split-inline warnings on intentionally paired label/detail rows. The visual contact sheet was reviewed after the 10-slide rewrite.

Claim QA:
- Includes bad BPB results as diagnostic evidence.
- Does not claim leaderboard win, record-ready status, true XSA-inside-SDPA, persistent-CTA whole-block backward, full train-step CUDA graph, or complete generated quant compiler.
- Torch.compile framing is bounded: "not automatically discoverable from ordinary PyTorch code," not "torch.compile cannot do this."
- Revised for larger type, 10-slide pacing, and presentation-facing titles.
