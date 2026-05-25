import { bg, footer, panel, rect, style, text, title } from "./shared.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Kernel work", "Several fusions landed; the biggest ones are still open", "The implemented work is useful, but whole-block persistent CTA and true XSA-inside-SDPA are not done.");

  panel(slide, ctx, 78, 286, 520, 288);
  rect(slide, ctx, 78, 286, 520, 7, style.green);
  text(slide, ctx, "Landed", 108, 314, 220, 28, { size: 24, color: style.green, bold: true, face: style.title });
  [
    "BF16 direct-compact backward chain",
    "QK/RoPE/q-gain and QKV tail fusion",
    "Adjacent SparseAttnGate/XSA BF16 variants",
    "BigramHash merge + fused backward, spec-gated",
    "Exact recurrent boundary fusion",
    "Quant layout metadata and fingerprints",
  ].forEach((label, idx) => {
    rect(slide, ctx, 112, 368 + idx * 30, 9, 9, idx < 2 ? style.green : style.accent);
    text(slide, ctx, label, 136, 358 + idx * 30, 390, 24, { size: 15.2, color: style.ink });
  });

  panel(slide, ctx, 680, 286, 520, 288);
  rect(slide, ctx, 680, 286, 520, 7, style.red);
  text(slide, ctx, "Still open", 710, 314, 220, 28, { size: 24, color: style.red, bold: true, face: style.title });
  [
    "Persistent-CTA whole-block backward",
    "True XSA inside SDPA",
    "Full train-step CUDA graph",
    "Generated quant kernel compiler",
    "Full record-ready BPB evidence",
  ].forEach((label, idx) => {
    rect(slide, ctx, 714, 378 + idx * 36, 9, 9, style.red);
    text(slide, ctx, label, 738, 366 + idx * 36, 390, 26, { size: 16, color: style.ink, bold: idx < 2 });
  });
  footer(slide, ctx, 7);
  return slide;
}

