import { bg, footer, panel, rect, style, text, title } from "./shared.mjs";

function box(slide, ctx, x, y, w, h, label, detail, color) {
  panel(slide, ctx, x, y, w, h);
  rect(slide, ctx, x, y, w, 7, color);
  text(slide, ctx, label, x + 18, y + 18, w - 36, 28, { size: 18, color: style.ink, bold: true });
  text(slide, ctx, detail, x + 18, y + 58, w - 36, h - 70, { size: 13.5, color: style.muted });
}

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "The stack", "A custom runtime with audit hooks", "The project moved hot-path choices into measured runtime profiles.");

  box(slide, ctx, 74, 286, 188, 134, "Data", "CaseOps audit, sidecar hashes, device token ring", style.accent);
  box(slide, ctx, 304, 286, 188, 134, "Runtime", "RunSpec profiles, BF16 chain, graph controls", style.green);
  box(slide, ctx, 534, 258, 214, 190, "GPU kernels", "cuDNN BF16 SDPA, cuBLASLt GEMMs, custom QKV/XSA/recurrent paths", style.amber);
  box(slide, ctx, 790, 286, 188, 134, "Optimizer", "sharded Muon, graph regions, byte-aware export", style.red);
  box(slide, ctx, 1020, 286, 188, 134, "Audit", "timing, BPB, artifact bytes, hashes, gates", style.accent);

  [264, 494, 750, 980].forEach((x) => {
    rect(slide, ctx, x, 350, 40, 3, style.rule);
    text(slide, ctx, ">", x + 32, 337, 24, 28, { size: 24, color: style.rule, bold: true });
  });

  panel(slide, ctx, 154, 518, 972, 72, { fill: "#FFFCF5" });
  text(slide, ctx, "What changed", 182, 540, 150, 24, { size: 16, color: style.accent, bold: true, face: style.mono });
  text(slide, ctx, "The run now explains which profile ran, where time went, and whether the result can be checked.", 350, 531, 690, 42, {
    size: 20,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  footer(slide, ctx, 3);
  return slide;
}

