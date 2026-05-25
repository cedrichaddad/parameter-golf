import { bg, badge, bullet, footer, panel, rect, style, text, title } from "./shared.mjs";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Compiler boundary", "Why this was not automatic compiler work", "torch.compile can fuse many graph patterns. These changes cross boundaries that ordinary PyTorch does not expose as one region.");

  const boxes = [
    ["Library calls", "cuDNN SDPA and cuBLASLt are not transparent graph regions.", 92, style.accent],
    ["Runtime state", "Profiles control BF16 layout, graph capture, recurrent replay, and audit gates.", 490, style.amber],
    ["Custom layouts", "Packed BF16 activations and export bytes affect legality and speed.", 888, style.green],
  ];
  boxes.forEach(([label, detail, x, color]) => {
    panel(slide, ctx, x, 312, 300, 150);
    rect(slide, ctx, x, 312, 300, 7, color);
    text(slide, ctx, label, x + 24, 342, 240, 28, { size: 23, color: style.ink, bold: true, face: style.title });
    text(slide, ctx, detail, x + 24, 386, 238, 48, { size: 15, color: style.muted });
  });
  panel(slide, ctx, 148, 530, 984, 74, { fill: "#FFFCF5" });
  badge(slide, ctx, "CAREFUL FRAMING", 178, 554, "amber", 164);
  text(slide, ctx, "Not impossible for compilers; just not automatically recovered from ordinary PyTorch code without custom-op integration.", 366, 546, 654, 42, {
    size: 20,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  bullet(slide, ctx, 178, 620, "Sources: torch.compile API/troubleshooting and torch.library custom-op docs.", { w: 760, h: 24, size: 12.5, color: style.soft });
  footer(slide, ctx, 9);
  return slide;
}

