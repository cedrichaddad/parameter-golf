import { bg, bullet, footer, panel, rect, style, text, title } from "./shared.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Why it matters", "Parameter Golf is a systems problem", "The same design has to satisfy quality, runtime, bytes, data residency, and evaluation rules.");

  const nodes = [
    ["TIME", "600 s train and eval windows", 190, 312, style.accent],
    ["BYTES", "16 MB artifact cap", 545, 250, style.green],
    ["BPB", "full-validation score", 885, 312, style.amber],
    ["LEGALITY", "score-first eval and data audit", 545, 462, style.red],
  ];
  nodes.forEach(([name, desc, x, y, color]) => {
    panel(slide, ctx, x, y, 220, 112);
    rect(slide, ctx, x, y, 220, 7, color);
    text(slide, ctx, name, x + 18, y + 22, 180, 30, { size: 24, color, bold: true, face: style.title });
    text(slide, ctx, desc, x + 18, y + 64, 180, 34, { size: 14, color: style.muted });
  });
  text(slide, ctx, "One design surface", 486, 364, 310, 34, { size: 28, color: style.ink, bold: true, face: style.title, align: "center" });

  bullet(slide, ctx, 92, 588, "Context length, TTT, and quantization affect both speed and quality.", { w: 530 });
  bullet(slide, ctx, 680, 588, "Data residency and kernel boundaries decide whether the run fits.", { w: 470, color: style.amber });
  footer(slide, ctx, 2);
  return slide;
}

