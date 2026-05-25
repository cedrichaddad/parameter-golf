import { bg, badge, footer, line, metricCard, metrics, rect, style, text } from "./shared.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  rect(slide, ctx, 58, 54, 5, 70, style.accent);
  text(slide, ctx, "PARAMETER GOLF SYSTEMS UPDATE", 78, 54, 520, 22, {
    size: 11,
    color: style.accent,
    bold: true,
    face: style.mono,
  });
  text(slide, ctx, "Making 600 Seconds\nMatter", 58, 142, 690, 150, {
    size: 54,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  text(slide, ctx, "A small-model challenge where runtime, artifact bytes, GPU data movement, and legal evaluation collide.", 62, 314, 650, 70, {
    size: 22,
    color: style.muted,
  });
  badge(slide, ctx, "SYSTEMS TALK", 62, 410, "blue", 144);
  badge(slide, ctx, "NOT A WIN CLAIM", 224, 410, "amber", 150);

  metricCard(slide, ctx, 820, 88, 340, 132, metrics.timeBudget, "runtime cap", style.accent, "The clock changes which kernels and data paths are viable.");
  metricCard(slide, ctx, 820, 246, 340, 132, "16 MB", "artifact cap", style.green, "Export bytes and hashes are part of the result.");
  metricCard(slide, ctx, 820, 404, 340, 132, `${metrics.targetBpb}`, "frontier BPB", style.amber, "Quality is not solved yet.");

  line(slide, ctx, 62, 532, 666, style.rule, 1);
  text(slide, ctx, "Thesis", 62, 548, 90, 22, { size: 13, color: style.accent, bold: true, face: style.mono });
  text(slide, ctx, "When the budget is this tight, kernel boundaries become part of the algorithm.", 158, 540, 650, 58, {
    size: 24,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  footer(slide, ctx, 1);
  return slide;
}
