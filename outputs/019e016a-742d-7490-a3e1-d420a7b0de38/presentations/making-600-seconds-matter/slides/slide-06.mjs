import { bg, footer, metricCard, panel, style, text, title } from "./shared.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Runtime evidence", "The checks that worked", "The strongest contribution so far is making the fast path measurable and enforceable.");

  metricCard(slide, ctx, 82, 300, 330, 180, "0", "host batch copies", style.green, "Device-resident token ring removed per-step host batch construction.");
  metricCard(slide, ctx, 474, 300, 330, 180, "0", "BF16/F32 bridges", style.green, "Runtime counters replaced static precision readiness checks.");
  metricCard(slide, ctx, 866, 300, 330, 180, "no-loss", "backward graph", style.accent, "Captured backward path reduced launch overhead.");

  panel(slide, ctx, 152, 540, 976, 64, { fill: "#FFFCF5" });
  text(slide, ctx, "Why it matters", 180, 560, 170, 22, { size: 15, color: style.accent, bold: true, face: style.mono });
  text(slide, ctx, "The run now has evidence for speed, data residency, precision, and export checks.", 370, 552, 660, 34, {
    size: 22,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  footer(slide, ctx, 6);
  return slide;
}

