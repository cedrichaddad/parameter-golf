import { bg, badge, footer, metricCard, panel, style, text, title } from "./shared.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Quality", "Speed exposed the next problem", "The BPB numbers are not competitive. They show that timing alone is not the finish line.");

  metricCard(slide, ctx, 92, 300, 300, 168, "~1.05651", "frontier BPB", style.accent, "The #2135 target remains the quality bar.");
  metricCard(slide, ctx, 490, 300, 300, 168, "2.254799", "hybrid-ST run", style.red, "Fast enough to diagnose, not good enough to claim.");
  metricCard(slide, ctx, 888, 300, 300, 168, "2.438243", "exported all-ST", style.red, "Post-export quality was worse.");

  panel(slide, ctx, 164, 542, 952, 64, { fill: "#FFFCF5" });
  badge(slide, ctx, "TAKEAWAY", 192, 562, "amber", 118);
  text(slide, ctx, "The systems stack is now fast enough to reveal quality and export failures directly.", 330, 554, 660, 34, {
    size: 23,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  footer(slide, ctx, 8);
  return slide;
}

