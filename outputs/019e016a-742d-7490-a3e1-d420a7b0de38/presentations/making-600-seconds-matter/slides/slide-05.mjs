import { bg, footer, hbar, metrics, panel, rect, style, text, title } from "./shared.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Step anatomy", "The remaining time is concentrated", "Active recurrent replay is the difference between near-target and above-target.");

  panel(slide, ctx, 70, 274, 540, 328);
  text(slide, ctx, "Clean exact proxy", 102, 300, 270, 30, { size: 24, color: style.ink, bold: true, face: style.title });
  text(slide, ctx, "137.735 ms/step", 102, 340, 300, 38, { size: 34, color: style.green, bold: true, face: style.title });
  const parts = [
    ["backward graph", metrics.backwardGraph, style.accent],
    ["bank update", metrics.bank, style.amber],
    ["non-bank update", metrics.nonBank, style.green],
    ["other", metrics.other, style.rule],
  ];
  let cur = 102;
  parts.forEach(([label, value, color]) => {
    const w = 450 * value / metrics.cleanExact;
    rect(slide, ctx, cur, 408, w, 42, color);
    cur += w;
  });
  parts.forEach(([label, value, color], idx) => {
    const yy = 480 + idx * 28;
    rect(slide, ctx, 102, yy + 6, 12, 12, color);
    text(slide, ctx, label, 126, yy, 250, 24, { size: 14, color: style.muted });
    text(slide, ctx, `${value.toFixed(3)} ms`, 430, yy, 110, 24, { size: 14, color: style.ink, bold: true, face: style.mono, align: "right" });
  });

  panel(slide, ctx, 674, 274, 520, 328, { fill: "#FFFCF5" });
  text(slide, ctx, "Active recurrence is the gap", 706, 300, 340, 30, { size: 24, color: style.ink, bold: true, face: style.title });
  hbar(slide, ctx, 706, 374, 210, 36, metrics.inactive / metrics.active, style.green, "inactive", `${metrics.inactive.toFixed(3)} ms`, { labelW: 120, size: 15 });
  hbar(slide, ctx, 706, 450, 210, 36, 1, style.red, "active", `${metrics.active.toFixed(3)} ms`, { labelW: 120, size: 15 });
  text(slide, ctx, "Small flags are no longer enough. The next cut has to reduce exact recurrent replay.", 706, 522, 400, 38, {
    size: 17,
    color: style.ink,
    bold: true,
  });
  footer(slide, ctx, 5);
  return slide;
}
