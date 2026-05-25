import { bg, badge, footer, hbar, metrics, panel, style, text, title } from "./shared.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Timing", "Speed became real", "The clean exact path is close to the target; the sub-target all-ST result is diagnostic.");

  panel(slide, ctx, 84, 274, 1090, 324);
  const maxLog = Math.log10(metrics.oldBroken);
  [
    ["Broken real path", metrics.oldBroken, "real path", style.red],
    ["Old clean floor", metrics.oldClean, "clean proxy", style.accent],
    ["Clean exact proxy", metrics.cleanExact, "record-shaped", style.green],
    ["Short exact probe", metrics.shortExact, "short probe", style.green],
    ["All-ST quality run", metrics.diagnosticAllSt, "diagnostic", style.amber],
  ].forEach(([label, value, runType, color], idx) => {
    const frac = Math.log10(value) / maxLog;
    hbar(slide, ctx, 126, 322 + idx * 42, 520, 30, frac, color, label, `${value.toFixed(value > 1000 ? 0 : 3)} ms`, { labelW: 210, size: 14 });
    badge(slide, ctx, runType, 1000, 325 + idx * 42, runType === "diagnostic" ? "amber" : runType === "real path" ? "red" : "green", 130);
  });
  text(slide, ctx, "Target: <=120 ms/step median, p90 <=125 ms on the exact #2135 profile.", 126, 540, 780, 32, {
    size: 18,
    color: style.ink,
    bold: true,
  });
  footer(slide, ctx, 4);
  return slide;
}
