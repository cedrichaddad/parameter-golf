import { bg, badge, footer, panel, rect, style, text, title } from "./shared.mjs";

export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "Next steps", "The finish line is evidence", "The project is close on systems throughput, but it still needs quality, artifact, and full-run proof.");

  const items = [
    ["1", "Cut exact active recurrence", "Reduce recurrent replay enough to clear <=120 ms/step.", style.red],
    ["2", "Run full train/eval/export", "Canonical data, no validation truncation, final audit after export.", style.amber],
    ["3", "Prove artifact budget", "Report decimal bytes and hashes under the 16 MB cap.", style.green],
    ["4", "Validate BPB", "Use full validation and three seeds before any win claim.", style.accent],
  ];
  items.forEach(([num, label, detail, color], idx) => {
    const y = 274 + idx * 78;
    panel(slide, ctx, 122, y, 1036, 60);
    rect(slide, ctx, 122, y, 58, 60, color);
    text(slide, ctx, num, 122, y + 15, 58, 30, { size: 27, color: "#FFFFFF", bold: true, face: style.title, align: "center" });
    text(slide, ctx, label, 210, y + 12, 390, 24, { size: 19, color: style.ink, bold: true });
    text(slide, ctx, detail, 610, y + 14, 450, 24, { size: 15, color: style.muted });
  });
  badge(slide, ctx, "BOTTOM LINE", 122, 612, "blue", 130);
  text(slide, ctx, "We have not won yet. We built the measured systems stack that shows what remains.", 280, 604, 700, 42, {
    size: 24,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  footer(slide, ctx, 10);
  return slide;
}

