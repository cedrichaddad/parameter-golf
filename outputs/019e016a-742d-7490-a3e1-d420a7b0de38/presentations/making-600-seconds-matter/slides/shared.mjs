export const style = {
  bg: "#F7F3EA",
  panel: "#FFFFFF",
  ink: "#17212B",
  muted: "#5B6573",
  soft: "#87909D",
  rule: "#D7D0C3",
  accent: "#2D6CDF",
  green: "#0A8F64",
  amber: "#D9822B",
  red: "#C2413D",
  dark: "#17212B",
  body: "Aptos",
  title: "Aptos Display",
  mono: "Aptos Mono",
};

export const metrics = {
  targetBpb: "1.05651",
  timeBudget: "600 s",
  byteBudget: "16,000,000 bytes",
  oldBroken: 91218.335,
  oldClean: 256.787,
  cleanExact: 137.735,
  shortExact: 127.110,
  diagnosticAllSt: 117.499,
  diagnosticBpb: 2.254799,
  exportedAllStBpb: 2.438243,
  inactive: 120.963,
  active: 148.852,
  backwardGraph: 121.959,
  bank: 10.341,
  nonBank: 2.259,
  other: 3.181,
};

export function rect(slide, ctx, x, y, w, h, fill = style.panel, opts = {}) {
  return ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    geometry: opts.geometry ?? "rect",
    fill,
    line: opts.line ?? ctx.line(opts.stroke ?? "#00000000", opts.strokeWidth ?? 0),
    name: opts.name,
  });
}

export function text(slide, ctx, value, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    text: String(value ?? ""),
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: opts.size ?? 18,
    color: opts.color ?? style.ink,
    bold: Boolean(opts.bold),
    typeface: opts.face ?? style.body,
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line(),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    name: opts.name,
  });
}

export function line(slide, ctx, x, y, w, color = style.rule, weight = 1) {
  rect(slide, ctx, x, y, w, weight, color);
}

export function bg(slide, ctx) {
  rect(slide, ctx, 0, 0, 1280, 720, style.bg);
}

export function title(slide, ctx, kicker, headline, subhead) {
  text(slide, ctx, kicker.toUpperCase(), 58, 42, 620, 22, {
    size: 10.5,
    color: style.accent,
    bold: true,
    face: style.mono,
  });
  text(slide, ctx, headline, 58, 76, 960, 92, {
    size: 38,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  if (subhead) {
    text(slide, ctx, subhead, 58, 176, 880, 44, { size: 17, color: style.muted });
  }
  line(slide, ctx, 58, 232, 1164, style.rule, 1);
}

export function footer(slide, ctx, n, label = "Rust/CUDA Parameter Golf systems deck") {
  line(slide, ctx, 58, 670, 1164, style.rule, 1);
  text(slide, ctx, label, 58, 684, 820, 20, { size: 9, color: style.soft });
  text(slide, ctx, String(n).padStart(2, "0"), 1184, 684, 38, 20, {
    size: 9,
    color: style.soft,
    align: "right",
    face: style.mono,
  });
}

export function badge(slide, ctx, label, x, y, tone = "blue", w = 132) {
  const fill = tone === "green" ? style.green : tone === "amber" ? style.amber : tone === "red" ? style.red : style.accent;
  rect(slide, ctx, x, y, w, 25, fill);
  text(slide, ctx, label.toUpperCase(), x, y + 5, w, 15, {
    size: 8.2,
    color: "#FFFFFF",
    bold: true,
    face: style.mono,
    align: "center",
  });
}

export function panel(slide, ctx, x, y, w, h, opts = {}) {
  rect(slide, ctx, x, y, w, h, opts.fill ?? style.panel, {
    stroke: opts.stroke ?? style.rule,
    strokeWidth: opts.strokeWidth ?? 1,
  });
}

export function metricCard(slide, ctx, x, y, w, h, value, label, tone = style.accent, note = "") {
  panel(slide, ctx, x, y, w, h);
  rect(slide, ctx, x, y, 5, h, tone);
  text(slide, ctx, value, x + 20, y + 16, w - 30, 42, {
    size: 32,
    color: style.ink,
    bold: true,
    face: style.title,
  });
  text(slide, ctx, label.toUpperCase(), x + 20, y + 64, w - 30, 18, {
    size: 9.8,
    color: tone,
    bold: true,
    face: style.mono,
  });
  if (note) text(slide, ctx, note, x + 20, y + 92, w - 30, Math.max(30, h - 102), { size: 12, color: style.muted });
}

export function bullet(slide, ctx, x, y, textValue, opts = {}) {
  rect(slide, ctx, x, y + 9, 8, 8, opts.color ?? style.accent);
  text(slide, ctx, textValue, x + 18, y, opts.w ?? 420, opts.h ?? 30, {
    size: opts.size ?? 16,
    color: opts.textColor ?? style.ink,
    bold: Boolean(opts.bold),
  });
}

export function hbar(slide, ctx, x, y, w, h, frac, color, label, value, opts = {}) {
  text(slide, ctx, label, x, y + 2, opts.labelW ?? 214, h, { size: opts.size ?? 12.5, color: style.muted });
  rect(slide, ctx, x + (opts.labelW ?? 214), y + 2, w, h - 4, "#ECE6DA");
  rect(slide, ctx, x + (opts.labelW ?? 214), y + 2, Math.max(2, w * frac), h - 4, color);
  text(slide, ctx, value, x + (opts.labelW ?? 214) + w + 12, y + 2, 120, h, {
    size: opts.size ?? 12.5,
    color: style.ink,
    bold: true,
    face: style.mono,
  });
}

export function columnHeader(slide, ctx, value, x, y, w, color = style.accent) {
  text(slide, ctx, value.toUpperCase(), x, y, w, 16, {
    size: 8.8,
    color,
    bold: true,
    face: style.mono,
  });
  line(slide, ctx, x, y + 22, w, color, 2);
}

export function statusRow(slide, ctx, x, y, w, label, status, tone, detail) {
  badge(slide, ctx, status, x, y, tone, 104);
  text(slide, ctx, label, x + 120, y + 2, 210, 22, { size: 11.5, color: style.ink, bold: true });
  text(slide, ctx, detail, x + 340, y + 1, Math.max(120, w - 340), 38, { size: 9.3, color: style.muted });
}
