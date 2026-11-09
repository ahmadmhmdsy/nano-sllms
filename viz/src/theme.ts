// Single source of styling truth (plan D9). No CSS framework.
export const theme = {
  bg: "#0f0f13",
  panel: "#16161d",
  panel2: "#1c1c26",
  line: "#2a2a36",
  line2: "#3a3a48",
  ink: "#e9e7e2",
  dim: "#9a97a3",
  faint: "#6a6776",
  // component colors
  embed: "#56c8d8",
  attn: "#e0a458",
  ffn: "#a78bdc",
  norm: "#8a8f98",
  out: "#e8635a",
  res: "#7fc8a9",
  mono: "ui-monospace, 'Cascadia Mono', Consolas, monospace",
  serif: "Georgia, 'Iowan Old Style', 'Times New Roman', serif",
  sans: "system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
};
export const kindColors = {
  tokens: "#4a4a5a",
  embed: theme.embed,
  lnorm: "#4a4a5a",
  attn: theme.attn,
  ffn: theme.ffn,
  add: theme.res,
  band: theme.ffn,
  finalnorm: theme.norm,
  lmhead: theme.out,
  logits: "#4a4a5a",
  layer: "#33333f",
};
