// Seeded fake activations (plan D5). ONE engine, model-level knobs only.
// Everything rendered from here carries the SIMULATED badge; nothing is real inference.
export function mulberry32(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface AttCell { j: number; v: number }
// causal attention weights for the newest token, band-shaped by depth (SIMULATED)
export function attRow(layers: number, n: number, layer: number, seed: number, entropyBias = 0): AttCell[] {
  const rnd = mulberry32(seed * 131 + layer * 17 + layers);
  const T = 3.2 - 2.6 * (layer / layers); // sharpens with depth (low T late)
  const w: number[] = [];
  for (let j = 0; j <= n - 1; j++) {
    let v = 1 + (Math.abs(n - 1 - j) <= 1 ? 1.6 : 0) + rnd() * 1.4;
    if (j === 0) v += 0.6;
    if (j === n - 1) v += 0.8;
    w.push(Math.pow(v, T - entropyBias));
  }
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v, j) => ({ j, v: v / s }));
}
// FFN slot activity for 8 shown slots (SIMULATED), scaled by mid-band heat
export function ffnSlots(layers: number, layer: number, heatNow: number, seed = layer, moe = false): number[] {
  const rnd = mulberry32(seed * 77 + layer * 13 + 1);
  const out: number[] = [];
  for (let i = 0; i < 8; i++) {
    const base = rnd() * 0.55 + heatNow * (0.35 + rnd() * 0.55);
    out.push(Math.min(1, moe ? base * (0.85 + rnd() * 0.3) : base));
  }
  return out;
}
// residual-stream norm growth curve sqrt(1+0.20*i) (SIMULATED), as points 0..1
export function residCurve(layers: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  const vMax = Math.sqrt(1 + 0.2 * layers);
  for (let i = 1; i <= layers; i++) {
    const v = Math.sqrt(1 + 0.2 * i);
    pts.push({ x: (i - 1) / (layers - 1), y: (v - 1) / (vMax - 1) });
  }
  return pts;
}
// fallback top-5 when a model ships no authored preds: neutral flat list
export function fallbackPreds(label: string): [string, number][] {
  return [
    [label || "Ġnext", 22],
    ["Ġthe", 9],
    ["Ġa", 7],
    ["Ġand", 5],
    ["Ġof", 4],
  ];
}
