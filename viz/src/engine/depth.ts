// Depth bands + knowledge-heat ramp. Rule-of-thumb heuristic (plan D6) —
// labeled "rule-of-thumb" wherever rendered; overridable per-model via YAML bands.
type Band = readonly [number, number];
export type Bands = { early: Band; mid: Band; late: Band };

export function bandOf(bands: Bands, layer: number): "early" | "mid" | "late" {
  if (layer >= bands.early[0] && layer <= bands.early[1]) return "early";
  if (layer >= bands.mid[0] && layer <= bands.mid[1]) return "mid";
  return "late";
}
export function bandWidthPct(bands: Bands, layers: number, which: "early" | "mid" | "late"): number {
  const [a, b] = bands[which];
  return ((b - a + 1) / layers) * 100;
}
// heat(i): triangular ramp peaking at the mid-band center; floor 0.12
export function heat(bands: Bands, i: number): number {
  const c = (bands.mid[0] + bands.mid[1]) / 2;
  const half = (bands.mid[1] - bands.mid[0]) / 2 + 2;
  return Math.max(0.12, 1 - Math.abs(i - c) / half);
}
// lighten a #rrggbb color by frac (heat tint)
export function mix(hex: string, frac: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const f = (a: number) => Math.round(a + (255 - a) * frac * 0.45);
  return "rgb(" + f(r) + "," + f(g) + "," + f(b) + ")";
}
