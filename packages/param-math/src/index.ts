// Shared param math — the ONE canonical implementation for viz/ and flow/
// (TypeScript) and for webui/ and scripts/ (Python: param_math.py).
// Computed from spec shapes, NEVER hand-written numbers in UI text.
//
// Dense anchors (unit-tested): nano_SLMs 226,526,208 · SmolLM2-135M 134,515,008.
// Norm accounting: pre-norm ×2 per layer + 1 final norm = (2L+1)·d — counted ONCE.
//
// This package is deliberately dependency-free: the app modules (viz, flow)
// re-export from here, so the formulas below are the only place they exist.

/** Dense shape inputs — the fields every dense formula needs. */
export interface DenseShape {
  layers: number;
  d: number;
  heads: number;
  kv: number;
  ffn: number;
  vocab: number;
}

/** Hybrid (linear-attention + full-attention + sparse MoE) config — mirrors
 *  the `hybrid` block of the viz model YAMLs (camelCase keys). */
export interface HybridCfg {
  fullEvery?: number;
  layerPattern?: string;
  linearMixer?: { kHeads: number; vHeads: number; qHeads: number; headDim: number; conv: number };
  fullMixer?: { partialRotary?: string; indexer?: string; headDim?: number };
  experts?: { total: number; active: number; shared: number; inter: number };
  extras?: { label: string; params: number };
}

export type HybridLike = {
  shape: DenseShape;
  hybrid?: HybridCfg;
  archKind?: string;
};

export function headDim(s: { d: number; heads: number }): number {
  return Math.floor(s.d / s.heads);
}
export function kvDim(s: { d: number; heads: number; kv: number }): number {
  return s.kv * headDim(s);
}
export function pEmbed(s: { d: number; vocab: number }): number {
  return s.vocab * s.d;
}
export function pAttentionDense(s: { d: number; heads: number; kv: number }): number {
  const kv = kvDim(s);
  return 2 * s.d * s.d + 2 * s.d * kv; // WQ,WO (square) + WK,WV (d×kvdim)
}
export function pFFNDense(s: { d: number; ffn: number }): number {
  return 3 * s.d * s.ffn; // gate + up + down (SwiGLU, no bias)
}
export function pNorms(s: { d: number; layers: number }): number {
  return (2 * s.layers + 1) * s.d;
}
export function pLayerDense(s: { d: number; heads: number; kv: number; ffn: number }): number {
  // 2 * s.d = the two pre-norms' weights for this one layer (d each).
  // pNorms counts all (2L+1) norms exactly once — it is added in pTotalDense,
  // never here, so norms are never double-counted.
  return pAttentionDense(s) + pFFNDense(s) + 2 * s.d;
}
// pTotal — norms counted exactly once (NOT inside the per-layer loop):
export function pTotalDense(s: DenseShape): number {
  return pEmbed(s) + s.layers * (pAttentionDense(s) + pFFNDense(s)) + pNorms(s);
}

// ---- hybrid (linear-attention + full-attention + sparse MoE) ----

export function layerKindHybrid(l: number, hybrid: HybridCfg): "linear" | "full" {
  const full = hybrid.fullEvery ?? 4;
  return l % full === 0 ? "full" : "linear";
}
// GDN linear mixer, computed from its recipe; dt/norm remainder lives in extras:
export function pMixerLinear(s: { d: number }, lm: NonNullable<HybridCfg["linearMixer"]>): number {
  const inProj = s.d * (lm.kHeads + lm.qHeads + lm.vHeads) * lm.headDim;
  const outProj = lm.vHeads * lm.headDim * s.d;
  const conv = lm.conv * (lm.kHeads + lm.vHeads) * lm.headDim;
  return inProj + outProj + conv;
}
// gated full attention (GQA): WQ + WKV + WO
export function pMixerFull(s: { d: number; heads: number; kv: number }, headDimFull: number): number {
  const q = s.heads * headDimFull;
  const kv = s.kv * headDimFull;
  return s.d * q + 2 * s.d * kv + q * s.d;
}
// router: d -> experts (per MoE layer)
export function pRouter(s: { d: number }, experts: { total: number }): number {
  return s.d * experts.total;
}
export function pMoeLayer(s: { d: number }, e: { total: number; shared: number; inter: number }): number {
  const perExpert = 3 * s.d * e.inter;
  return e.total * perExpert + e.shared * perExpert + pRouter(s, e);
}
export function nFullLayers(layers: number, hybrid: HybridCfg): number {
  return Math.floor(layers / (hybrid.fullEvery ?? 4));
}
export function nLinearLayers(layers: number, hybrid: HybridCfg): number {
  return layers - nFullLayers(layers, hybrid);
}
export function pTotalHybrid(m: { shape: DenseShape; hybrid?: HybridCfg }): number {
  const s = m.shape, h = m.hybrid ?? {};
  const e = h.experts ?? { total: 0, shared: 0, active: 0, inter: s.ffn };
  const headDimFull = h.fullMixer?.headDim ?? Math.round(s.d / s.heads);
  const linPer = h.linearMixer ? pMixerLinear(s, h.linearMixer) : 0;
  const fullPer = pMixerFull(s, headDimFull);
  const moePer = pMoeLayer(s, e);
  const extras = h.extras?.params ?? 0;
  return (
    2 * pEmbed(s) + // untied: input embedding + separate output head
    nFullLayers(s.layers, h) * fullPer +
    nLinearLayers(s.layers, h) * linPer +
    s.layers * moePer +
    extras
  );
}

export function isHybrid(m: { archKind?: string }): boolean {
  return m.archKind === "hybridLinearMoe";
}
export function fmtP(n: number): string {
  return n.toLocaleString("en-US");
}
export function fmtB(n: number): string {
  return n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : fmtP(n);
}
export function pTotal(m: HybridLike): number {
  return isHybrid(m) ? pTotalHybrid(m) : pTotalDense(m.shape);
}
