// archKind = hybridLinearMoe — Qwen-style GDN hybrid + sparse MoE branches.
// Values computed via engine/params; labels are the only concern here.
import type { ModelSpec } from "../../schema/spec";
import {
  pEmbed, pMixerLinear, pMixerFull, pMoeLayer, pTotalHybrid, fmtP, fmtB,
} from "../../engine/params";

export function attnTitle(spec: ModelSpec, l: number, full: boolean): string {
  return full ? "Gated attention (1-in-4) — the exact mixer" : "GDN linear attention — the state mixer";
}
export function attnBoxLabel(spec: ModelSpec, full: boolean): string {
  return full ? "Gated Attention · GQA" : "Gated DeltaNet (linear attn)";
}
export function attnBoxSub(spec: ModelSpec, full: boolean): string {
  if (full) {
    return "1-in-4 layers · " + spec.shape.heads + " Q → " + spec.shape.kv + " KV · partial RoPE";
  }
  const lm = spec.hybrid?.linearMixer;
  if (!lm) return "state mixer";
  return "state mixer · " + lm.kHeads + " K / " + lm.vHeads + " V heads ×" + lm.headDim + " · conv-" + lm.conv;
}
export function blockParamsAttn(spec: ModelSpec, full: boolean): string {
  if (full) {
    const d = spec.hybrid?.fullMixer?.headDim ?? 256;
    return "≈" + fmtB(pMixerFull(spec.shape, d));
  }
  const lm = spec.hybrid?.linearMixer;
  return lm ? "≈" + fmtB(pMixerLinear(spec.shape, lm)) : "—";
}
export function ffnBoxLabel(spec: ModelSpec): string {
  return "Sparse MoE FFN";
}
export function ffnBoxSub(spec: ModelSpec): string {
  const e = spec.hybrid?.experts;
  if (!e) return "moe";
  return e.total + " experts · " + e.active + " active + " + e.shared + " shared";
}
export function blockParamsFfn(spec: ModelSpec): string {
  const e = spec.hybrid?.experts;
  return e ? "≈" + fmtB(pMoeLayer(spec.shape, e)) : "—";
}
export function layerLabel(spec: ModelSpec, idx: number, suffix = "", full = false): string {
  return "block " + idx + " · " + (full ? "gated attention (1 in 4)" : "linear attention (GDN)") + (suffix ? " · " + suffix : "");
}
export function headTitle(spec: ModelSpec): string {
  return "LM head — separate (untied)";
}
export function headSub(spec: ModelSpec): string {
  return spec.shape.d + " → " + fmtP(spec.shape.vocab) + " scores · separate matrix";
}
export function shareRows(spec: ModelSpec): [string, number, string][] {
  const s = spec.shape, h = spec.hybrid!;
  const e = h.experts ?? { total: 0, active: 0, shared: 0, inter: s.ffn };
  const nFull = Math.floor(s.layers / (h.fullEvery ?? 4));
  const nLin = s.layers - nFull;
  const fullPer = pMixerFull(s, h.fullMixer?.headDim ?? 256);
  const linPer = h.linearMixer ? pMixerLinear(s, h.linearMixer) : 0;
  return [
    ["Embedding + output head (untied)", 2 * pEmbed(s), "#56c8d8"],
    ["Token mixers — " + nLin + " linear + " + nFull + " full", nFull * fullPer + nLin * linPer, "#e0a458"],
    ["Sparse MoE — " + e.total + " experts ×" + s.layers + " + shared", s.layers * pMoeLayer(s, e), "#a78bdc"],
    ["Router-adjacent, MTP + extras", h.extras?.params ?? 0, "#8a8f98"],
  ];
}
export function headBadge(spec: ModelSpec): string {
  return "≈" + fmtB(pTotalHybrid(spec)) + " (text stack) · ≈6B active";
}
export function isFullLayer(spec: ModelSpec, l: number): boolean {
  const full = spec.hybrid?.fullEvery ?? 4;
  return l % full === 0;
}
