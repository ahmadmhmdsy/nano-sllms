// archKind = denseGqaMoe — every dense (GQA + dense SwiGLU) branch lives here.
// Shared components call these; no per-model special cases (plan §6).
import type { ModelSpec } from "../../schema/spec";
import { pAttentionDense, pEmbed, pFFNDense, pNorms, pTotal, fmtP } from "../../engine/params";

export function attnTitle(spec: ModelSpec): string {
  return "GQA attention — the router";
}
export function attnBoxLabel(spec: ModelSpec): string {
  return "Gated Attention · GQA";
}
export function attnBoxSub(spec: ModelSpec): string {
  return spec.shape.heads + " Q → " + spec.shape.kv + " KV · head_dim " + Math.floor(spec.shape.d / spec.shape.heads) + " · RoPE";
}
export function ffnBoxLabel(spec: ModelSpec): string {
  return "SwiGLU FFN";
}
export function ffnBoxSub(spec: ModelSpec): string {
  return "gate·up " + fmtP(spec.shape.ffn) + " → down " + spec.shape.d;
}
export function blockParamsAttn(spec: ModelSpec): string {
  return fmtP(pAttentionDense(spec.shape));
}
export function blockParamsFfn(spec: ModelSpec): string {
  return fmtP(pFFNDense(spec.shape));
}
export function layerLabel(spec: ModelSpec, idx: number, suffix = ""): string {
  return "decoder block · layer " + idx + (suffix ? " " + suffix : "");
}
export function headTitle(spec: ModelSpec): string {
  return "LM head — tied to the embedding table";
}
export function headSub(spec: ModelSpec): string {
  return spec.shape.d + " → " + fmtP(spec.shape.vocab) + " scores · same matrix reused";
}
// param-share rows — computed in engine, labeled here
export function shareRows(spec: ModelSpec): [string, number, string][] {
  return [
    ["Embedding — meaning of single tokens", pEmbed(spec.shape), "#56c8d8"],
    ["Attention — routing between tokens", pAttentionDense(spec.shape) * spec.shape.layers, "#e0a458"],
    ["FFN — the knowledge base", pFFNDense(spec.shape) * spec.shape.layers, "#a78bdc"],
    ["Norms — stability", pNorms(spec.shape), "#8a8f98"],
  ];
}
export function headBadge(spec: ModelSpec): string {
  return fmtP(pTotal(spec)) + " params";
}
