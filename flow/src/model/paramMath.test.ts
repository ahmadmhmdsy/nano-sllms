import { describe, expect, it } from "vitest";
import {
  headDim,
  kvDim,
  pEmbed,
  pAttentionDense,
  pFFNDense,
  pNorms,
  pTotalDense,
} from "./paramMath";

// Anchor totals pinned to EXACT integers so any formula drift fails loudly.
describe("pTotalDense anchors", () => {
  it("smoke shape", () => {
    expect(pTotalDense({ layers: 4, d: 256, heads: 4, kv: 2, ffn: 1024, vocab: 32768 })).toBe(12323072);
  });
  it("pilot shape", () => {
    expect(pTotalDense({ layers: 12, d: 768, heads: 12, kv: 4, ffn: 2048, vocab: 32768 })).toBe(100682496);
  });
  it("target shape", () => {
    expect(pTotalDense({ layers: 16, d: 1024, heads: 16, kv: 4, ffn: 3072, vocab: 32768 })).toBe(226526208);
  });
  it("SmolLM2-135M shape", () => {
    expect(pTotalDense({ layers: 30, d: 576, heads: 9, kv: 3, ffn: 1536, vocab: 49152 })).toBe(134515008);
  });
});

describe("component formulas (target shape d=1024, heads=16, kv=4)", () => {
  it("headDim = d/heads floored", () => {
    expect(headDim({ d: 1024, heads: 16 })).toBe(64);
    expect(headDim({ d: 576, heads: 9 })).toBe(64);
  });
  it("kvDim = kv * headDim", () => {
    expect(kvDim({ d: 1024, heads: 16, kv: 4 })).toBe(256);
  });
  it("pEmbed = vocab * d", () => {
    expect(pEmbed({ d: 1024, vocab: 32768 })).toBe(33554432);
  });
  it("pAttentionDense = 2d² + 2d·kvdim", () => {
    expect(pAttentionDense({ d: 1024, heads: 16, kv: 4 })).toBe(2621440);
  });
  it("pFFNDense = 3·d·ffn", () => {
    expect(pFFNDense({ d: 1024, ffn: 3072 })).toBe(9437184);
  });
  it("pNorms = (2L+1)·d counted once", () => {
    expect(pNorms({ d: 1024, layers: 16 })).toBe(33792);
  });
});
