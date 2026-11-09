// Param-math unit tests — anchors from plan §5 (computed from shapes; norms ONCE).
import { describe, it, expect } from "vitest";
import {
  pTotalDense, pTotalHybrid, pMixerLinear, pMixerFull, pMoeLayer, pEmbed, pNorms,
} from "../src/engine/params";
import { MODELS_FOR_TESTS } from "./helpers";

const nano = MODELS_FOR_TESTS.nano;
const smol = MODELS_FOR_TESTS.smollm2;
const qwen = MODELS_FOR_TESTS.qwen38;

describe("params — exact anchors", () => {
  it("nano total = 226,526,208 (norms counted once)", () => {
    expect(pTotalDense(nano.shape)).toBe(226_526_208);
  });
  it("smollm2 total = 134,515,008", () => {
    expect(pTotalDense(smol.shape)).toBe(134_515_008);
  });
  it("norm accounting: (2L+1)·d, never per-layer", () => {
    expect(pNorms(nano.shape)).toBe(33 * 1024);
    expect(pNorms(smol.shape)).toBe(61 * 576);
  });
  it("embedding math", () => {
    expect(pEmbed(nano.shape)).toBe(32768 * 1024);
  });
});
describe("params — hybrid text stack", () => {
  it("qwen total within ±1.5% of the 124.6B published estimate", () => {
    const t = pTotalHybrid(qwen);
    expect(t).toBeGreaterThan(1.2e11);
    expect(t).toBeLessThan(1.27e11);
    expect(Math.abs((t - 1.246e11) / 1.246e11)).toBeLessThan(0.015);
  });
  it("mixers/moe are sane positive values", () => {
    expect(pMixerLinear(qwen.shape, qwen.hybrid!.linearMixer!)).toBeGreaterThan(0);
    expect(pMixerFull(qwen.shape, 256)).toBeGreaterThan(0);
    expect(pMoeLayer(qwen.shape, qwen.hybrid!.experts!)).toBeGreaterThan(1e9);
  });
});
