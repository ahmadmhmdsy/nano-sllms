import { describe, it, expect } from "vitest";
import { mulberry32, attRow, ffnSlots, residCurve, fallbackPreds } from "../src/engine/sim";

describe("seeded sim engine", () => {
  it("mulberry32 is deterministic", () => {
    const a = mulberry32(42), b = mulberry32(42);
    expect(a()).toBe(b());
    expect(a()).toBe(b());
  });
  it("attRow is causal-normalized", () => {
    const row = attRow(16, 5, 3, 1);
    expect(row).toHaveLength(5);
    const sum = row.reduce((s, r) => s + r.v, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
  it("ffnSlots inside [0,1], 8 slots", () => {
    const s = ffnSlots(16, 8, 1, 1, false);
    expect(s).toHaveLength(8);
    for (const v of s) expect(v).toBeGreaterThanOrEqual(0);
    for (const v of s) expect(v).toBeLessThanOrEqual(1);
  });
  it("residCurve is monotonic increasing", () => {
    const p = residCurve(16);
    for (let i = 1; i < p.length; i++) expect(p[i].y).toBeGreaterThan(p[i - 1].y);
  });
  it("fallbackPreds is a flat neutral top-5", () => {
    const f = fallbackPreds("Ġnext");
    expect(f).toHaveLength(5);
    expect(f[0][1]).toBeGreaterThan(f[4][1]);
  });
});
