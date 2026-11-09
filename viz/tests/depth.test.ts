import { describe, it, expect } from "vitest";
import { bandOf, bandWidthPct, heat, mix } from "../src/engine/depth";

const B = { early: [1, 5], mid: [6, 11], late: [12, 16] } as const;
describe("depth bands", () => {
  it("band membership covers every layer exactly once", () => {
    for (let l = 1; l <= 16; l++) {
      expect(["early", "mid", "late"]).toContain(bandOf(B, l));
    }
    expect(bandOf(B, 1)).toBe("early");
    expect(bandOf(B, 5)).toBe("early");
    expect(bandOf(B, 6)).toBe("mid");
    expect(bandOf(B, 11)).toBe("mid");
    expect(bandOf(B, 12)).toBe("late");
    expect(bandOf(B, 16)).toBe("late");
  });
  it("band widths sum to 100%", () => {
    const s = bandWidthPct(B, 16, "early") + bandWidthPct(B, 16, "mid") + bandWidthPct(B, 16, "late");
    expect(s).toBeCloseTo(100, 5);
  });
  it("heat peaks mid-band and floors", () => {
    expect(heat(B, 9)).toBeCloseTo(0.889, 2); // near mid-band center (8.5)
    expect(heat(B, 11)).toBeGreaterThan(0.4);
    expect(heat(B, 1)).toBe(0.12); // floor clamps jumps
    const b2 = { early: [1, 10], mid: [11, 20], late: [21, 30] } as const;
    expect(heat(b2, 40)).toBe(0.12);
  });
  it("mix lightens toward white", () => {
    expect(mix("#000000", 0)).toBe("rgb(0,0,0)");
    expect(mix("#000000", 1)).toBe("rgb(115,115,115)");
  });
});
