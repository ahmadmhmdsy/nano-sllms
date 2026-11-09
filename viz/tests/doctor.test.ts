// pnpm doctor — the PASS/FAIL gate for schema + arithmetic + structure.
// Runs on every registry/config change (plan §5 validation habits).
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { modelSpecSchema } from "../src/schema/spec";
import { MODELS_FOR_TESTS } from "./helpers";
import { pTotalDense, pTotalHybrid } from "../src/engine/params";

function yamlFiles(): string[] {
  return readdirSync(join(__dirname, "..", "models")).filter((f) => f.endsWith(".yaml") && !f.startsWith("_"));
}
describe("doctor — model registry", () => {
  it("every yaml parses + validates against the zod schema", () => {
    const fails: string[] = [];
    for (const f of yamlFiles()) {
      const raw = readFileSync(join(__dirname, "..", "models", f), "utf8");
      const res = modelSpecSchema.safeParse(parse(raw));
      if (!res.success) {
        fails.push(f + ": " + res.error.issues.map((i) => i.path.join(".") + " " + i.message).join(" | "));
      }
    }
    expect(fails, "fix hint: schema mismatch — see listed files").toEqual([]);
  });
  it("unique ids and accents across the registry", () => {
    const ids = Object.keys(MODELS_FOR_TESTS).map((k) => MODELS_FOR_TESTS[k].id);
    expect(new Set(ids).size).toBe(ids.length);
    const acc = Object.keys(MODELS_FOR_TESTS).map((k) => MODELS_FOR_TESTS[k].accent);
    expect(new Set(acc).size).toBe(acc.length);
  });
  it("every model cites at least one external/repo source", () => {
    for (const k of Object.keys(MODELS_FOR_TESTS)) {
      expect(MODELS_FOR_TESTS[k].honesty.sources.length).toBeGreaterThan(0);
    }
  });
  it("every band triple covers the full layer range, contiguous", () => {
    for (const k of Object.keys(MODELS_FOR_TESTS)) {
      const m = MODELS_FOR_TESTS[k];
      const L = m.shape.layers;
      const { early, mid, late } = m.bands;
      expect(early[0]).toBe(1);
      expect(early[1] + 1).toBe(mid[0]);
      expect(mid[1] + 1).toBe(late[0]);
      expect(late[1]).toBe(L);
    }
  });
  it("anchor params match published repo numbers", () => {
    expect(pTotalDense(MODELS_FOR_TESTS.nano.shape)).toBe(226_526_208);
    expect(pTotalDense(MODELS_FOR_TESTS.smollm2.shape)).toBe(134_515_008);
    const q = pTotalHybrid(MODELS_FOR_TESTS.qwen38);
    expect(q).toBeGreaterThan(1.2e11);
    expect(q).toBeLessThan(1.27e11);
  });
});
