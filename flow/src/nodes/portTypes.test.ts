import { describe, it, expect } from "vitest";
import { portTypesCompatible } from "./portTypes";

/* T3: the compat rule now lives in its own module; these tests pin the
 * prefix-compatibility semantics that store.validateConnect relies on. */
describe("portTypesCompatible (prefix-compatible rule)", () => {
  it("accepts exact type matches", () => {
    expect(portTypesCompatible("raw-dir", "raw-dir")).toBe(true);
    expect(portTypesCompatible("shard-dir", "shard-dir")).toBe(true);
  });

  it("accepts a descriptive suffix on either side (base prefix match)", () => {
    expect(portTypesCompatible("raw-dir", "raw-dir: dataset name/rows")).toBe(true);
    expect(portTypesCompatible("raw-dir: dataset name/rows", "raw-dir")).toBe(true);
  });

  it("rejects distinct base types", () => {
    expect(portTypesCompatible("raw-dir", "cleaned-dir")).toBe(false);
    expect(portTypesCompatible("shard-dir", "ckpt-dir")).toBe(false);
    expect(portTypesCompatible("ckpt-dir", "report")).toBe(false);
  });

  it("ignores surrounding whitespace around the base", () => {
    expect(portTypesCompatible(" raw-dir ", "raw-dir")).toBe(true);
  });

  it("only the first ':...' segment forms the description (distinct bases never match)", () => {
    expect(portTypesCompatible("raw-dir: x", "shard-dir: x")).toBe(false);
  });
});
