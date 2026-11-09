import { describe, expect, it } from "vitest";
import { LAYER_REGISTRY, MODEL_KINDS } from "./registry";
import * as paramMath from "./paramMath";

const KINDS: readonly string[] = MODEL_KINDS;

describe("LAYER_REGISTRY integrity", () => {
  it("has a key for every MODEL_KINDS entry (unique kinds)", () => {
    expect(Object.keys(LAYER_REGISTRY).sort()).toEqual([...KINDS].sort());
    const seen = new Set<string>();
    for (const k of KINDS) {
      expect(seen.has(k)).toBe(false);
      seen.add(k);
    }
  });

  it("every spec has a non-empty label and summary", () => {
    for (const [kind, spec] of Object.entries(LAYER_REGISTRY)) {
      expect(spec.kind, kind).toBe(kind);
      expect(spec.label.length).toBeGreaterThan(0);
      expect(spec.summary.length).toBeGreaterThan(0);
    }
  });

  it("every spec has exactly one output", () => {
    for (const [kind, spec] of Object.entries(LAYER_REGISTRY)) {
      expect(spec.outputs.length, kind).toBe(1);
      for (const port of spec.outputs) {
        expect(port.name.length, kind).toBeGreaterThan(0);
        expect(port.label.length, kind).toBeGreaterThan(0);
      }
    }
  });

  it("props sanity: int titles unique, enum has options", () => {
    for (const [kind, spec] of Object.entries(LAYER_REGISTRY)) {
      const names = spec.props.map((p) => p.name);
      expect(new Set(names).size, kind).toBe(names.length);
      for (const p of spec.props) {
        if (p.kind === "enum") {
          expect(p.options?.length, kind + ":" + p.name).toBeGreaterThan(0);
        }
      }
    }
  });

  it("paramCount + inferShape are pure (same args -> same result)", () => {
    const inputProps = { vocab_size: 32768, ctx: 1024 };
    const base = LAYER_REGISTRY.input.inferShape(inputProps, []);
    const again = LAYER_REGISTRY.input.inferShape(inputProps, []);
    expect(again).toEqual(base);
    expect(base).toEqual({ s: 1024, d: 32768 });
    expect(LAYER_REGISTRY.input.paramCount(inputProps, base)).toBe(0);
  });
});

describe("layer spot-checks (dense GQA params, cross paramMath)", () => {
  const smokeBody = LAYER_REGISTRY.gqaAttention;
  const smokeOut = { d: 256, s: 1024 };

  it("gqaAttention params on d=256 heads=4 kv=2 equal pAttentionDense (196,608)", () => {
    const direct = paramMath.pAttentionDense({ d: 256, heads: 4, kv: 2 });
    expect(direct).toBe(196_608);
    expect(
      smokeBody.paramCount({ heads: 4, kv_heads: 2 }, smokeOut)
    ).toBe(direct);
  });

  it("swigluFfn params on ffn=1024 d=256 equal pFFNDense (786,432)", () => {
    const direct = paramMath.pFFNDense({ d: 256, ffn: 1024 });
    expect(direct).toBe(786_432);
    expect(
      LAYER_REGISTRY.swigluFfn.paramCount({ ffn: 1024 }, smokeOut)
    ).toBe(direct);
  });

  it("lmHead tied = 0, untied = vocab*d", () => {
    expect(LAYER_REGISTRY.lmHead.paramCount({ vocab_size: 32_768, tied: true }, smokeOut)).toBe(0);
    expect(
      LAYER_REGISTRY.lmHead.paramCount({ vocab_size: 32_768, tied: false }, smokeOut, [smokeOut])
    ).toBe(32_768 * 256); // untied head prices the matrix over the HIDDEN input d
  });

  it("layerStack counts ONLY outer pNorms: depth 4 d 256 = 2,304", () => {
    expect(paramMath.pNorms({ d: 256, layers: 4 })).toBe(2_304);
    expect(
      LAYER_REGISTRY.layerStack.paramCount({ depth: 4, d: 256, heads: 4, kv: 2, ffn: 1024 }, smokeOut)
    ).toBe(2_304);
  });

  it("rmsnorm weight-only params = d", () => {
    expect(LAYER_REGISTRY.rmsnorm.paramCount({}, smokeOut)).toBe(256);
  });
});

describe("shape inference + validation errors", () => {
  const ctx = { s: 1024, d: 32768 };

  it("input produces {s: ctx, d: vocab_size} and rejects non-positive props", () => {
    expect(LAYER_REGISTRY.input.inferShape({ vocab_size: 32768, ctx: 1024 }, [])).toEqual({ s: 1024, d: 32768 });
    expect(() => LAYER_REGISTRY.input.inferShape({ vocab_size: 0, ctx: 1024 }, [])).toThrow(/vocab_size/);
    expect(() => LAYER_REGISTRY.input.inferShape({ vocab_size: 32768, ctx: -1 }, [])).toThrow(/ctx/);
  });

  it("embedding propagates s, sets d, params = vocab*d via pEmbed", () => {
    const out = LAYER_REGISTRY.embedding.inferShape({ vocab_size: 32768, d: 256 }, [ctx]);
    expect(out).toEqual({ s: 1024, d: 256 });
    expect(LAYER_REGISTRY.embedding.paramCount({ vocab_size: 32768, d: 256 }, out)).toBe(32_768 * 256);
  });

  it("rmsnorm/rope/layerStack propagate the input shape", () => {
    const sh = { s: 512, d: 256 };
    expect(LAYER_REGISTRY.rmsnorm.inferShape({}, [sh])).toEqual(sh);
    expect(LAYER_REGISTRY.rope.inferShape({}, [sh])).toEqual(sh);
    expect(LAYER_REGISTRY.layerStack.inferShape({ depth: 4, d: 256 }, [sh])).toEqual(sh);
  });

  it("gqaAttention: mismatched q/k/v d throws a readable error", () => {
    const ok = { s: 1024, d: 256 };
    expect(() =>
      LAYER_REGISTRY.gqaAttention.inferShape({ heads: 4, kv_heads: 2 }, [ok, ok, { ...ok, d: 512 }])
    ).toThrow(/v.*d=512.*q.*d=256/);
    expect(() => LAYER_REGISTRY.gqaAttention.inferShape({ heads: 4, kv_heads: 5 }, [ok, ok, ok])).toThrow(/kv_heads/);
    expect(() =>
      LAYER_REGISTRY.gqaAttention.inferShape({ heads: 4 }, [ok, ok])
    ).toThrow(/missing input/);
  });

  it("residual requires bypass d == stream d", () => {
    const ok = { s: 512, d: 256 };
    expect(LAYER_REGISTRY.residual.inferShape({}, [ok, ok])).toEqual(ok);
    expect(() => LAYER_REGISTRY.residual.inferShape({}, [ok, { ...ok, d: 512 }])).toThrow(/bypass d=512 must equal stream d=256/);
    expect(LAYER_REGISTRY.residual.paramCount({}, ok)).toBe(0);
  });

  it("lmHead maps d -> vocab_size and validates tied is bool", () => {
    const out = LAYER_REGISTRY.lmHead.inferShape({ vocab_size: 32_768, tied: true }, [{ s: 1024, d: 256 }]);
    expect(out.d).toBe(32_768);
    expect(() => LAYER_REGISTRY.lmHead.paramCount({ vocab_size: 32_768, tied: "yes" }, out)).toThrow(/tied.*boolean/);
  });
});

describe("smoke-block digest (registry-only, wiring-free)", () => {
  // One dense transformer block + outer norm share: demonstrates the kinds
  // re-compose the Task-0 formulas exactly (embed excluded; counts are
  // per-layer body + norms only).
  it("one smoke layer: attention + ffn body + norms recompose pTotalDense", () => {
    const d = 256;
    const body =
      LAYER_REGISTRY.gqaAttention.paramCount({ heads: 4, kv_heads: 2 }, { d }) +
      LAYER_REGISTRY.swigluFfn.paramCount({ ffn: 1024 }, { d });
    const norms = paramMath.pNorms({ d, layers: 4 });
    // smoke pTotalDense = embed + 4*block + norms
    const rebuilt =
      paramMath.pEmbed({ vocab: 32_768, d }) +
      4 * body + norms;
    expect(rebuilt).toBe(paramMath.pTotalDense({ layers: 4, d: 256, heads: 4, kv: 2, ffn: 1024, vocab: 32_768 }));
    expect(rebuilt).toBe(12_323_072);
  });
});
