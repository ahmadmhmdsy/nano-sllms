// Registry — the ONLY place models enter the app (plan §3, G1).
// Vite glob-imports every models/*.yaml (raw) + models/*.overrides.ts (escape
// hatch G2), validates with zod, exposes the typed list. A bad YAML produces a
// LOUD fix-hint, never a blank page.
import { parse as parseYAML } from "yaml";
import { modelSpecSchema, type ModelSpec, type Facts } from "../schema/spec";
import { DEFAULT_FACTS } from "./defaults";

const yamlRaw = import.meta.glob("/models/*.yaml", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const overrideMods = import.meta.glob("/models/*.overrides.ts", { eager: true }) as Record<
  string,
  { default?: { facts?: Partial<Facts>; hybrid?: Record<string, unknown> } }
>;

function findOverride(id: string): ((s: any) => any) | null {
  for (const [path, mod] of Object.entries(overrideMods)) {
    const base = path.replace(/\\/g, "/").split("/").pop()!;
    if (base === id + ".overrides.ts" && mod.default) {
      return (spec: any) => {
        spec.facts = { ...spec.facts, ...mod.default!.facts };
        if (mod.default!.hybrid) spec.hybrid = { ...spec.hybrid, ...mod.default!.hybrid };
        return spec;
      };
    }
  }
  return null;
}

export interface RegistryResult {
  models: ModelSpec[];
  problems: string[];
}
const result: RegistryResult = { models: [], problems: [] };
(function load() {
  for (const [path, raw] of Object.entries(yamlRaw)) {
    const base = path.replace(/\\/g, "/").split("/").pop()!;
    if (base.startsWith("_")) continue; // kickers: _template.yaml etc.
    const parsed: any = parseYAML(raw);
    const id = typeof parsed.overrides === "string" ? parsed.overrides : base.replace(/\.yaml$/, "");
    const candidate = findOverride(id)?.(parsed) ?? parsed;
    const res = modelSpecSchema.safeParse(candidate);
    if (!res.success) {
      result.problems.push(
        base + ": " + res.error.issues.map((i) => i.path.join(".") + " " + i.message).join(" | ")
      );
      continue;
    }
    result.models.push(res.data);
  }
})();

export const MODELS: ModelSpec[] = result.models;
export function factoryDefaults(spec: ModelSpec): NonNullable<Facts> {
  return { ...DEFAULT_FACTS, ...(spec.facts ?? {}) };
}
export function doctorBaseline(): string {
  return result.problems.length
    ? result.problems.join("\n")
    : MODELS.length + " model YAMLs validated OK";
}
