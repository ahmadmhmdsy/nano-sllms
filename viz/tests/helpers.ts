// test helper: load a model YAML from disk, apply overrides, zod-validate.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { modelSpecSchema, type ModelSpec } from "../src/schema/spec";

export function loadModelYaml(fname: string): ModelSpec {
  const raw = readFileSync(join(__dirname, "..", "models", fname), "utf8");
  const parsed: any = parse(raw);
  // apply the same override convention the registry uses (id field, not filename)
  return modelSpecSchema.parse(parsed) as ModelSpec;
}
export const MODELS_FOR_TESTS: Record<string, ModelSpec> = {
  nano: loadModelYaml("nano-target.yaml"),
  smollm2: loadModelYaml("smollm2-135m.yaml"),
  qwen38: loadModelYaml("qwen3.8-flash-next.yaml"),
};
