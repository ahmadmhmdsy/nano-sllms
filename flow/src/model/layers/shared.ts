// Shared LayerSpec types + validation helpers for the model layer registry.
// Type/prop validation throws readable Errors — the frontend renders them
// verbatim in the node error badge.

export interface TensorPort {
  name: string;
  label: string;
}

export interface PropSpec {
  name: string;
  kind: "int" | "float" | "bool" | "enum";
  options?: string[];
  default?: number | boolean;
  min?: number;
  max?: number;
}

/** Tensor shape flowing between layer nodes. `d` is always required;
 * for the `input` node `d` semantically carries the vocab index. */
export interface Shape {
  b?: number;
  s?: number;
  d: number;
}

export interface LayerSpec {
  kind: string;
  label: string;
  summary: string;
  inputs: TensorPort[];  // 0 (input) .. 3 (gqaAttention q/k/v)
  outputs: TensorPort[]; // exactly 1 everywhere
  props: PropSpec[];
  inferShape(props: Record<string, unknown>, inputs: Shape[]): Shape;
  /** `inputs` lets projection heads price a matrix over the HIDDEN input d
   * (the output d is the vocab they project onto, not feature width). */
  paramCount(props: Record<string, unknown>, out: Shape, inputs?: Shape[]): number;
}

/** Validate a numeric prop against its spec; returns the coerced int/float. */
export function numProp(name: string, raw: unknown, spec: PropSpec): number {
  const v = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(v)) throw new Error(`prop '${name}' must be a finite number (got ${JSON.stringify(raw)})`);
  if (spec.kind === "int" && !Number.isInteger(v)) throw new Error(`prop '${name}' must be an integer (got ${v})`);
  if (spec.min !== undefined && v < spec.min) throw new Error(`prop '${name}' must be >= ${spec.min} (got ${v})`);
  if (spec.max !== undefined && v > spec.max) throw new Error(`prop '${name}' must be <= ${spec.max} (got ${v})`);
  return v;
}

/** Validate a bool prop, tolerating booleans only. */
export function boolProp(name: string, raw: unknown): boolean {
  if (typeof raw !== "boolean") throw new Error(`prop '${name}' must be a boolean (got ${JSON.stringify(raw)})`);
  return raw;
}

/** Assert an input shape is present and carries a positive `d`. */
export function requireIn(inputs: Shape[], index: number, kind: string, port: TensorPort): Shape {
  const sh = inputs[index];
  if (!sh) throw new Error(`${kind}: missing input '${port.name}'`);
  if (typeof sh.d !== "number" || !(sh.d > 0) || !Number.isInteger(sh.d)) {
    throw new Error(`${kind}: input '${port.name}' has invalid d (${String(sh.d)})`);
  }
  return sh;
}
