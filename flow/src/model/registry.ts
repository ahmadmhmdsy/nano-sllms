/**
 * Model layer-node registry (F2 plan Task 2).
 *
 * Same registry-driven pattern as flow/src/nodes/registry.ts, but for
 * tensor layer nodes of the Model tab: each kind declares typed tensor
 * ports, prop specs, a shape inference and a param count (pure functions).
 * All param formulas come from paramMath.ts — never re-derive them here.
 */
export type { Shape, TensorPort, LayerSpec, PropSpec } from "./layers/shared";

import type { LayerSpec } from "./layers/shared";
import { input } from "./layers/input";
import { embedding } from "./layers/embedding";
import { rmsnorm } from "./layers/rmsnorm";
import { rope } from "./layers/rope";
import { gqaAttention } from "./layers/gqaAttention";
import { swigluFfn } from "./layers/swigluFfn";
import { residual } from "./layers/residual";
import { lmHead } from "./layers/lmHead";
import { layerStack } from "./layers/layerStack";

/** Ordered kind list — drives the Model palette order. */
export const MODEL_KINDS = [
  "input",
  "embedding",
  "rmsnorm",
  "rope",
  "gqaAttention",
  "swigluFfn",
  "residual",
  "lmHead",
  "layerStack",
] as const;

export type ModelKind = (typeof MODEL_KINDS)[number];

// Any new kind must be added to BOTH the module import and MODEL_KINDS.
export const LAYER_REGISTRY: Record<ModelKind, LayerSpec> = {
  input,
  embedding,
  rmsnorm,
  rope,
  gqaAttention,
  swigluFfn,
  residual,
  lmHead,
  layerStack,
};
