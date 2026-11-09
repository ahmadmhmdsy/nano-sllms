// Param math — the canonical implementations live in the shared package
// (packages/param-math, src/index.ts). This module re-exports them unchanged
// so every existing `from "<app>/engine/params"` import keeps working
// (including the HybridCfg type, which the shared package owns).
export * from "../../../packages/param-math/src/index";
