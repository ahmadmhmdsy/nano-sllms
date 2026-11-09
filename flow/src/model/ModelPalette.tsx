import { MODEL_PALETTE_KINDS, useModelStore } from "./modelStore";
import { LAYER_REGISTRY } from "./registry";

/** Model palette (F2 Task 5): one card per LAYER_REGISTRY kind, in the
 * registry's MODEL_KINDS order. Click-to-add per the task contract — the
 * card title carries the spec summary (spec-driven, no kind conditionals). */
export default function ModelPalette() {
  const addNode = useModelStore((s) => s.addNode);

  return (
    <aside className="palette" aria-label="model layer palette">
      <p className="palette-status">layers — click to add</p>
      <div className="palette-cards">
        {MODEL_PALETTE_KINDS.map((kind) => {
          const spec = LAYER_REGISTRY[kind as keyof typeof LAYER_REGISTRY];
          return (
            <button
              key={kind}
              className="palette-card model-palette-card"
              data-kind={kind}
              title={spec ? spec.kind + ": " + spec.summary : kind}
              onClick={() => addNode(kind)}
            >
              {spec?.label ?? kind}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
