import { useFlowStore, FALLBACK_REGISTRY } from "./store";

/** Palette kinds come from GET /api/nodes (cached in the store). The T1
 * hard-coded kind list is removed. A static fallback copy of the registry
 * is used ONLY when the registry fetch failed (store.registryError); that
 * is documented UX, not a hidden default. A card drag starts HTML5 dnd
 * with the kind in "application/flow-node-kind". */
export function usePaletteKinds(): {
  kinds: string[];
  source: "registry" | "fallback" | "loading";
} {
  const registry = useFlowStore((s) => s.registry);
  const registryError = useFlowStore((s) => s.registryError);
  if (registry) {
    return { kinds: registry.valid_kinds, source: "registry" };
  }
  if (registryError) {
    return { kinds: FALLBACK_REGISTRY.valid_kinds, source: "fallback" };
  }
  return { kinds: [], source: "loading" };
}

export default function Palette() {
  const collapsed = useFlowStore((s) => s.paletteCollapsed);
  const setCollapsed = useFlowStore((s) => s.setPaletteCollapsed);
  const { kinds, source } = usePaletteKinds();

  return (
    <aside className={"palette" + (collapsed ? " palette-collapsed" : "")}>
      <button
        onClick={() => setCollapsed(!collapsed)}
        aria-label={collapsed ? "Expand palette" : "Collapse palette"}
      >
        {collapsed ? "»" : "«"}
      </button>
      {collapsed ? null : source === "loading" ? (
        <p className="palette-status" aria-live="polite">
          loading registry…
        </p>
      ) : (
        <div className="palette-cards">
          {source === "fallback" && (
            <p className="palette-status palette-error" role="status">
              registry unavailable — built-in kinds
            </p>
          )}
          {kinds.map((kind) => (
            <div
              key={kind}
              className="palette-card"
              draggable
              data-kind={kind}
              onDragStart={(e) => {
                e.dataTransfer.setData("application/flow-node-kind", kind);
                e.dataTransfer.effectAllowed = "copy";
              }}
            >
              {kind}
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
