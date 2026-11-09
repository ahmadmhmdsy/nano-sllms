import { useModelStore } from "./modelStore";
import { LAYER_REGISTRY, type PropSpec } from "./registry";
import type { PropWidget } from "../stores/registryStore";

/**
 * Model inspector (F2 Task 5), mirroring Inspector.tsx PropertiesPanel:
 * a props form built from the LAYER_REGISTRY spec of the selected node
 * (widget kind per PropSpec.kind — int -> number, bool -> checkbox,
 * enum -> select, float -> number; NO kind-string conditionals), plus a
 * live shape/param readout from the walk cache and the node's error list.
 *
 * "Reusing PropWidget" note: PropWidget is a type union, not a component,
 * so it is reused as the widget-type contract here; the render rows are
 * spec-kind-driven (the pipeline's text-widget fallback would silently
 * accept "abc" for an int prop — the Model inspector must not).
 */

/** Widget per PropSpec (pure, spec-driven). */
export function modelPropWidget(spec: PropSpec): PropWidget | "bool" | "enum" {
  if (spec.kind === "bool") return "bool";
  if (spec.kind === "enum") return "enum";
  if (spec.kind === "int" || spec.kind === "float") return "number";
  return "text";
}

function PropRow({
  spec,
  value,
  onChange,
}: {
  spec: PropSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const widget = modelPropWidget(spec);
  const inputId = "model-prop-" + spec.name;
  return (
    <div className="prop-row">
      <label htmlFor={inputId}>
        {spec.name}
        <span className="model-prop-kind"> ({spec.kind})</span>
      </label>
      {widget === "bool" ? (
        <input
          id={inputId}
          type="checkbox"
          checked={value === true}
          onChange={(e) => onChange(e.target.checked)}
        />
      ) : widget === "enum" ? (
        <select id={inputId} value={typeof value === "string" ? value : ""} onChange={(e) => onChange(e.target.value)}>
          {!spec.options?.includes(String(value)) && <option value="">(unset)</option>}
          {(spec.options ?? []).map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : widget === "number" ? (
        <input
          id={inputId}
          type="number"
          step={spec.kind === "int" ? 1 : "any"}
          min={spec.min}
          max={spec.max}
          value={typeof value === "number" && Number.isFinite(value) ? String(value) : ""}
          onChange={(e) => {
            if (e.target.value === "") onChange(undefined);
            else onChange(Number(e.target.value));
          }}
        />
      ) : (
        <input
          id={inputId}
          type="text"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}

function ModelPropertiesPanel() {
  const selection = useModelStore((s) => s.selection);
  const nodes = useModelStore((s) => s.nodes);
  const updateProps = useModelStore((s) => s.updateProps);
  const node = selection === null ? null : nodes.find((n) => n.id === selection);
  if (!node) return <p className="inspector-empty">select a layer node on the canvas</p>;

  const spec = LAYER_REGISTRY[node.kind as keyof typeof LAYER_REGISTRY];
  if (!spec) {
    return <p className="inspector-empty">unknown kind '{node.kind}' — not in the layer registry</p>;
  }

  const setProp = (name: string, value: unknown) => {
    const next = { ...node.props };
    if (value === undefined) delete next[name];
    else next[name] = value;
    updateProps(node.id, next);
  };

  return (
    <>
      <p className="prop-kind">{spec.label} <code>({node.id})</code></p>
      <p className="inspector-empty">{spec.summary}</p>
      {spec.props.length === 0 ? (
        <p className="inspector-empty">(no editable properties)</p>
      ) : (
        spec.props.map((p) => (
          <PropRow key={p.name} spec={p} value={node.props[p.name]} onChange={(v) => setProp(p.name, v)} />
        ))
      )}
    </>
  );
}

/** Live walk readout + per-node errors for the selected node; the graph
 * totals live in the canvas header strip. */
function ModelReadoutPanel() {
  const selection = useModelStore((s) => s.selection);
  const walk = useModelStore((s) => s.walk);
  const nodes = useModelStore((s) => s.nodes);

  const node = selection === null ? null : nodes.find((n) => n.id === selection);
  if (!node) return null;
  const shape = walk.shapes.get(node.id);
  const params = walk.params[node.id];
  const errors = walk.errors.filter((e) => e.nodeId === node.id);
  return (
    <div className="model-readout" aria-label="live shape and parameter readout">
      <p className="preview-section">live readout</p>
      <p className="model-readout-line">
        shape:{" "}
        {shape ? (
          <code>
            [b{String(shape.b ?? "?")}, s{String(shape.s ?? "?")}, d{String(shape.d)}]
          </code>
        ) : (
          <span className="inspector-empty">unavailable (see errors)</span>
        )}
      </p>
      <p className="model-readout-line">
        params: {params !== undefined ? <code>{params.toLocaleString("en-US")}</code> : <span className="inspector-empty">n/a</span>}
      </p>
      {errors.length > 0 && (
        <>
          <p className="preview-section">errors ({errors.length})</p>
          <ul className="model-error-list">
            {errors.map((e, i) => (
              <li key={i} className="model-error-item">
                {e.message}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default function ModelInspector() {
  return (
    <aside className="inspector" aria-label="model inspector">
      <p className="modal-title">layer properties</p>
      <ModelPropertiesPanel />
      <ModelReadoutPanel />
    </aside>
  );
}
