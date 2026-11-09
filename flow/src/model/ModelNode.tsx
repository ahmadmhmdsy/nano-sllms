import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { ModelNode } from "./modelStore";
import "./ModelNode.css";

/** Spec-driven tensor-layer node renderer (F2 Task 5).
 *
 * Everything kind-specific arrives in node.data from the registry via
 * toXYNodes: label, input/output ports, walk shape/params/error. There
 * are NO kind-string conditionals in this render — adding a new layer
 * kind to LAYER_REGISTRY needs zero renderer changes.
 *
 * Layout mirrors PipelineNode: handles per declared input port on the
 * left, the single output port on the right, param badge + red error
 * badge (tooltip = the full walk error message) in the head.
 *
 * Task M3.1 visual error indication: the wrapper itself carries the
 * state — `model-node-error` when the walk reported an error (red
 * border), `model-node-uncomputed` when no shape was inferred
 * (dashed border) — and the full error message is the wrapper's
 * title tooltip, so hovering anywhere on the node shows it. The "!"
 * and "?" badges in the head mirror the same two states.
 */
function ModelNode({ data, selected }: NodeProps<ModelNode>) {
  const shape = data.shape;
  const shapeText =
    shape === null ? "" : "[b" + String(shape.b ?? "?") + ", s" + String(shape.s ?? "?") + ", d" + String(shape.d) + "]";
  const wrapperClass =
    "model-node" +
    (selected ? " selected" : "") +
    (data.error !== null ? " model-node-error" : "") +
    (data.shape === null ? " model-node-uncomputed" : "");
  return (
    <div className={wrapperClass} title={data.error ?? undefined}>
      <div className="model-node-head">
        <span className="model-node-title" title={data.kind}>
          {data.label}
        </span>
        <span
          className={
            "model-node-params" + (data.params === null ? " model-node-params-none" : "")
          }
          title={data.error ?? "parameters: " + String(data.params ?? "n/a")}
        >
          {data.params === null ? "∅" : formatParams(data.params)}
        </span>
        {data.shape === null && (
          <span
            className="model-node-uncomputed-badge"
            role="status"
            title="output shape not inferred"
          >
            ?
          </span>
        )}
        {data.error !== null && (
          <span className="model-node-error-badge" role="status" title={data.error}>
            !
          </span>
        )}
      </div>
      {shapeText !== "" && (
        <div className="model-node-shape" title={"inferred output shape " + shapeText}>
          {shapeText}
        </div>
      )}
      <div className="model-node-ports">
        {data.inputs.map((p) => (
          <div key={p.name} className="model-port model-port-in">
            <span className="model-port-label">{p.label}</span>
            <Handle id={p.name} type="target" position={Position.Left} isConnectable={true} />
          </div>
        ))}
        {data.outputs.map((p) => (
          <div key={p.name} className="model-port model-port-out">
            <span className="model-port-label">{p.label}</span>
            <Handle id={p.name} type="source" position={Position.Right} isConnectable={true} />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Compact param readout: exact integers stay exact; 1e7+ gets a k/M
 * suffix with the exact value in the title tooltip. */
function formatParams(n: number): string {
  if (n < 1_000_000) return n.toLocaleString("en-US");
  const m = n / 1_000_000;
  return (m >= 100 ? m.toFixed(0) : m.toFixed(1)) + "M";
}

export default memo(ModelNode);
