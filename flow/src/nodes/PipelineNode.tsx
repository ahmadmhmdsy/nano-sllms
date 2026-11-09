import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { PipelineNode } from "../store";
import { useFlowStore } from "../store";
import { INFER_CHECKPOINT_NOTE, WEBUI_URL } from "../inferHandoff";
// T3 registry promotion: the chat button is a registry FEATURE flag,
// not a kind-string check (inferNodeHasChatButton stays exported for tests).
import { CHAT_BUTTON_FEATURE } from "../nodes/registry";

import "./PipelineNode.css";

/** xyflow node renderer: title, prop badges, run-state dot, and one
 * Handle per registered port (direction -> Handle type). Port metadata
 * comes from the registry snapshot cached in the store (via /api/nodes).
 *
 * Task 11 infer node (render-only shortcut): kind=infer nodes get a
 * visible "Open in webui Chat" button plus the honest "checkpoint:" note
 * (the webui-recognized checkpoint path is NOT determinable client-side
 * — the ckpt-dir value arrives only as the connected edge, never as
 * node props). The click calls store.openInferHandoff, which App's
 * dialog consumes (manual instructions) and which attempts
 * window.open(WEBUI_URL) to the default webui location. */
function PipelineNode({ id, data, selected }: NodeProps<PipelineNode>) {
  const ports = data.ports ?? [];
  const propEntries = Object.entries(data.props ?? {});
  const openInfer = useFlowStore((s) => s.openInferHandoff);
  /* Task 10 dot wiring: toXYNodes threads the store's runStatus into every
   * node's data (honest MVP simplification — the backend runs one
   * whole-graph pipeline, so "running"/"done"/"error" are set-wide, not
   * per node; the tooltip shows the backend exit_code at terminal states).
   * Per-node phases are a future engine feature. */
  const state = data.runState ?? "idle";
  const terminal = state === "done" || state === "error";
  const title =
    "run state: " + state +
    (terminal && data.exitCode != null ? " (exit code " + String(data.exitCode) + ")" : "");
  return (
    <div className={"pipeline-node" + (selected ? " selected" : "")}>
      <div className="pipeline-node-head">
        <span className={"run-dot run-dot-" + state} title={title} />
        <span className="pipeline-node-title">{data.label ?? data.kind}</span>
      </div>
      {propEntries.length > 0 && (
        <div className="pipeline-node-props">
          {propEntries.map(([k, v]) => (
            <span key={k} className="prop-badge">
              {k}={String(v)}
            </span>
          ))}
        </div>
      )}
      {/* Task 11 render-only shortcut, declared by the registry feature
        * flag (T3) — no kind-string conditional here. */}
      {(data.features ?? []).includes(CHAT_BUTTON_FEATURE) && (
        <div className="pipeline-node-infer">
          <div className="infer-note">{INFER_CHECKPOINT_NOTE}</div>
          <button
            type="button"
            className="infer-open-webui"
            title={"attempt window.open(" + WEBUI_URL + ', "_blank") — manual instructions always shown'}
            onClick={() => openInfer?.(id)}
          >
            Open in webui Chat
          </button>
        </div>
      )}
      <div className="pipeline-node-ports">
        {ports.map((p) => {
          const isIn = p.direction === "in";
          const label = p.name;
          return (
            <div key={p.name} className={"pipeline-port pipeline-port-" + p.direction}>
              {isIn ? <span className="pipeline-port-label">{label}</span> : null}
              {/* Task M5.5 (a11y): handles render as plain non-focusable
                  divs — give them a tabindex + accessible name so keyboard
                  users can reach them (the canvas keydown handler cycles
                  Tab through them). Handle forwards ...rest to its div. */}
              <Handle
                id={p.name}
                type={isIn ? "target" : "source"}
                position={isIn ? Position.Left : Position.Right}
                isConnectable={true}
                tabIndex={0}
                aria-label={(isIn ? "input" : "output") + " port " + p.name + " of " + (data.label ?? data.kind)}
              />
              {!isIn ? <span className="pipeline-port-label">{label}</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default memo(PipelineNode);
