import { useCallback, useEffect, useRef, useState } from "react";
import { ReactFlow, Background, ReactFlowProvider, useReactFlow } from "@xyflow/react";
import type { Connection, EdgeChange, NodeChange } from "@xyflow/react";
import { useModelStore, toXYNodes, toXYEdges } from "./modelStore";
import ModelNode from "./ModelNode";
import ModelPalette from "./ModelPalette";
import ModelInspector from "./ModelInspector";
import { api, errorMessage } from "../api";
import { isValidFlowName } from "../stores/graphStore";
import "@xyflow/react/dist/style.css";
import "./ModelCanvas.css";

const nodeTypes = { model: ModelNode };

/** Header strip (F2 Task 5): model name input, live total param count,
 * error count, and Open/Save via GET/POST /api/models — mirroring the
 * pipeline Toolbar's modal wiring (in-flight guard, slug validation,
 * backdrop dismiss). */
function ModelHeader() {
  const name = useModelStore((s) => s.name);
  const setName = useModelStore((s) => s.setModelName);
  const dirty = useModelStore((s) => s.dirty);
  const total = useModelStore((s) => s.walk.total);
  const errorCount = useModelStore((s) => s.walk.errors.length);
  const save = useModelStore((s) => s.save);
  const open = useModelStore((s) => s.open);
  const setError = useModelStore((s) => s.setError);

  /* File modal mode: null | "open" (model list picker) | "save" (name
   * prompt). Mirrors the pipeline Toolbar modals. */
  const [modal, setModal] = useState<null | "open" | "save">(null);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (modal === "save") inputRef.current?.focus();
  }, [modal]);

  const submitSave = async () => {
    if (saving) return; // in-flight guard (mirrors pipeline Save)
    const modelName = inputRef.current?.value ?? "";
    if (!isValidFlowName(modelName)) {
      setError("model name must match [a-z0-9-]{1,64} (lowercase letters, digits, dashes)");
      return;
    }
    setSaving(true);
    try {
      const ok = await save(modelName);
      if (ok) {
        setModal(null);
        setError(null);
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="model-header">
      <input
        className="model-name-input"
        aria-label="model name"
        placeholder="model-name (slug)"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <span className="model-total" title="total parameter count over error-free nodes">
        params: {total.toLocaleString("en-US")}
      </span>
      <span
        className={"model-error-count" + (errorCount > 0 ? " model-error-count-bad" : "")}
        role="status"
        title="walk errors — see node badges and the inspector error list"
      >
        errors: {errorCount}
      </span>
      <button id="model-open" aria-label="Open saved model" onClick={() => setModal("open")}>
        Open
      </button>
      <button id="model-save" aria-label="Save current model" onClick={() => setModal("save")}>
        Save{dirty ? " *" : ""}
      </button>

      {modal === "open" && (
        <ModelPicker
          onPick={(picked) => {
            setModal(null);
            void open(picked);
          }}
          onDismiss={() => setModal(null)}
        />
      )}

      {modal === "save" && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div
            className="modal"
            role="dialog"
            aria-label="Save model"
            onClick={(e) => e.stopPropagation()}
          >
            <label htmlFor="model-save-name">model name</label>
            <input
              ref={inputRef}
              id="model-save-name"
              placeholder="my-model"
              defaultValue={name}
              onKeyDown={(e) => {
                if (e.key === "Enter") void submitSave();
              }}
            />
            <div className="modal-actions">
              <button onClick={() => setModal(null)}>Cancel</button>
              <button disabled={saving} onClick={() => void submitSave()}>
                {saving ? "saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Body for the Open modal: names come straight from GET /api/models
 * (api.getModels). Mirrors the pipeline FlowPicker. */
function ModelPicker({
  onPick,
  onDismiss,
}: {
  onPick: (name: string) => void;
  onDismiss: () => void;
}) {
  const [names, setNames] = useState<string[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getModels()
      .then((n) => {
        if (!cancelled) setNames(n);
      })
      .catch((e) => {
        if (!cancelled) setListError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="modal-backdrop" onClick={onDismiss}>
      <div className="modal" role="dialog" aria-label="Open model" onClick={(e) => e.stopPropagation()}>
        <p className="modal-title">saved models</p>
        {listError !== null && <p className="honesty-label">{listError}</p>}
        {names === null && <p className="modal-note">loading…</p>}
        {names !== null && names.length === 0 && (
          <p className="modal-note">
            no saved models yet — committed example graphs live under flow/models/
          </p>
        )}
        <ul className="modal-list">
          {(names ?? []).map((n) => (
            <li key={n}>
              <button onClick={() => onPick(n)}>{n}</button>
            </li>
          ))}
        </ul>
        <div className="modal-actions">
          <button onClick={onDismiss}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

/** Canvas: controlled ReactFlow wired into the model store (same apply
 * changes / validated connect / toast pattern as FlowCanvas), plus the
 * empty-state hint pointing at the committed example graphs. */
function ModelCanvasInner() {
  const nodes = useModelStore((s) => s.nodes);
  const edges = useModelStore((s) => s.edges);
  const walk = useModelStore((s) => s.walk);
  const projection = useModelStore((s) => s.projection);
  const error = useModelStore((s) => s.error);
  const setError = useModelStore((s) => s.setError);
  const applyNodesChanges = useModelStore((s) => s.applyNodesChanges);
  const applyEdgesChanges = useModelStore((s) => s.applyEdgesChanges);
  const connect = useModelStore((s) => s.connect);

  const { screenToFlowPosition } = useReactFlow();

  /* Toast clears itself a moment after appearing (canvas owns the timer,
   * mirroring FlowCanvas). */
  useEffect(() => {
    if (!error) return;
    const t = window.setTimeout(() => setError(null), 4000);
    return () => window.clearTimeout(t);
  }, [error, setError]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => applyNodesChanges(changes),
    [applyNodesChanges],
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => applyEdgesChanges(changes),
    [applyEdgesChanges],
  );
  const onConnect = useCallback((c: Connection) => connect(c), [connect]);

  return (
    <main className="canvas" aria-label="model canvas">
      <ReactFlow
        nodes={projection.length > 0 ? projection : toXYNodes({ nodes, edges }, walk)}
        edges={toXYEdges(edges)}
        nodeTypes={nodeTypes}
        /* Task M6.1: React Flow v12's built-in node virtualization (same
         * rationale as FlowCanvas) — once measured, only viewport-overlapping
         * nodes render, so panning a large model graph no longer re-renders
         * every layer node per frame. */
        onlyRenderVisibleElements
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_, node) => useModelStore.getState().setSelection(node.id)}
        onPaneClick={() => useModelStore.getState().setSelection(null)}
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
        }}
        onDrop={(e) => {
          e.preventDefault();
          const kind = e.dataTransfer.getData("application/model-layer-kind");
          if (!kind) return;
          useModelStore.getState().addNode(kind, screenToFlowPosition({ x: e.clientX, y: e.clientY }));
        }}
      >
        <Background />
      </ReactFlow>

      {nodes.length === 0 && (
        <div className="model-empty-state" role="note">
          <p>empty model — add layer nodes from the palette, then wire tensors between them.</p>
          <p className="honesty-label">
            committed example graphs (smoke/pilot/target decoders) live under flow/models/ —
            Open them once saved.
          </p>
        </div>
      )}

      {error !== null && (
        <div className="toast" role="alert" aria-live="polite">
          {error}
        </div>
      )}
    </main>
  );
}

/** Model mode layout: palette | canvas (with header strip) | inspector. */
export default function ModelCanvas() {
  return (
    <ReactFlowProvider>
      <div className="model-mode">
        <ModelPalette />
        <div className="model-main">
          <ModelHeader />
          <ModelCanvasInner />
        </div>
        <ModelInspector />
      </div>
    </ReactFlowProvider>
  );
 }
