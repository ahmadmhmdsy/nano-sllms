import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState, type ChangeEvent } from "react";
import Palette from "./Palette";
import FlowCanvas from "./FlowCanvas";
import Inspector from "./Inspector";
import { useFlowStore, isValidFlowName } from "./store";
import { api, errorMessage } from "./api";
import { serializeFlow, parseFlowDocument } from "./stores/flowDocument";
import InferHandoffDialog from "./InferHandoffDialog";
/* Shared UI library (packages/ui-components): the ConfirmDialog, Spinner
 * and ErrorBoundary implementations were moved here from flow/ (M2.4 /
 * M2.5 / pre-M2) - flow/ now imports the shared ones. */
import { ConfirmDialog, ErrorBoundary, Spinner } from "../../packages/ui-components/src/index";
import "./App.css";

/* Task M6.2: code splitting — ModelCanvas (and its @xyflow/react
 * dependency) is only needed in Model mode, so it lazy-loads as a separate
 * chunk on first mode switch instead of shipping in the initial bundle. */
const ModelCanvas = lazy(() => import("./model/ModelCanvas"));

/* Throttle window: a focus event refetches the registry at most once per
 * this window (the mount fetch itself counts as an attempt). */
const REGISTRY_REFETCH_THROTTLE_MS = 5000;

/** Cache the /api/nodes registry snapshot in the store on mount; on
 * failure record the error so Palette falls back to its static copy.
 * A refetch is also attempted whenever the window regains focus (at most
 * once per throttle window) so a late-starting backend heals without a
 * manual reload (T8 review MINOR-9). */
export default function App() {
  /* Top-level editor mode (F2 Task 5): the pipeline and model canvases are
   * deliberately DIFFERENT components with separate stores — no shared
   * mode state leaks into the pipeline graph actions. */
  const [mode, setMode] = useState<"pipeline" | "model">("pipeline");
  const setRegistry = useFlowStore((s) => s.setRegistry);
  const setRegistryError = useFlowStore((s) => s.setRegistryError);

  useEffect(() => {
    let cancelled = false;
    let lastAttempt = Date.now();
    const refetch = () => {
      if (cancelled) return;
      api
        .getNodes()
        .then((r) => {
          if (!cancelled) setRegistry(r);
        })
        .catch((e) => {
          if (!cancelled) setRegistryError(errorMessage(e));
        });
    };
    refetch();
    const onFocus = () => {
      if (Date.now() - lastAttempt >= REGISTRY_REFETCH_THROTTLE_MS) {
        lastAttempt = Date.now();
        refetch();
      }
    };
    window.addEventListener("focus", onFocus);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", onFocus);
    };
  }, [setRegistry, setRegistryError]);

  /* Task M6.3: below 1200px the palette and inspector become toggleable
   * overlays (App.css media query). The store collapse flags double as
   * the overlay open/closed state; on load — and when crossing into the
   * narrow range — both overlays start closed so the canvas owns the
   * viewport until the user opens one. Resizing within the narrow range
   * leaves the user's open/closed state untouched. */
  useLayoutEffect(() => {
    const mq = window.matchMedia("(max-width: 1200px)");
    let narrow = mq.matches;
    if (narrow) {
      const st = useFlowStore.getState();
      st.setPaletteCollapsed(true);
      st.setInspectorCollapsed(true);
    }
    const onChange = () => {
      if (mq.matches === narrow) return;
      narrow = mq.matches;
      if (narrow) {
        const st = useFlowStore.getState();
        st.setPaletteCollapsed(true);
        st.setInspectorCollapsed(true);
      }
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return (
    <ErrorBoundary>
      <div className="app-root">
        <div className="mode-switch" role="tablist" aria-label="editor mode">
        <button
          role="tab"
          aria-selected={mode === "pipeline"}
          className={"mode-tab" + (mode === "pipeline" ? " mode-tab-active" : "")}
          onClick={() => setMode("pipeline")}
        >
          Pipeline
        </button>
        <button
          role="tab"
          aria-selected={mode === "model"}
          className={"mode-tab" + (mode === "model" ? " mode-tab-active" : "")}
          onClick={() => setMode("model")}
        >
          Model
        </button>
      </div>
      {mode === "pipeline" ? (
        <>
          <Toolbar />
          <div className="shell">
            <Palette />
            <FlowCanvas />
            <Inspector />
          </div>
        </>
      ) : (
        <Suspense fallback={<Spinner />}>
          <ModelCanvas />
        </Suspense>
      )}
      <InferHandoffDialog />
      </div>
    </ErrorBoundary>
  );
}

/** Header toolbar. Open = GET /api/flows + picker modal -> GET the picked
 * flow -> store.openFlow (which also resets the xyflow projection via
 * setGraph; an empty projection just means an unmeasured canvas, visually
 * benign). Save = modal name prompt with client-side slug validation ->
 * store.saveFlow (PUT /api/flows/{name}). Errors surface through the
 * existing store.error toast channel (rendered in FlowCanvas). Both are
 * wired (Task 10): Validate toasts an "OK:" success marker; Run is
 * disabled while a run is live and routes an unsaved graph through the
 * Save modal first. */
function Toolbar() {
  const saveFlow = useFlowStore((s) => s.saveFlow);
  const openFlow = useFlowStore((s) => s.openFlow);
  const setGraph = useFlowStore((s) => s.setGraph);
  const setError = useFlowStore((s) => s.setError);
  const validateFlow = useFlowStore((s) => s.validateFlow);
  const runGraph = useFlowStore((s) => s.runGraph);
  const currentFlowName = useFlowStore((s) => s.currentFlowName);
  const runStatus = useFlowStore((s) => s.runStatus);
  const canUndo = useFlowStore((s) => s.past.length > 0);
  const canRedo = useFlowStore((s) => s.future.length > 0);
  const graph = useFlowStore((s) => s.graph);
  /* Task M2.4: destructive-action confirmation (Clear canvas / Delete
   * flow). The pending request lives in the store (like
   * inferHandoffNodeId) so the wiring is unit-testable through
   * useFlowStore.getState() without a DOM testing library. */
  const confirm = useFlowStore((s) => s.confirm);
  const openConfirm = useFlowStore((s) => s.openConfirm);
  const dismissConfirm = useFlowStore((s) => s.dismissConfirm);
  const deleteFlow = useFlowStore((s) => s.deleteFlow);
  /* Task M2.2 node search: the query filters canvas nodes (highlight/dim
   * via toXYNodes classNames); Enter cycles through the matches. */
  const searchQuery = useFlowStore((s) => s.searchQuery);
  const setSearchQuery = useFlowStore((s) => s.setSearchQuery);
  /* Task M6.3: the palette/inspector collapse flags double as the
   * overlay open/closed state below 1200px (see App.css); the toggle
   * buttons at the end of the toolbar flip them. */
  const paletteCollapsed = useFlowStore((s) => s.paletteCollapsed);
  const setPaletteCollapsed = useFlowStore((s) => s.setPaletteCollapsed);
  const inspectorCollapsed = useFlowStore((s) => s.inspectorCollapsed);
  const setInspectorCollapsed = useFlowStore((s) => s.setInspectorCollapsed);

  /* File modal mode: null | "open" (flow list picker) | "save" (name
   * prompt). */
  const [modal, setModal] = useState<null | "open" | "save">(null);
  /* In-flight guard for the Save modal (carry-over T9 finding (a)): the
   * submit button (and the Enter shortcut) are inert while the PUT is
   * awaiting, so a double-Enter cannot fire two saves. */
  const [saving, setSaving] = useState(false);
  /* Task M2.5: the same useState + try/finally pattern for Validate and
   * Run. The store actions are async, so the Toolbar tracks the
   * in-flight window itself (runStore's runStatus only flips to
   * "running" once the POST /api/run round-trip completes). */
  const [validating, setValidating] = useState(false);
  const [runInFlight, setRunInFlight] = useState(false);
  /* One-shot flag so a Run click on an unsaved graph opens the Save modal
   * and continues with runGraph() after that save succeeds. */
  const runAfterSaveRef = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);
  /* Task M2.3 import: the hidden file picker the Import button clicks. */
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* Centralized Save-modal dismissal (T10 review CRITICAL-1): EVERY
   * non-submit dismiss path - backdrop click, Escape, Cancel - must clear
   * the one-shot runAfterSave flag along with closing the modal. A
   * backdrop-dismissed Run prompt that left the flag armed would make a
   * later ordinary Save silently launch an unintended GPU job. Opening
   * the modal WITHOUT the run intent (plain Save button) resets it too.
   * The flag lives in a ref, so this chokepoint is not reachable through
   * reducer-level store tests; covered by this single-path wiring plus
   * this comment (noted in the task report). */
  const dismissSave = () => {
    runAfterSaveRef.current = false;
    setModal(null);
  };

  /* Escape closes the active modal (capture phase so it wins over the
   * canvas handlers; closing twice is a no-op). */
  useEffect(() => {
    if (!modal) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        dismissSave();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [modal]);

  useEffect(() => {
    if (modal === "save") inputRef.current?.focus();
  }, [modal]);

  // Auto-save: persist graph to localStorage on every change (debounced)
  useEffect(() => {
    if (modal === "save" || modal === "open") return; // Don't save while modal open
    const timeout = setTimeout(() => {
      try {
        localStorage.setItem("flow:draft", JSON.stringify(graph));
      } catch (e) {
        // localStorage may be full or unavailable; fail silently
      }
    }, 2000);
    return () => clearTimeout(timeout);
  }, [graph, modal]);

  // Draft recovery: check for unsaved draft on mount
  useEffect(() => {
    try {
      const draft = localStorage.getItem("flow:draft");
      if (draft) {
        const parsed = JSON.parse(draft);
        if (parsed.nodes?.length > 0) {
          setError("Unsaved draft recovered — Save to keep it (or Clear draft in Save modal)");
        }
      }
    } catch (e) {
      // Corrupted draft; clear it silently
      localStorage.removeItem("flow:draft");
    }
  }, []);


  /* Task M2.3 export: serialize the CURRENT graph to a flow/0.1 document
   * and trigger a browser download. The serialization is the pure
   * serializeFlow (unit-tested); only the Blob/anchor dance is
   * browser-only. The document name echoes the saved slug, else
   * "untitled" (same convention as validateFlow). */
  const handleExport = () => {
    const name =
      currentFlowName !== null && isValidFlowName(currentFlowName) ? currentFlowName : "untitled";
    const text = serializeFlow(name, graph);
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name + ".flow.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setError("OK: exported " + name + ".flow.json");
  };

  /* Task M2.3 import: read the picked file, schema-validate it with the
   * pure parseFlowDocument, and load a valid doc via setGraph (which also
   * resets the projection and pushes undo history). Failures toast the
   * parser's error line; the input value is reset so re-picking the same
   * file re-fires onChange. */
  const handleImportFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    let text: string;
    try {
      text = await file.text();
    } catch (err) {
      setError("import failed: " + errorMessage(err));
      return;
    }
    const res = parseFlowDocument(text);
    if (res.error !== undefined || res.doc === undefined) {
      setError("import failed: " + (res.error ?? "invalid flow document"));
      return;
    }
    setGraph(res.doc.graph);
    setError("OK: imported " + file.name + " as '" + res.doc.meta.name + "'");
  };

  const submitSave = async () => {
    if (saving) return; // in-flight guard (T9 fix a)
    const name = inputRef.current?.value ?? "";
    if (!isValidFlowName(name)) {
      setError(
        "flow name must match [a-z0-9-]{1,64} (lowercase letters, digits, dashes)",
      );
      return;
    }
    setSaving(true);
    try {
      const ok = await saveFlow(name);
      if (ok) {
        setModal(null);
        setError(null);
        if (runAfterSaveRef.current) {
          runAfterSaveRef.current = false;
          // The save itself is done (modal closed): hand the in-flight
          // indicator to the Run button so it reads "starting…" while
          // the PUT-save + POST /api/run round-trip is awaiting.
          setSaving(false);
          setRunInFlight(true);
          try {
            await runGraph();
          } finally {
            setRunInFlight(false);
          }
        }
      }
    } finally {
      setSaving(false);
    }
  };

  /* Task M2.5: async handlers wrapping the store actions with the
   * in-flight guards above (double-click inert + Spinner on the
   * button). */
  const handleValidate = async () => {
    if (validating) return;
    setValidating(true);
    try {
      await validateFlow();
    } finally {
      setValidating(false);
    }
  };

  const handleRun = async () => {
    if (runInFlight) return;
    if (currentFlowName !== null && isValidFlowName(currentFlowName)) {
      setRunInFlight(true);
      try {
        await runGraph();
      } finally {
        setRunInFlight(false);
      }
    } else {
      runAfterSaveRef.current = true;
      setModal("save");
    }
  };

  return (
    <header className="toolbar">
      <span className="toolbar-title">flow</span>
      <button id="toolbar-open" aria-label="Open saved flow" title="Open a saved flow" onClick={() => setModal("open")}>
        Open
      </button>
      {/* Task M2.4: Delete removes the CURRENT saved flow server-side
          (DELETE /api/flows/{name}) after confirmation. Disabled while
          the graph is unsaved — there is no saved flow to delete. */}
      <button
        id="toolbar-delete"
        aria-label="Delete saved flow"
        title="Delete saved flow"
        disabled={currentFlowName === null}
        onClick={() => {
          const name = currentFlowName;
          if (name === null) return;
          openConfirm(
            "Delete flow",
            "Permanently delete the saved flow '" + name + "' from the server? This cannot be undone.",
            () => void deleteFlow(name),
          );
        }}
      >
        Delete
      </button>
      {/* Plain Save opens the modal WITHOUT the run intent; reset the
          one-shot flag here so a prior dismissed Run prompt cannot leak
          its armed state into an ordinary save (CRITICAL-1). */}
      <button
        id="toolbar-save"
        aria-label="Save current flow"
        title="Save flow to server"
        disabled={saving}
        onClick={() => {
          runAfterSaveRef.current = false;
          setModal("save");
        }}
      >
        {saving ? (
          <>
            <Spinner className="spinner-btn" />
            saving…
          </>
        ) : (
          "Save"
        )}
      </button>
      {/* Task M2.3: Export serializes the current graph to a flow/0.1
          * document and downloads it; Import opens a file picker and
          * loads a validated .flow.json via setGraph. Both toast
          * through the store.error channel (OK: prefix = success). */}
      <button id="toolbar-export" aria-label="Export flow as JSON" title="Export graph as JSON" onClick={handleExport}>
        Export
      </button>
      <button
        id="toolbar-import"
        aria-label="Import flow from JSON"
        title="Import graph from JSON"
        onClick={() => fileInputRef.current?.click()}
      >
        Import
      </button>
      {/* Task 10 wiring: Validate POSTs the UNSAVED graph document
        * (meta.name = saved slug or "untitled") and toasts ok / first-3
        * errors + "+N more"; Run PUT-saves the graph under its slug and
        * POSTs /api/run, switching to the Run log tab. An unsaved graph
        * routes Run through the Save modal first (one-shot flag). */}
      <button
        id="toolbar-validate"
        aria-label="Validate current graph"
        title="Validate current graph"
        disabled={validating}
        onClick={() => void handleValidate()}
      >
        {validating ? (
          <>
            <Spinner className="spinner-btn" />
            validating…
          </>
        ) : (
          "Validate"
        )}
      </button>
      {/* Running guard (T10 review MINOR-3): Run is inert while a job
          is live, matching Stop's running-only affordance. Task M2.5
          extends the guard to the runGraph in-flight window (Spinner +
          "starting…" while the PUT-save + POST /api/run is awaiting). */}
      <button
        id="toolbar-run"
        aria-label="Run current flow"
        title="Run the flow"
        disabled={runStatus?.state === "running" || runInFlight}
        onClick={() => void handleRun()}
      >
        {runInFlight ? (
          <>
            <Spinner className="spinner-btn" />
            starting…
          </>
        ) : (
          "Run"
        )}
      </button>
      {/* M1.1: Undo/Redo buttons with disabled state from history store */}
      <button
        id="toolbar-undo"
        aria-label="Undo last action"
        disabled={!canUndo}
        onClick={() => useFlowStore.getState().undo()}
        title="Undo (Ctrl+Z)"
      >
        Undo
      </button>
      <button
        id="toolbar-redo"
        aria-label="Redo last undone action"
        disabled={!canRedo}
        onClick={() => useFlowStore.getState().redo()}
        title="Redo (Ctrl+Shift+Z)"
      >
        Redo
      </button>
      {/* Task M2.4: Clear canvas is destructive — it opens the shared
          ConfirmDialog before setGraph wipes the canvas. */}
      <button
        id="toolbar-clear"
        aria-label="Clear canvas"
        title="Clear canvas"
        onClick={() =>
          openConfirm(
            "Clear canvas",
            "Remove all nodes and edges from the canvas? This cannot be undone.",
            () => {
              setGraph({ nodes: [], edges: [] });
              setError("OK: canvas cleared");
            },
          )
        }
      >
        Clear
      </button>
      {/* Task M2.2 node search: filters canvas nodes by id/kind/label;
          * Enter selects + centers on the next match (the center request
          * flows through the store because setCenter only exists inside
          * the ReactFlowProvider). */}
      <span className="toolbar-search">
        <svg
          className="toolbar-search-icon"
          width="13"
          height="13"
          viewBox="0 0 16 16"
          aria-hidden="true"
          focusable="false"
        >
          <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <line x1="11" y1="11" x2="14.5" y2="14.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <input
          id="toolbar-search"
          type="search"
          placeholder="find node…"
          aria-label="Find node"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              useFlowStore.getState().cycleSearchMatch();
            } else if (e.key === "Escape") {
              setSearchQuery("");
            }
          }}
        />
      </span>

      {/* Task M6.3: below 1200px the palette and inspector are overlays;
          these toggles (CSS-hidden at >=1200px) open/close them through
          the same store collapse flags the in-panel «/» buttons use. */}
      <button
        id="toolbar-palette-toggle"
        className="toolbar-panel-toggle"
        aria-label="Toggle palette panel"
        aria-pressed={!paletteCollapsed}
        title="Show/hide the node palette"
        onClick={() => setPaletteCollapsed(!paletteCollapsed)}
      >
        Palette
      </button>
      <button
        id="toolbar-inspector-toggle"
        className="toolbar-panel-toggle"
        aria-label="Toggle inspector panel"
        aria-pressed={!inspectorCollapsed}
        title="Show/hide the inspector"
        onClick={() => setInspectorCollapsed(!inspectorCollapsed)}
      >
        Inspector
      </button>

      {/* Task M2.3: hidden picker for the Import button (always rendered
          * so the ref is stable; accept restricts to flow documents). */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".flow.json,application/json"
        aria-hidden="true"
        tabIndex={-1}
        style={{ display: "none" }}
        onChange={(e) => void handleImportFile(e)}
      />

      {modal === "open" && (
        <FlowPicker
          /* A pick must ALSO dismiss the modal: openFlow alone left the Open
           * dialog covering the canvas after loading (browser drill finding). */
          onPick={(name) => { setModal(null); void openFlow(name); }}
          onDismiss={() => setModal(null)}
        />
      )}

      {modal === "save" && (
        <div className="modal-backdrop" onClick={dismissSave}>
          <div
            className="modal"
            role="dialog"
            aria-label="Save flow"
            onClick={(e) => e.stopPropagation()}
          >
            <label htmlFor="save-name">flow name</label>
            <input
              ref={inputRef}
              id="save-name"
              placeholder="my-run"
              onKeyDown={(e) => {
                if (e.key === "Enter") void submitSave();
              }}
            />
            <div className="modal-actions">
              <button onClick={dismissSave}>Cancel</button>
              <button
                onClick={() => {
                  localStorage.removeItem("flow:draft");
                  setError(null);
                }}
              >
                Clear draft
              </button>
              <button disabled={saving} onClick={() => void submitSave()}>
                {modal === "save" && saving ? (
                  <>
                    <Spinner className="spinner-btn" />
                    saving…
                  </>
                ) : (
                  "Save"
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Task M2.4: shared destructive-action confirmation. Confirm runs
          the requested effect (dismissing first so a synchronous throw
          cannot leave the dialog up); Cancel / Escape / backdrop click
          all route through onDismiss. */}
      <ConfirmDialog
        open={confirm !== null}
        title={confirm?.title ?? ""}
        message={confirm?.message ?? ""}
        onConfirm={() => {
          const action = confirm?.action;
          dismissConfirm();
          action?.();
        }}
        onDismiss={dismissConfirm}
      />
    </header>
  );
}

/** Body for the Open modal: names come straight from GET /api/flows;
 * the list fetch failure is shown inline and API failures from the
 * subsequent GET/PUT land in the store.error toast. */
function FlowPicker({
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
      .getFlows()
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
      <div
        className="modal"
        role="dialog"
        aria-label="Open flow"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="modal-title">saved flows</p>
        {listError !== null && <p className="honesty-label">{listError}</p>}
        {names === null && <p className="modal-note">loading…</p>}
        {names !== null && names.length === 0 && (
          <p className="modal-note">no saved flows yet</p>
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