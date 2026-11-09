import { useEffect, useRef, useState } from "react";
import {
  useFlowStore,
  FALLBACK_REGISTRY,
  propWidgetType,
  propSelectOptions,
  updateNodePropsReducer,
  nextPollDelay,
  type InspectorTab,
  type NodeSpec,
  type PropWidget,
} from "./store";
import { api, errorMessage, mapRunStatus } from "./api";
import { Spinner } from "../../packages/ui-components/src/index";
// T3 registry promotion: the label placeholder semantic comes from the
// registry (label_semantic), not from a kind-string literal.
import { DATASET_LABEL_SEMANTIC } from "./nodes/registry";

const TABS: { id: InspectorTab; label: string }[] = [
  { id: "properties", label: "Properties" },
  { id: "run-log", label: "Run Log" },
  { id: "preview", label: "Preview" },
];

/** One labeled editor row for a registry prop. Widget selection comes
 * from propWidgetType (registry-driven); number inputs coerce on change. */
function PropRow({
  name,
  value,
  widget,
  onChange,
}: {
  name: string;
  value: unknown;
  widget: PropWidget;
  onChange: (v: unknown) => void;
}) {
  const options = widget === "select" ? propSelectOptions(name) : null;
  const inputId = "prop-" + name;
  return (
    <div className="prop-row">
      <label htmlFor={inputId}>{name}</label>
      {widget === "select" && options !== null ? (
        <select
          id={inputId}
          value={typeof value === "string" ? value : ""}
          onChange={(e) => onChange(e.target.value)}
        >
          {!options.includes(String(value)) && <option value="">(unset)</option>}
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : widget === "number" ? (
        <input
          id={inputId}
          type="number"
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

/** Advanced disclosure: raw props JSON editor. Honesty label is explicit
 * (DESIGN advanced YAML/Knobs section): what you type is the node's props
 * object verbatim — no validation happens here; the server re-checks
 * everything on POST /api/validate and on PUT /api/flows/{name}. */
function AdvancedPropsEditor({ nodeId, propsJson }: { nodeId: string; propsJson: string }) {
  const updateNodeProps = useFlowStore((s) => s.updateNodeProps);
  const setError = useFlowStore((s) => s.setError);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(propsJson);
  // Reset the draft when the selected node (or its props) change.
  useEffect(() => setDraft(propsJson), [nodeId, propsJson]);

  return (
    <details className="advanced-props" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Advanced: raw props JSON</summary>
      {open && (
        <>
          <p className="honesty-label">
            honest: this is YAML-adjacent — the JSON below becomes the node's
            props verbatim; nothing is validated here, the server does that
            on save/validate.
          </p>
          <textarea
            aria-label={"raw props JSON for node " + nodeId}
            value={draft}
            rows={6}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button
            onClick={() => {
              try {
                const parsed: unknown = JSON.parse(draft);
                if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
                  setError("advanced props: JSON must be an object (props are key → value)");
                  return;
                }
                updateNodeProps(nodeId, parsed as Record<string, unknown>);
                setError(null);
              } catch {
                setError("advanced props: not valid JSON");
              }
            }}
          >
            Apply to node
          </button>
        </>
      )}
    </details>
  );
}

/** Properties tab for store.selection: widgets per the registry snapshot
 * (never the kind guess), "(no editable properties)" for the empty-props
 * kinds (dataset / eval / infer per the T7 contract table). */
function PropertiesPanel() {
  const selection = useFlowStore((s) => s.selection);
  const graph = useFlowStore((s) => s.graph);
  const registry = useFlowStore((s) => s.registry);
  const setError = useFlowStore((s) => s.setError);
  const updateNodeProps = useFlowStore((s) => s.updateNodeProps);
  const updateNodeLabel = useFlowStore((s) => s.updateNodeLabel);

  const node = selection === null ? null : graph.nodes.find((n) => n.id === selection);
  if (!node) return <p className="inspector-empty">select a node on the canvas</p>;
  const spec: NodeSpec | undefined =
    (registry ?? FALLBACK_REGISTRY).nodes[node.kind];

  /* Label editor (browser drill finding): the MVP config mapping reads
   * the dataset node's label as the HF dataset name, so the label is
   * editable for EVERY kind regardless of registry props. */
  const labelRow = (
    <div className="prop-row">
      <label htmlFor="node-label">label</label>
      <input
        id="node-label"
        type="text"
        value={node.label ?? ""}
        placeholder={spec?.label_semantic === DATASET_LABEL_SEMANTIC ? "HF dataset name" : "display name"}
        onChange={(e) => {
          setError(null);
          updateNodeLabel(node.id, e.target.value);
        }}
      />
    </div>
  );

  if (!spec || spec.props.length === 0) {
    return (
      <>
        <p className="inspector-empty">(no editable properties)</p>
        {labelRow}
        {spec && spec.props.length === 0 && (
          /** Advanced JSON editing still makes sense for raw round-trip
           * fidelity (e.g. an old document's extra keys) even when the
           * registry lists no widgets for the kind. */
          <AdvancedPropsEditor nodeId={node.id} propsJson={JSON.stringify(node.props, null, 2)} />
        )}
      </>
    );
  }

  const setProp = (name: string, value: unknown) => {
    const next = { ...node.props };
    if (value === undefined) delete next[name];
    else next[name] = value;
    // Server validates on save/validate; the client write path keeps the
    // reducer pure (updateNodePropsReducer) and optimism is honest here.
    updateNodeProps(node.id, next);
  };

  return (
    <>
      <p className="prop-kind">kind: {node.kind}</p>
      {labelRow}
      {spec.props.map((p) => (
        <PropRow
          key={p}
          name={p}
          value={node.props[p]}
          widget={propWidgetType(spec, p)}
          onChange={(v) => {
            setError(null);
            setProp(p, v);
          }}
        />
      ))}
      <AdvancedPropsEditor nodeId={node.id} propsJson={JSON.stringify(node.props, null, 2)} />
    </>
  );
}

/** Run log tab: while a run is watched (store.runStatus.state running) it
 * polls GET /api/run/status (nextPollDelay: 2 s base cadence, backoff to
 * 30 s after 3 consecutive failures, restored on an OK read) in a
 * setTimeout chain, streams the tail into a scrollable pre with bottom
 * auto-adherence (sticks unless the user scrolled up) and clears the tail
 * when a fresh run re-arms the watcher (carry-over T9 fix (c)) so old
 * output cannot bleed into the new run view. A Stop button (the only run
 * control; there is NO kill button by design) writes the checkpoint-
 * aligned STOP flag while running. */
function RunLogPanel() {
  const runStatus = useFlowStore((s) => s.runStatus);
  const setRunStatus = useFlowStore((s) => s.setRunStatus);
  const stopRun = useFlowStore((s) => s.stopRun);
  const stopInfo = useFlowStore((s) => s.stopInfo);
  const [tail, setTail] = useState<string[]>([]);
  const [pollError, setPollError] = useState<string | null>(null);
  const preRef = useRef<HTMLPreElement | null>(null);
  // Stick-to-bottom heuristic: true until the user scrolls up away from
  // the bottom edge; every successful tail read re-evaluates adherence.
  const atBottomRef = useRef(true);

  /* Poll only while a run is being watched AND still live; a terminal
   * mapped state (done/error) settles the panel and stops the interval
   * (the toolbar Run button re-arms via setRunStatus in Task 10). */
  const watching = runStatus !== null && runStatus.state === "running";

  useEffect(() => {
    if (!watching) return;
    let cancelled = false;
    let timer = 0;
    let failures = 0; // consecutive failures feeding nextPollDelay
    // setTimeout CHAIN (not setInterval): differing success/failure and
    // backoff cadences cannot overlap long requests. A failed read
    // increments the failure count (delay grows to 30 s at 3), a good
    // read resets it to the base 2 s cadence (T9 fix d).
    const poll = async () => {
      try {
        const s = await api.runStatus();
        if (cancelled) return;
        failures = 0;
        setPollError(null);
        const next = mapRunStatus(s);
        setRunStatus(next);
        if (s.tail.length > 0) setTail(s.tail);
        if (atBottomRef.current && preRef.current) {
          preRef.current.scrollTop = preRef.current.scrollHeight;
        }
        // Once the mapped state is terminal the watcher stops (a fresh
        // run re-arms polling via runGraph -> setRunStatus).
        if (next.state === "running") timer = window.setTimeout(poll, nextPollDelay(0));
      } catch (e) {
        if (cancelled) return;
        failures += 1;
        setPollError(errorMessage(e));
        timer = window.setTimeout(poll, nextPollDelay(failures));
      }
    };
    timer = window.setTimeout(poll, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [watching, setRunStatus]);

  // Carry-over T9 fix (c): when the watcher re-arms (idle/null ->
  // running persisted across polls does not count), resets the tail so a
  // brand-new run starts with a clean view. Cleanest: track the previous
  // runStatus object identity — same-object updates don't clear.
  const prevWatchedRef = useRef<unknown>(null);
  useEffect(() => {
    const cur = runStatus;
    if (
      cur !== prevWatchedRef.current &&
      cur !== null &&
      cur.state === "running" &&
      prevWatchedRef.current === null
    ) {
      setTail([]);
    }
    prevWatchedRef.current = cur === null || cur.state !== "running" ? null : cur;
  }, [runStatus]);

  useEffect(() => {
    if (atBottomRef.current && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [tail]);

  if (runStatus === null || runStatus.state === "idle") {
    return <p className="inspector-empty">no live run</p>;
  }

  return (
    <div className="run-log">
      <p className={runStatus.state === "error" ? "run-state run-state-error" : "run-state"} role="status">
        {runStatus.state}
        {runStatus.message ? " — " + runStatus.message : ""}
      </p>
      {stopInfo !== null && <p className="honesty-label">{stopInfo}</p>}
      {runStatus.state === "running" && (
        /* Task 10 Stop: checkpoint-aligned stop ONLY — it writes the U11
         * STOP flag; the trainer exits at the next checkpoint save. There
         * is deliberately NO kill button (per design). 409 detail (e.g.
         * "no live GPU job to stop") lands in the store.error toast. */
        <button
          className="run-stop"
          onClick={() => void stopRun()}
          title="write STOP flag; trainer exits at next checkpoint save"
        >
          Stop
        </button>
      )}
      {pollError !== null && <p className="honesty-label">{pollError}</p>}
      {/* Task M2.5: the watcher polls on a setTimeout chain; while it is
          waiting and no tail line has arrived yet, show the Spinner so a
          live run reads as active rather than stuck. It disappears on the
          first tail read and re-arms when a fresh run resets the tail. */}
      {watching && tail.length === 0 && <Spinner className="run-log-spinner" />}
      <pre
        ref={preRef}
        className="run-tail"
        aria-label="live run output tail"
        onScroll={(e) => {
          const el = e.currentTarget;
          atBottomRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
        }}
      >
        {tail.length === 0 ? "(no output yet)" : tail.join("\n")}
      </pre>
    </div>
  );
}

/** Preview tab (Task 10): a read-only summary card built from the graph
 * the server ACCEPTED in the last successful POST /api/validate
 * (store.validatedDoc). Label is explicit that this is the last
 * Validation snapshot, not a fresh compile: NO server dry-run runs here —
 * we simply show the document the server just accepted (naive but honest
 * MVP); a fresh Validate refreshes it, any graph edit keeps it stale. */
function PreviewPanel() {
  const doc = useFlowStore((s) => s.validatedDoc);
  if (doc === null) {
    return <p className="inspector-empty">no successful validation yet — press Validate first</p>;
  }
  return (
    <div className="preview-summary" aria-label="compiled config summary">
      <p className="preview-label">
        compiled config summary (from last Validation)
      </p>
      <p className="honesty-label">
        honest: this is the exact graph the server just accepted in the last
        reflection-free validation — not a separate server dry-run. Edits
        after Validating keep this stale until the next Validate.
      </p>
      <p className="preview-meta">
        meta.name: <code>{doc.meta.name}</code> · schema: {doc.schema}
      </p>
      <p className="preview-section">nodes ({doc.graph.nodes.length})</p>
      <ul className="preview-list">
        {doc.graph.nodes.map((n) => (
          <li key={n.id} className="preview-one-liner">
            <code>{n.id}</code> {n.kind}
            {Object.entries(n.props).length > 0 && (
              <span>
                {" · "}
                {Object.entries(n.props)
                  .map(([k, v]) => k + "=" + String(v))
                  .join(", ")}
              </span>
            )}
          </li>
        ))}
        {doc.graph.nodes.length === 0 && <li className="inspector-empty">(empty graph)</li>}
      </ul>
      <p className="preview-section">edges ({doc.graph.edges.length})</p>
      <ul className="preview-list">
        {doc.graph.edges.map((e) => (
          <li key={e.id} className="preview-one-liner">
            <code>{e.id}</code> {e.from}.{e.fromPort} → {e.to}.{e.toPort}
          </li>
        ))}
        {doc.graph.edges.length === 0 && <li className="inspector-empty">(no edges)</li>}
      </ul>
    </div>
  );
}

export default function Inspector() {
  const tab = useFlowStore((s) => s.inspectorTab);
  const setTab = useFlowStore((s) => s.setInspectorTab);
  const collapsed = useFlowStore((s) => s.inspectorCollapsed);
  const setCollapsed = useFlowStore((s) => s.setInspectorCollapsed);

  if (collapsed) {
    return (
      <aside className="inspector inspector-collapsed">
        <button onClick={() => setCollapsed(false)} aria-label="Expand inspector">
          «
        </button>
      </aside>
    );
  }

  return (
    <aside className="inspector">
      <button onClick={() => setCollapsed(true)} aria-label="Collapse inspector">
        »
      </button>
      <div className="inspector-tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={"inspector-tab-" + t.id}
            aria-selected={tab === t.id}
            aria-controls={"inspector-panel-" + t.id}
            onClick={() => setTab(t.id)}
          >
              {t.label}
          </button>
        ))}
      </div>
      {/* One panel element per known tab (hidden ones inert) so each tab
       * button's aria-controls target actually exists. */}
      {TABS.map((t) => (
        <div
          key={t.id}
          id={"inspector-panel-" + t.id}
          role="tabpanel"
          className="inspector-panel"
          hidden={tab !== t.id}
          aria-labelledby={"inspector-tab-" + t.id}
        >
          {tab !== t.id ? null : t.id === "properties" ? (
            <PropertiesPanel />
          ) : t.id === "run-log" ? (
            <RunLogPanel />
          ) : (
            <PreviewPanel />
          )}
        </div>
      ))}
    </aside>
  );
}