import { useCallback, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import {
  ReactFlow,
  Background,
  ReactFlowProvider,
  useReactFlow,
  Controls,
  MiniMap,
} from "@xyflow/react";
import type { Connection, EdgeChange, NodeChange } from "@xyflow/react";
import { useFlowStore, toXYNodes, toXYEdges, FALLBACK_REGISTRY } from "./store";
import PipelineNode from "./nodes/PipelineNode";
import { usePaletteKinds } from "./Palette";
import "@xyflow/react/dist/style.css";

const nodeTypes = { pipeline: PipelineNode };

/* Approximate viewport footprint used to clamp the context-menu open
 * position (the add-menu is ~170px min-width plus the search row, up to
 * seven rows tall; 200x260 covers it with headroom). */
const MENU_W = 200;
const MENU_H = 260;

/** Add-node menu opened by right-click anywhere on the canvas (MVP
 * required UX) with one entry per registry kind plus "search… (Ctrl+K)".
 * Adds happen at the mouse position (flow coordinates). */
function AddMenu({
  kinds,
  onPick,
  onSearch,
}: {
  kinds: string[];
  onPick: (kind: string) => void;
  onSearch: () => void;
}) {
  return (
    <div className="add-menu" role="menu">
      {kinds.map((kind) => (
        <button key={kind} role="menuitem" onClick={() => onPick(kind)}>
          {kind}
        </button>
      ))}
      <button role="menuitem" onClick={onSearch} className="add-menu-search">
        search… (Ctrl+K)
      </button>
    </div>
  );
}

/** Ctrl+K search modal: filter box, arrow-key navigation, Enter adds the
 * selected kind at the mouse position; Escape closes, selection no-op. */
function SearchModal({
  kinds,
  onPick,
  open,
}: {
  kinds: string[];
  onPick: (kind: string) => void;
  open: boolean;
}) {
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      inputRef.current?.focus();
    }
  }, [open]);

  if (!open) return null;
  const filtered = kinds.filter((k) => k.includes(query.toLowerCase()));
  return (
    <div className="search-backdrop" onClick={() => onPick("__close__")}>
      <div
        className="search-modal"
        role="dialog"
        aria-label="Add node search"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          placeholder="search kinds…"
          onChange={(e) => {
            setQuery(e.target.value);
            setIndex(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") onPick("__close__");
            else if (e.key === "ArrowDown") setIndex((i) => Math.min(i + 1, filtered.length - 1));
            else if (e.key === "ArrowUp") setIndex((i) => Math.max(i - 1, 0));
            else if (e.key === "Enter") {
              if (filtered.length > 0 && filtered[index]) onPick(filtered[index]);
            }
          }}
        />
        <ul className="search-list">
          {filtered.map((k, i) => (
            <li key={k}>
              <button
                className={i === index ? "active" : ""}
                onMouseEnter={() => setIndex(i)}
                onClick={() => onPick(k)}
              >
                {k}
              </button>
            </li>
          ))}
          {filtered.length === 0 && <li className="search-empty">no match</li>}
        </ul>
      </div>
    </div>
  );
}

/* Task M5.5 (a11y): arrow-key navigation between nodes. The four
 * arrows map to a unit direction in domain coordinates — React Flow's
 * viewport transform is a uniform scale + translate, so a node that is
 * "to the right" on screen is also to the right in domain space. */
const ARROW_KEY_DIRECTIONS: Record<string, { dx: number; dy: number }> = {
  ArrowUp: { dx: 0, dy: -1 },
  ArrowDown: { dx: 0, dy: 1 },
  ArrowLeft: { dx: -1, dy: 0 },
  ArrowRight: { dx: 1, dy: 0 },
};

/* Move the visual node selection through the same select-change path a
 * node click uses: applyNodesChanges mirrors select changes into
 * store.selection AND the projection's selected flags, so the canvas
 * highlight and the Inspector panel follow together. */
const selectNode = (id: string) => {
  const st = useFlowStore.getState();
  const changes: NodeChange[] = [];
  if (st.selection !== null && st.selection !== id) {
    changes.push({ type: "select", id: st.selection, selected: false });
  }
  changes.push({ type: "select", id, selected: true });
  st.applyNodesChanges(changes);
};

/* Pick the nearest node in the given direction from the current
 * selection: candidates must be clearly in that half-plane (deadzone),
 * scored by distance along the direction minus twice the perpendicular
 * offset, so a node straight ahead beats a nearer diagonal one. With no
 * selection yet, start from the top-left-most node. */
const moveSelectionInDirection = (dir: { dx: number; dy: number }) => {
  const { graph, selection } = useFlowStore.getState();
  const nodes = graph.nodes;
  if (nodes.length === 0) return;
  const current = nodes.find((n) => n.id === selection) ?? null;
  if (current === null) {
    const first = nodes.reduce((a, b) =>
      b.position.y < a.position.y - 1 ||
      (Math.abs(b.position.y - a.position.y) <= 1 && b.position.x < a.position.x)
        ? b
        : a,
    );
    selectNode(first.id);
    return;
  }
  let bestId: string | null = null;
  let bestScore = -Infinity;
  for (const n of nodes) {
    if (n.id === current.id) continue;
    const dx = n.position.x - current.position.x;
    const dy = n.position.y - current.position.y;
    const along = dx * dir.dx + dy * dir.dy;
    if (along <= 8) continue;
    const perp = Math.abs(dx * dir.dy - dy * dir.dx);
    const score = along - perp * 2;
    if (score > bestScore) {
      bestScore = score;
      bestId = n.id;
    }
  }
  if (bestId !== null) selectNode(bestId);
};

/** Canvas: controlled ReactFlow wired into the store (apply node AND edge
 * changes into store.graph — edge deletion included), registry-validated
 * onConnect (refuse + toast), HTML5 drag-from-palette, right-click add
 * menu and Ctrl+K. Both menu paths add at the mouse position. */
function FlowCanvasInner() {
  const graph = useFlowStore((s) => s.graph);
  const error = useFlowStore((s) => s.error);
  const setError = useFlowStore((s) => s.setError);
  const addNode = useFlowStore((s) => s.addNode);
  const connect = useFlowStore((s) => s.connect);
  const applyNodesChanges = useFlowStore((s) => s.applyNodesChanges);
  const applyEdgesChanges = useFlowStore((s) => s.applyEdgesChanges);
  const registry = useFlowStore((s) => s.registry);
  const projection = useFlowStore((s) => s.projection);
  // Task 10 dot wiring: the whole-graph run state (and terminal exit
  // code) threads from store.runStatus into every node's data via
  // toXYNodes — see its doc comment for the honest MVP simplification. */
  const runStatus = useFlowStore((s) => s.runStatus);
  // Task M2.2 node search: the query drives the node-highlight/node-dim
  // classNames via toXYNodes; centerRequest is answered with setCenter
  // (only available inside the provider - hence the store round-trip).
  const searchQuery = useFlowStore((s) => s.searchQuery);
  const centerRequest = useFlowStore((s) => s.centerRequest);

  const { screenToFlowPosition, setCenter } = useReactFlow();
  const { kinds } = usePaletteKinds();

  /* Canvas ref replaces the old document.querySelector(".canvas") global
   * lookup (T8 review MINOR-7); refs for the menu/modal outside-close. */
  const canvasRef = useRef<HTMLElement | null>(null);
  const menuWrapRef = useRef<HTMLDivElement | null>(null);
  const searchWrapRef = useRef<HTMLDivElement | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  /* Toast clears itself a moment after appearing (canvas owns the timer). */
  useEffect(() => {
    if (!error) return;
    const t = window.setTimeout(() => setError(null), 4000);
    return () => window.clearTimeout(t);
  }, [error, setError]);

  /* Task M2.2: answer store-issued center requests (search Enter cycling)
   * with setCenter. The nonce makes each request idempotent under effect
   * re-runs (StrictMode double-invokes effects in dev); the position was
   * snapshotted from the domain graph when the request was issued. */
  const lastCenterNonceRef = useRef(0);
  useEffect(() => {
    if (!centerRequest || centerRequest.nonce === lastCenterNonceRef.current) return;
    lastCenterNonceRef.current = centerRequest.nonce;
    setCenter(centerRequest.x, centerRequest.y, { duration: 300 });
  }, [centerRequest, setCenter]);

  const setLastMouse = useCallback((clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    lastMouseRef.current = { x: clientX - (rect?.left ?? 0), y: clientY - (rect?.top ?? 0) };
  }, []);

  /* Ctrl+K toggles the search modal anywhere; Escape closes the menu and
   * modal at document level (also when focus is outside the input). The
   * modal's own Escape handler runs alongside — closing twice is a no-op.
   * Ctrl+Z = undo, Ctrl+Shift+Z / Ctrl+Y = redo (M1.1). */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Undo/redo: ignore when typing in an input/textarea/select
      const tag = (document.activeElement?.tagName ?? "").toLowerCase();
      const isTyping = tag === "input" || tag === "textarea" || tag === "select";
      if ((e.ctrlKey || e.metaKey) && !isTyping) {
        if (e.key === "z" || e.key === "Z") {
          e.preventDefault();
          if (e.shiftKey) {
            useFlowStore.getState().redo();
          } else {
            useFlowStore.getState().undo();
          }
          return;
        }
        if (e.key === "y" || e.key === "Y") {
          e.preventDefault();
          useFlowStore.getState().redo();
          return;
        }
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setSearchOpen((open) => !open);
        setMenu(null);
      } else if (e.key === "Escape") {
        setMenu(null);
        setSearchOpen(false);
      } else if (e.key === "Delete" || e.key === "Backspace") {
        // Don't delete when typing in an input/textarea/select
        const tag = (document.activeElement?.tagName ?? "").toLowerCase();
        const isTyping = tag === "input" || tag === "textarea" || tag === "select";
        if (isTyping) return;
        // Delete selected nodes/edges
        const { nodes, edges } = useFlowStore.getState().graph;
        const selectedNodes = nodes.filter((n) => (n as any).selected);
        const selectedEdges = edges.filter((e) => (e as any).selected);
        if (selectedNodes.length > 0 || selectedEdges.length > 0) {
          e.preventDefault();
          // Push history before deleting
          useFlowStore.getState().push(useFlowStore.getState().graph, "delete");
          // Delete selected nodes and edges
          selectedNodes.forEach((n) => {
            useFlowStore.getState().applyNodesChanges([{ type: "remove", id: n.id }]);
          });
          selectedEdges.forEach((e) => {
            useFlowStore.getState().applyEdgesChanges([{ type: "remove", id: e.id }]);
          });
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /* Close the menu/modal on any pointer-down outside of them. The palette
   * and inspector areas live outside the canvas, so this covers them; the
   * pane/node click handlers keep closing on canvas clicks as before. */
  useEffect(() => {
    if (!menu && !searchOpen) return;
    const onDocPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (menuWrapRef.current?.contains(target) || searchWrapRef.current?.contains(target)) return;
      setMenu(null);
      setSearchOpen(false);
    };
    document.addEventListener("mousedown", onDocPointerDown);
    return () => document.removeEventListener("mousedown", onDocPointerDown);
  }, [menu, searchOpen]);

  /* Task M5.5 (a11y): canvas keyboard navigation, on the canvas container
   * in the CAPTURE phase so the handle case can pre-empt React Flow's
   * built-in node keyboard handler (which moves a focused, selected
   * node) without disturbing that existing behavior for plain node
   * focus. Arrow keys move the node selection; Tab cycles focus through
   * the node connection handles. */
  const onCanvasKeyDownCapture = (e: ReactKeyboardEvent) => {
    // Never hijack keys while typing in a form field.
    const tag = (document.activeElement?.tagName ?? "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    // The add-menu / search modal own their keyboard interactions.
    if (menu !== null || searchOpen) return;

    const dir = ARROW_KEY_DIRECTIONS[e.key];
    if (dir !== undefined) {
      const target = e.target as HTMLElement | null;
      const onHandle = target?.closest?.(".react-flow__handle") != null;
      // Focus on a node (but not one of its handles): defer to React
      // Flow's own handler, which moves the focused node.
      if (target?.closest?.(".react-flow__node") != null && !onHandle) return;
      e.preventDefault();
      if (onHandle) e.stopPropagation(); // keep the node from also moving
      moveSelectionInDirection(dir);
      return;
    }

    if (e.key === "Tab") {
      const canvas = canvasRef.current;
      if (canvas == null) return;
      const handles = Array.from(canvas.querySelectorAll<HTMLElement>(".react-flow__handle"));
      if (handles.length === 0) return;
      const active = document.activeElement as HTMLElement | null;
      const idx = active !== null ? handles.indexOf(active) : -1;
      if (idx === -1) return; // focus is not on a handle — default tab order
      const nextIdx = e.shiftKey ? idx - 1 : idx + 1;
      // At the ends of the handle set, let Tab leave the canvas normally
      // instead of trapping focus in the handles.
      if (nextIdx < 0 || nextIdx >= handles.length) return;
      e.preventDefault();
      handles[nextIdx].focus();
    }
  };

  /* Context-menu open position, clamped inside the window so it cannot
   * clip at the right/bottom canvas/viewport edges (T8 review MINOR-5). */
  const openMenuAt = useCallback(
    (clientX: number, clientY: number) => {
      setLastMouse(clientX, clientY);
      setMenu({
        x: Math.max(0, Math.min(clientX, window.innerWidth - MENU_W)),
        y: Math.max(0, Math.min(clientY, window.innerHeight - MENU_H)),
      });
    },
    [setLastMouse],
  );

  const addAtMouse = useCallback(
    (kind: string) => {
      setMenu(null);
      setSearchOpen(false);
      if (kind === "" || kind === "__close__") return;
      const m = lastMouseRef.current;
      const rect = canvasRef.current?.getBoundingClientRect();
      const pos = screenToFlowPosition({
        x: (rect?.left ?? 0) + m.x,
        y: (rect?.top ?? 0) + m.y,
      });
      addNode(kind, pos);
    },
    [addNode, screenToFlowPosition],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => applyNodesChanges(changes),
    [applyNodesChanges],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      // Edge deletions must round-trip into store.graph (T8 review
      // IMPORTANT-2); non-removal changes are projection-only there.
      applyEdgesChanges(changes);
    },
    [applyEdgesChanges],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      // Registry-validated in the reducer; on reject nothing in the graph
      // changes and store.error drives the toast below.
      connect(c);
    },
    [connect],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const kind = e.dataTransfer.getData("application/flow-node-kind");
      if (!kind) return;
      addNode(kind, screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    },
    [addNode, screenToFlowPosition],
  );

  return (
    <main
      ref={canvasRef}
      className="canvas"
      aria-label="Flow canvas"
      tabIndex={0}
      onKeyDownCapture={onCanvasKeyDownCapture}
    >
      <ReactFlow
        nodes={toXYNodes(
          graph,
          registry ?? FALLBACK_REGISTRY,
          projection,
          runStatus?.state ?? "idle",
          (runStatus?.state === "done" || runStatus?.state === "error")
            ? (runStatus.exitCode ?? null)
            : null,
          searchQuery,
        )}
        edges={toXYEdges(graph)}
        nodeTypes={nodeTypes}
        /* Task M6.1: React Flow v12's built-in node virtualization. Once
         * the ResizeObserver has measured the nodes, only nodes overlapping
         * the viewport (partially-visible edge nodes and in-flight drags
         * included) are rendered — off-screen nodes of large graphs no
         * longer re-render on every pan/zoom/selection change, which is
         * where the frame drops came from. */
        onlyRenderVisibleElements
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onPaneContextMenu={(e) => { e.preventDefault(); openMenuAt(e.clientX, e.clientY); }}
        onNodeContextMenu={(e) => { e.preventDefault(); openMenuAt(e.clientX, e.clientY); }}
        onPaneClick={() => { setMenu(null); setSearchOpen(false); }}
        onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }}
        onDrop={onDrop}
        onNodeClick={() => setMenu(null)}
      >
        <Background />
        <Controls position="bottom-left" />
        <MiniMap position="bottom-right" className="minimap" />
      </ReactFlow>

      {menu !== null && (
        <div className="add-menu-wrap" style={{ left: menu.x, top: menu.y }} ref={menuWrapRef}>
          <AddMenu kinds={kinds} onPick={addAtMouse} onSearch={() => { setMenu(null); setSearchOpen(true); }} />
        </div>
      )}

      {searchOpen && (
        /* Rendered directly under the canvas (no positioning wrapper):
         * .search-modal is position:fixed (App.css), so absolutely
         * offsetting a wrapper at lastMouseRef was dead positioning — the
         * centered dialog placement lives entirely in the modal CSS. */
        <div ref={searchWrapRef}>
          <SearchModal open kinds={kinds} onPick={addAtMouse} />
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

export default function FlowCanvas() {
  return (
    <ReactFlowProvider>
      <FlowCanvasInner />
    </ReactFlowProvider>
  );
}