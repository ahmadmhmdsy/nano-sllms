import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { useFlowStore } from "./store";
import { ConfirmDialog } from "../../packages/ui-components/src/index";

/* ------------------------------------------------------------------ */
/* Task M2.4: confirmation dialog for destructive actions. The dialog */
/* request lives in the store (openConfirm/dismissConfirm + the       */
/* pending confirm request), so the wiring is tested through the       */
/* composed store exactly like the infer-handoff flag routing — the    */
/* repo has no @testing-library, and the brief allows testing the     */
/* wiring via the store actions. The presentational contract (null     */
/* when closed, modal markup when open) is verified through           */
/* renderToStaticMarkup.                                               */
/* ------------------------------------------------------------------ */

describe("Task M2.4 ConfirmDialog presentational rendering", () => {
  it("renders nothing when closed", () => {
    const html = renderToStaticMarkup(
      <ConfirmDialog
        open={false}
        title="Clear canvas"
        message="Remove all nodes and edges from the canvas?"
        onConfirm={() => {}}
        onDismiss={() => {}}
      />,
    );
    expect(html).toBe("");
  });

  it("renders the modal with title, message, Cancel and Confirm when open", () => {
    const html = renderToStaticMarkup(
      <ConfirmDialog
        open={true}
        title="Clear canvas"
        message="Remove all nodes and edges from the canvas?"
        onConfirm={() => {}}
        onDismiss={() => {}}
      />,
    );
    expect(html).toContain("modal-backdrop");
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-label="Clear canvas"');
    expect(html).toContain("Clear canvas");
    expect(html).toContain("Remove all nodes and edges from the canvas?");
    expect(html).toContain("Cancel");
    expect(html).toContain("Confirm");
  });
});

describe("Task M2.4 confirm dialog wiring (store slice)", () => {
  beforeEach(() => {
    useFlowStore.setState({
      graph: {
        nodes: [{ id: "n1", kind: "dataset", props: {}, position: { x: 0, y: 0 } }],
        edges: [],
      },
      confirm: null,
      error: null,
    });
  });

  it("dialog shows on Clear click (openConfirm stores title/message/action)", () => {
    const action = () => useFlowStore.getState().setGraph({ nodes: [], edges: [] });
    useFlowStore.getState().openConfirm(
      "Clear canvas",
      "Remove all nodes and edges from the canvas? This cannot be undone.",
      action,
    );
    const s = useFlowStore.getState();
    expect(s.confirm).not.toBeNull();
    expect(s.confirm?.title).toBe("Clear canvas");
    expect(s.confirm?.message).toBe(
      "Remove all nodes and edges from the canvas? This cannot be undone.",
    );
    expect(s.confirm?.action).toBe(action);
  });

  it("dialog confirm clears canvas (the wired action runs, then dismiss)", () => {
    useFlowStore.getState().openConfirm(
      "Clear canvas",
      "Remove all nodes and edges from the canvas? This cannot be undone.",
      () => {
        // The exact effect the Toolbar's Clear button wires.
        useFlowStore.getState().setGraph({ nodes: [], edges: [] });
        useFlowStore.getState().setError("OK: canvas cleared");
      },
    );
    // The Toolbar's onConfirm: capture the action, dismiss, run it.
    const req = useFlowStore.getState().confirm;
    useFlowStore.getState().dismissConfirm();
    req?.action();
    const s = useFlowStore.getState();
    expect(s.graph).toEqual({ nodes: [], edges: [] });
    expect(s.error).toBe("OK: canvas cleared");
    expect(s.confirm).toBeNull();
  });

  it("dialog cancel does nothing (dismissConfirm leaves the graph intact)", () => {
    const before = useFlowStore.getState().graph;
    useFlowStore.getState().openConfirm(
      "Clear canvas",
      "Remove all nodes and edges from the canvas? This cannot be undone.",
      () => {
        useFlowStore.getState().setGraph({ nodes: [], edges: [] });
      },
    );
    useFlowStore.getState().dismissConfirm();
    const s = useFlowStore.getState();
    expect(s.confirm).toBeNull();
    expect(s.graph).toBe(before);
    expect(s.error).toBeNull();
  });
});

describe("Task M2.4 deleteFlow store action (mocked fetch)", () => {
  beforeEach(() => {
    useFlowStore.setState({
      graph: {
        nodes: [{ id: "n1", kind: "dataset", props: {}, position: { x: 0, y: 0 } }],
        edges: [],
      },
      projection: [],
      currentFlowName: "my-run",
      validatedDoc: null,
      error: null,
      confirm: null,
      past: [],
      future: [],
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("success DELETEs the flow, resets the store to a fresh unsaved graph and toasts OK:", async () => {
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      expect(path).toBe("/api/flows/my-run");
      expect(init?.method).toBe("DELETE");
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const ok = await useFlowStore.getState().deleteFlow("my-run");
    expect(ok).toBe(true);
    const st = useFlowStore.getState();
    expect(st.graph).toEqual({ nodes: [], edges: [] });
    expect(st.currentFlowName).toBeNull();
    expect(st.projection).toEqual([]);
    expect(st.validatedDoc).toBeNull();
    expect(st.error).toBe("OK: deleted flow 'my-run'");
  });

  it("404 detail failure leaves the graph and the slug intact and toasts", async () => {
    const before = useFlowStore.getState().graph;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({ detail: "flow 'my-run' not found" }),
          { status: 404, statusText: "Not Found" },
        ),
      ),
    );
    const ok = await useFlowStore.getState().deleteFlow("my-run");
    expect(ok).toBe(false);
    const st = useFlowStore.getState();
    expect(st.graph).toBe(before);
    expect(st.currentFlowName).toBe("my-run");
    expect(st.error).toBe("flow 'my-run' not found");
  });

  it("network failure toasts and keeps the graph", async () => {
    const before = useFlowStore.getState().graph;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      }),
    );
    const ok = await useFlowStore.getState().deleteFlow("my-run");
    expect(ok).toBe(false);
    const st = useFlowStore.getState();
    expect(st.error).toBe("fetch failed");
    expect(st.graph).toBe(before);
    expect(st.currentFlowName).toBe("my-run");
  });
});