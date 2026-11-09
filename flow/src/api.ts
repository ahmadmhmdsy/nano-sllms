import type { Graph, RegistrySnapshot, RunStatus } from "./store";
import type { DomainEdge, DomainNode } from "./stores/graphStore";

/** flow/0.1 document shape (GET /api/flows/{name} and PUT body). */
export interface FlowDocument {
  schema: "flow/0.1";
  meta: { name: string };
  graph: Graph;
}

/** model/0.1 document shape (GET/POST /api/models/{name}). */
export interface ModelDocument {
  format: "model/0.1";
  name: string;
  nodes: DomainNode[];
  edges: DomainEdge[];
  meta: Record<string, unknown>;
}

/** Pull the payload out; 400 uses {"errors": [...]}, 404/409 {"detail"}. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new Error(formatApiError(res.status, res.statusText, path, body));
  }
  return (await res.json()) as T;
}

/** Human-readable message from the two backend error shapes. */
export function formatApiError(
  status: number,
  statusText: string,
  path: string,
  body: unknown,
): string {
  const rec = body as Record<string, unknown> | null;
  if (rec && Array.isArray(rec.errors) && rec.errors.length > 0) {
    return rec.errors.map(String).join("; ");
  }
  if (rec && typeof rec.detail === "string" && rec.detail.length > 0) {
    return rec.detail;
  }
  return status + " " + statusText + ": " + path;
}

/** Error message from a caught unknown (fetch/network or request error). */
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** GET /api/run/status payload — flow server is the bearer of truth for
 * this shape (T8 review MINOR-8). Mapping into the UI-level store.RunStatus
 * happens in the Task 9/10 run wiring. */
export interface ApiRunStatus {
  running: boolean;
  exit_code: null | number;
  started_at: string | null;
  exit_at: string | null;
  tail: string[];
}

/** Map the GET /api/run/status backend payload into the UI-level
 * store.RunStatus (T7 contract + runner.py passthrough semantics):
 * - running: true                        -> "running" (live values)
 * - running === false, exit_code !== null -> "done" when 0, else "error"
 * - running === false, exit_code === null -> "idle" (last run finished
 *   before a tail existed / no run since process start) */
export function mapRunStatus(s: ApiRunStatus): RunStatus {
  if (s.running) return { state: "running", message: "running…", exitCode: undefined };
  if (s.exit_code !== null) {
    return s.exit_code === 0
      ? { state: "done", message: "exit code 0 — done", exitCode: 0 }
      : { state: "error", message: "exit code " + String(s.exit_code) + " — error", exitCode: s.exit_code };
  }
  return { state: "idle", message: "no live run", exitCode: undefined };
}

export const api = {
  getNodes: () => request<RegistrySnapshot>("/api/nodes"),
  getFlows: () => request<string[]>("/api/flows"),
  getFlow: (name: string) => request<FlowDocument>("/api/flows/" + encodeURIComponent(name)),
  saveFlow: (name: string, doc: FlowDocument) =>
    request<{ ok: true; name: string }>("/api/flows/" + encodeURIComponent(name), {
      method: "PUT",
      body: JSON.stringify(doc),
    }),
  /** Task M2.4: DELETE /api/flows/{name} — removes the saved flow
   * server-side. Returns {"ok": true} on success; the shared request
   * helper throws with the backend's {"detail": ...} on failure. */
  deleteFlow: (name: string) =>
    request<{ ok: true }>("/api/flows/" + encodeURIComponent(name), {
      method: "DELETE",
    }),
  validateFlow: (doc: FlowDocument) =>
    request<{ ok: boolean; errors: string[] }>("/api/validate", {
      method: "POST",
      body: JSON.stringify(doc),
    }),
  runFlow: (name: string) =>
    request<{ ok: true; pid: number }>("/api/run", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  /** Model-graph APIs (F2 Task 5): list slugs, load one doc, save one doc
   * (server: flow/server/app.py, atomic .modelgraph.json store). */
  getModels: async () => (await request<{ models: string[] }>("/api/models")).models,
  getModel: (name: string) =>
    request<ModelDocument>("/api/models/" + encodeURIComponent(name)),
  saveModel: (name: string, doc: ModelDocument) =>
    request<{ ok: true; name: string }>("/api/models/" + encodeURIComponent(name), {
      method: "POST",
      body: JSON.stringify(doc),
    }),
  runStatus: () => request<ApiRunStatus>("/api/run/status"),
  runStop: () => request<{ ok: true; stop_flag: string }>("/api/run/stop", { method: "POST" }),
};
