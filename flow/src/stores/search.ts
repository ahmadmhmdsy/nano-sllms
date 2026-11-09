/* ------------------------------------------------------------------ */
/* Task M2.2: node search - pure, React-free matching/cycling logic.  */
/* ------------------------------------------------------------------ */
/* The Toolbar search box (App.tsx) and the canvas highlight (via     */
/* toXYNodes' className) both key off these two functions, so the     */
/* whole feature is unit-testable without a DOM: searchMatches       */
/* decides WHICH nodes match, nextMatchIndex decides WHICH match      */
/* Enter selects next. Graph is the domain graph (flow/0.1 shape).    */
/* ------------------------------------------------------------------ */

import type { Graph } from "./graphStore";

/** Case-insensitive substring match of the query against a node's id,
 * kind, and label. Returns the matching node ids in graph order (the
 * order Enter cycles through them). An empty/whitespace query matches
 * nothing - the canvas treats that as "no search active", so callers
 * never have to special-case the empty string. */
export function searchMatches(graph: Graph, query: string): string[] {
  const q = query.trim().toLowerCase();
  if (q === "") return [];
  return graph.nodes
    .filter(
      (n) =>
        n.id.toLowerCase().includes(q) ||
        n.kind.toLowerCase().includes(q) ||
        (n.label ?? "").toLowerCase().includes(q),
    )
    .map((n) => n.id);
}

/** Index of the match Enter should focus next: the match AFTER
 * currentId in the match list, wrapping to the first match at the end.
 * A null currentId (nothing selected) or a currentId outside the match
 * list (selection not part of this search) starts at the first match.
 * Empty matches yield 0 - callers treat that as "no-op". */
export function nextMatchIndex(matches: string[], currentId: string | null): number {
  if (matches.length === 0) return 0;
  if (currentId === null) return 0;
  const i = matches.indexOf(currentId);
  if (i === -1) return 0;
  return (i + 1) % matches.length;
}
