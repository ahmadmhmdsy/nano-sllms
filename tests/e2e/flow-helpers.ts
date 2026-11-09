import { expect, type Page } from "@playwright/test";

/**
 * Shared helpers for the flow/ E2E suite.
 *
 * The flow editor (flow/src) is a React + @xyflow/react app that ships
 * stable affordances the suite drives directly:
 *   - toolbar buttons with ids (toolbar-save, toolbar-validate, ...)
 *   - canvas right-click add menu (role="menuitem" per node kind)
 *   - save/open modals (role="dialog")
 *   - a toast channel (role="alert")
 *   - react-flow__* classes plus data-nodeid / data-handleid on handles
 *
 * Every assertion goes through Playwright's auto-retrying expect(); there
 * are no fixed sleeps anywhere in the suite.
 */

/** Canvas-relative click points for the 4-node chain (1600x1000 viewport). */
export const CHAIN = {
  dataset: { x: 140, y: 150 },
  prepare: { x: 470, y: 150 },
  tokenize: { x: 800, y: 150 },
  train: { x: 470, y: 500 },
} as const;

/** Wait until the app shell is interactive: toolbar + registry-loaded palette. */
export async function gotoFlowEditor(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.locator("#toolbar-validate")).toBeVisible();
  // Palette cards only render after GET /api/nodes resolves (loading state otherwise).
  await expect(page.locator(".palette-card").first()).toBeVisible();
}

/**
 * Add a node of `kind` by right-clicking the canvas at canvas-relative
 * (x, y) and picking the kind from the add menu. Returns the new node's
 * deterministic id (n1, n2, ... in add order, mirroring the store's
 * nextNodeId counter: highest existing n<k> + 1, computed BEFORE the add).
 */
export async function addNode(
  page: Page,
  kind: string,
  at: { x: number; y: number },
): Promise<string> {
  const canvas = page.locator(".canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("canvas not visible");
  // Mirror the store's nextNodeId: highest existing n<k> + 1.
  const ids = await page.locator(".react-flow__node").evaluateAll((els) =>
    els.map((el) => (el as HTMLElement).dataset.id ?? ""),
  );
  const max = ids.reduce((m, id) => {
    const n = /^n(\d+)$/.exec(id);
    return n ? Math.max(m, Number(n[1])) : m;
  }, 0);
  const newId = "n" + (max + 1);
  await page.mouse.click(box.x + at.x, box.y + at.y, { button: "right" });
  const menu = page.locator(".add-menu");
  await expect(menu).toBeVisible();
  await menu.getByRole("menuitem", { name: kind, exact: true }).click();
  await expect(page.locator(".add-menu")).toHaveCount(0);
  // The new node must exist with the predicted id.
  await expect(
    page.locator(".react-flow__node[data-id=\"" + newId + "\"]"),
  ).toBeVisible();
  return newId;
}

/**
 * Drag from one node's port handle to another's to create an edge.
 * React Flow v12 tracks the closest handle under the pointer while
 * connecting, so moving onto the target handle's centre and releasing
 * completes the connection.
 */
export async function connect(
  page: Page,
  fromId: string,
  fromPort: string,
  toId: string,
  toPort: string,
): Promise<void> {
  const source = page.locator(
    ".react-flow__handle[data-nodeid=\"" + fromId + "\"][data-handleid=\"" + fromPort + "\"]",
  );
  const target = page.locator(
    ".react-flow__handle[data-nodeid=\"" + toId + "\"][data-handleid=\"" + toPort + "\"]",
  );
  await expect(source).toBeVisible();
  await expect(target).toBeVisible();
  const sb = await source.boundingBox();
  const tb = await target.boundingBox();
  if (!sb || !tb) throw new Error("handle not visible");
  await page.mouse.move(sb.x + sb.width / 2, sb.y + sb.height / 2);
  await page.mouse.down();
  await page.mouse.move(tb.x + tb.width / 2, tb.y + tb.height / 2, { steps: 10 });
  await page.mouse.up();
}

/** Click a node body to select it (the inspector then shows that node). */
export async function selectNode(page: Page, id: string): Promise<void> {
  await page.locator(".react-flow__node[data-id=\"" + id + "\"] .pipeline-node").click();
}

export async function setLabel(page: Page, label: string): Promise<void> {
  await page.locator("#node-label").fill(label);
}

export async function setNumberProp(page: Page, name: string, value: string): Promise<void> {
  await page.locator("#prop-" + name).fill(value);
}

export async function setSelectProp(page: Page, name: string, value: string): Promise<void> {
  await page.locator("#prop-" + name).selectOption(value);
}
