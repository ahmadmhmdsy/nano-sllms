import { expect, test } from "@playwright/test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CHAIN,
  addNode,
  connect,
  gotoFlowEditor,
  selectNode,
  setLabel,
  setNumberProp,
  setSelectProp,
} from "./flow-helpers";

/**
 * flow/ E2E: build a valid 4-node pipeline through the real UI
 * (right-click add menu, drag-connect, inspector props), validate it
 * against the FastAPI backend, save it, reload the page and reopen it.
 *
 * The knob values mirror the golden document
 * flow/flows/example-linear-smoke.flow.json, which the server-side
 * validator accepts.
 */
test.describe("flow editor", () => {
  test("add -> connect -> validate -> save -> reload -> open restores the graph", async ({ page }) => {
    await gotoFlowEditor(page);

    // 1. add the four nodes via the canvas right-click menu
    const dataset = await addNode(page, "dataset", CHAIN.dataset);
    const prepare = await addNode(page, "prepare", CHAIN.prepare);
    const tokenize = await addNode(page, "tokenize", CHAIN.tokenize);
    const train = await addNode(page, "train", CHAIN.train);
    await expect(page.locator(".react-flow__node")).toHaveCount(4);

    // 2. wire the chain: dataset -> prepare -> tokenize -> train
    await connect(page, dataset, "cleaned", prepare, "raw-dir");
    await connect(page, prepare, "cleaned-dir", tokenize, "cleaned-dir");
    await connect(page, tokenize, "shard-dir", train, "shard-dir");
    await expect(page.locator(".react-flow__edge")).toHaveCount(3);

    // 3. fill in the required knobs through the inspector
    await selectNode(page, dataset);
    await setLabel(page, "RowTour-360");

    await selectNode(page, prepare);
    await setNumberProp(page, "min_chars", "40");
    await setNumberProp(page, "rows", "480000");
    await setNumberProp(page, "val_fraction", "0.05");

    await selectNode(page, tokenize);
    await setNumberProp(page, "seq_len", "512");
    await setNumberProp(page, "vocab", "32000");

    await selectNode(page, train);
    await setSelectProp(page, "preset", "Small (~12M, smoke)");
    await setNumberProp(page, "steps", "200");
    await setSelectProp(page, "lr_preset", "Pretrain 4e-4");

    // 4. validate against the backend (config_gen dry-run)
    await page.locator("#toolbar-validate").click();
    await expect(page.getByRole("alert")).toContainText("OK: validation passed");

    // 5. save under a known slug
    await page.locator("#toolbar-save").click();
    const saveDialog = page.getByRole("dialog", { name: "Save flow" });
    await expect(saveDialog).toBeVisible();
    await saveDialog.locator("#save-name").fill("e2e-smoke-flow");
    await saveDialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(page.getByRole("dialog")).toHaveCount(0);
    // the delete button only enables once a flow is loaded/saved
    await expect(page.locator("#toolbar-delete")).toBeEnabled();

    // 6. reload and reopen the saved flow from the server
    await page.reload();
    await gotoFlowEditor(page);
    await expect(page.locator(".react-flow__node")).toHaveCount(0);
    await page.locator("#toolbar-open").click();
    const openDialog = page.getByRole("dialog", { name: "Open flow" });
    await expect(openDialog).toBeVisible();
    await openDialog.getByRole("button", { name: "e2e-smoke-flow", exact: true }).click();
    await expect(page.getByRole("dialog")).toHaveCount(0);

    // the full graph is restored from the server document
    await expect(page.locator(".react-flow__node")).toHaveCount(4);
    await expect(page.locator(".react-flow__edge")).toHaveCount(3);
    await expect(
      page.locator(".pipeline-node-title").filter({ hasText: "RowTour-360" }),
    ).toBeVisible();
  });

  test("undo/redo via toolbar buttons and keyboard shortcuts (M1.1)", async ({ page }) => {
    await gotoFlowEditor(page);

    const undo = page.locator("#toolbar-undo");
    const redo = page.locator("#toolbar-redo");

    // nothing to undo on a fresh graph
    await expect(undo).toBeDisabled();
    await expect(redo).toBeDisabled();

    await addNode(page, "dataset", CHAIN.dataset);
    await expect(page.locator(".react-flow__node")).toHaveCount(1);
    await expect(undo).toBeEnabled();
    await expect(redo).toBeDisabled();

    // toolbar undo removes the node, toolbar redo brings it back
    await undo.click();
    await expect(page.locator(".react-flow__node")).toHaveCount(0);
    await expect(redo).toBeEnabled();
    await redo.click();
    await expect(page.locator(".react-flow__node")).toHaveCount(1);

    // keyboard: Ctrl+Z undoes, Ctrl+Shift+Z redoes (focus on the canvas pane)
    await page.locator(".canvas").click({ position: { x: 30, y: 30 } });
    await page.keyboard.press("Control+z");
    await expect(page.locator(".react-flow__node")).toHaveCount(0);
    await page.keyboard.press("Control+Shift+z");
    await expect(page.locator(".react-flow__node")).toHaveCount(1);
  });

  test("delete key removes the selected node (M1.3)", async ({ page }) => {
    await gotoFlowEditor(page);

    const id = await addNode(page, "dataset", CHAIN.dataset);
    await expect(page.locator(".react-flow__node")).toHaveCount(1);

    // select the node, then press Backspace (react-flow's default deleteKeyCode)
    await selectNode(page, id);
    await expect(page.locator(".react-flow__node[data-id=\"" + id + "\"]")).toHaveClass(/selected/);
    await page.keyboard.press("Backspace");
    await expect(page.locator(".react-flow__node")).toHaveCount(0);
  });

  test("export downloads the document and import loads it back (M2.3)", async ({ page }) => {
    await gotoFlowEditor(page);

    // export the empty graph: the download name falls back to "untitled"
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#toolbar-export").click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe("untitled.flow.json");
    const path = await download.path();
    expect(path).not.toBeNull();
    const text = await readFile(path as string, "utf8");
    const doc = JSON.parse(text);
    expect(doc.schema).toBe("flow/0.1");
    expect(doc.meta.name).toBe("untitled");

    // import a two-node document through the hidden file input
    const imported = {
      schema: "flow/0.1",
      meta: { name: "imported-flow", description: "e2e import fixture" },
      graph: {
        nodes: [
          { id: "n1", kind: "dataset", label: "Imported-Data", props: {}, position: { x: 100, y: 100 } },
          { id: "n2", kind: "train", props: { steps: 10 }, position: { x: 400, y: 100 } },
        ],
        edges: [],
      },
    };
    const dir = await mkdtemp(join(tmpdir(), "e2e-import-"));
    const tmp = join(dir, "e2e-import.flow.json");
    await writeFile(tmp, JSON.stringify(imported), "utf8");
    try {
      await page.locator("input[type=\"file\"]").setInputFiles(tmp);
      await expect(page.getByRole("alert")).toContainText(
        "OK: imported e2e-import.flow.json as 'imported-flow'",
      );
      await expect(page.locator(".react-flow__node")).toHaveCount(2);
      await expect(
        page.locator(".pipeline-node-title").filter({ hasText: "Imported-Data" }),
      ).toBeVisible();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
