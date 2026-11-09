import { expect, test } from "@playwright/test";

/**
 * webui/ E2E: the Gradio shell (webui/app.py) boots on :7861 --no-browser
 * (the headless-check port from webui/AGENTS.md; 3010 is taken by the flow
 * backend and 7860 is the Gradio default).
 *
 * The suite verifies what is verifiable without a GPU: the dashboard
 * renders and the chat UI is present. Actually loading a chat model needs
 * GPU checkpoint weights (and this machine may be running a training job,
 * which the webui itself refuses to co-run with), so that part is skipped
 * with an explicit reason instead of being faked.
 */
test.describe("webui Gradio shell", () => {
  test("boot renders the dashboard", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle("nano_SLMs");

    // dashboard (default tab): heading + read-only run table
    await expect(page.getByRole("heading", { name: "nano_SLMs" })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Run dashboard (read-only)" }),
    ).toBeVisible();
    // the run table (gr.Dataframe) renders as a real table
    await expect(page.locator("table")).toBeVisible();
    // GPU/disk head line from the status snapshot
    await expect(page.getByText("Disk free:", { exact: false })).toBeVisible();

    // all six tabs exist (Dashboard/Monitor/Train/Chat/Settings + Model)
    for (const tab of ["Dashboard", "Monitor", "Train", "Chat", "Settings", "Model"]) {
      await expect(page.getByRole("tab", { name: tab })).toBeVisible();
    }
  });

  test("chat tab renders the chat UI; model loading skipped (needs GPU + weights)", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("tab", { name: "Chat" }).click();

    // chat surface: heading, message input, checkpoint picker, load/unload
    await expect(
      page.getByRole("heading", { name: "Chat with a trained checkpoint" }),
    ).toBeVisible();
    await expect(page.getByPlaceholder("Type a message...")).toBeVisible();
    await expect(page.getByRole("button", { name: "Load", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Unload", exact: true })).toBeVisible();
    // the checkpoint picker is a combobox input labelled "Checkpoint"
    await expect(page.getByRole("combobox", { name: "Checkpoint" })).toBeVisible();

    // no model is loaded on a fresh boot
    await expect(page.getByText("No model loaded.")).toBeVisible();

    // Unload with no model loaded is a safe no-op that keeps the status
    await page.getByRole("button", { name: "Unload", exact: true }).click();
    await expect(page.getByText("No model loaded.")).toBeVisible();

    test.skip(
      true,
      "Loading a chat model requires GPU checkpoint weights (and a training job may be " +
        "live on this machine, which the webui refuses to co-run with). The dashboard " +
        "and chat UI presence are the verifiable surface here.",
    );
  });
});
