import { expect, test } from "@playwright/test";

/**
 * viz/ E2E: the model-architecture explorer is a static React app whose
 * shared Tabs component ships role="tab" / role="tabpanel" affordances and
 * whose diagrams expose data-model / data-block / data-layer on every
 * clickable block. The suite drives those real affordances.
 */
test.describe("viz model architecture explorer", () => {
  test("switching tabs renders each tab's content", async ({ page }) => {
    await page.goto("/");

    // Stacks is the default tab: three model cards (nano, qwen38, smollm2)
    await expect(page.getByRole("tab", { name: "Stacks" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(page.locator(".mcard[data-model-col]")).toHaveCount(3);
    // the block inspector starts in its pick-a-block state
    await expect(page.locator("#detail")).toContainText("Pick a block");

    await page.getByRole("tab", { name: "Compare" }).click();
    await expect(page.getByRole("tabpanel")).toContainText("Side-by-side specs");

    await page.getByRole("tab", { name: "Demo" }).click();
    await expect(page.getByRole("tabpanel")).toContainText("Watch a prompt travel");

    await page.getByRole("tab", { name: "About" }).click();
    await expect(page.getByRole("tabpanel")).toContainText("About");

    // back to Stacks: the model cards are there again
    await page.getByRole("tab", { name: "Stacks" }).click();
    await expect(page.locator(".mcard[data-model-col]")).toHaveCount(3);
  });

  test("clicking a block in a model card updates the inspector", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".mcard[data-model-col]")).toHaveCount(3);

    const inspector = page.locator("#detail");
    await expect(inspector).toContainText("Pick a block");

    // click the embedding table inside the nano model card's diagram
    const embed = page.locator('[data-model="nano"][data-block="embed"]');
    await expect(embed).toBeVisible();
    await embed.click();

    // the inspector now shows the nano model's embedding entry
    await expect(inspector).toContainText("Embedding table");
    await expect(inspector).toContainText("nano_SLMs");
    await expect(inspector).not.toContainText("Pick a block");

    // click a decoder layer block -> the layer-specific entry
    const layer = page.locator('[data-model="nano"][data-block="layer"][data-layer="1"]');
    await layer.click();
    await expect(inspector).toContainText("decoder block");
  });
});
