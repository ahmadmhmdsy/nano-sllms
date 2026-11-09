import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConfirmDialog } from "../src/index";

/* Presentational contract (same cases flow/'s M2.4 suite pinned through
 * the local module): null when closed, modal markup when open. The
 * shared package keeps the behavior identical; the className and
 * CSS-variable theming are new shared-library guarantees. */
describe("ConfirmDialog", () => {
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

  it("applies the caller className to the dialog box", () => {
    const html = renderToStaticMarkup(
      <ConfirmDialog
        open={true}
        title="Delete flow"
        message="Permanently delete?"
        onConfirm={() => {}}
        onDismiss={() => {}}
        className="compact"
      />,
    );
    expect(html).toContain("modal compact");
  });

  it("themes surfaces through CSS variables", () => {
    const html = renderToStaticMarkup(
      <ConfirmDialog
        open={true}
        title="t"
        message="m"
        onConfirm={() => {}}
        onDismiss={() => {}}
      />,
    );
    expect(html).toContain("var(--panel, #fff)");
    expect(html).toContain("var(--ink, #333)");
    expect(html).toContain("var(--backdrop, rgba(0, 0, 0, 0.15))");
  });
});
