import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Toast } from "../src/index";

describe("Toast", () => {
  it("renders the message as an alert by default", () => {
    const html = renderToStaticMarkup(<Toast message="OK: saved" />);
    expect(html).toContain('role="alert"');
    expect(html).toContain('class="toast"');
    expect(html).toContain("OK: saved");
  });

  it("renders nothing for an empty message", () => {
    expect(renderToStaticMarkup(<Toast message="" />)).toBe("");
  });

  it("supports the passive status role", () => {
    const html = renderToStaticMarkup(<Toast message="imported" role="status" />);
    expect(html).toContain('role="status"');
  });

  it("applies the caller className after the base class", () => {
    const html = renderToStaticMarkup(<Toast message="hi" className="toast-btn" />);
    expect(html).toContain("toast toast-btn");
  });

  it("themes through the --out / --on-out CSS variables", () => {
    const html = renderToStaticMarkup(<Toast message="hi" />);
    expect(html).toContain("var(--out, #c62828)");
    expect(html).toContain("var(--on-out, #fff)");
  });
});
