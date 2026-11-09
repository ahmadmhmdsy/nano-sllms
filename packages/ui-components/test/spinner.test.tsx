import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Spinner } from "../src/index";

describe("Spinner", () => {
  it("renders a status element announced as loading", () => {
    const html = renderToStaticMarkup(<Spinner />);
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-label="Loading"');
    expect(html).toContain('class="spinner"');
  });

  it("appends the caller className after the base class", () => {
    const html = renderToStaticMarkup(<Spinner className="spinner-btn" />);
    expect(html).toContain("spinner spinner-btn");
  });

  it("themes its color through the --color-primary CSS variable", () => {
    const html = renderToStaticMarkup(<Spinner />);
    expect(html).toContain("var(--color-primary, currentColor)");
  });
});
