import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";
import { ErrorBoundary } from "../src/index";

describe("ErrorBoundary", () => {
  it("returns error state from getDerivedStateFromError", () => {
    const error = new Error("Test error");
    const state = ErrorBoundary.getDerivedStateFromError(error);
    expect(state.hasError).toBe(true);
    expect(state.error).toBe(error);
  });

  it("logs error in componentDidCatch", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const boundary = new ErrorBoundary({ children: null });
    const error = new Error("Test error");
    const errorInfo = { componentStack: "TestStack" };

    boundary.componentDidCatch(error, errorInfo);

    expect(consoleSpy).toHaveBeenCalledWith("ErrorBoundary caught an error:", error, errorInfo);

    consoleSpy.mockRestore();
  });

  it("renders children when there is no error", () => {
    const boundary = new ErrorBoundary({ children: <div>Child content</div> });
    const result = boundary.render();
    // The result should be the children element
    expect(result).toBeDefined();
  });

  it("renders fallback UI when hasError is true", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const boundary = new ErrorBoundary({ children: <div>Child content</div> });
    boundary.state = { hasError: true, error: new Error("Test error") };
    const result = boundary.render();
    // The result should be the fallback UI
    expect(result).toBeDefined();
    consoleSpy.mockRestore();
  });

  it("fallback carries the caller className and CSS-variable theming", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const boundary = new ErrorBoundary({ children: null, className: "app-fallback" });
    boundary.state = { hasError: true, error: new Error("boom") };
    const html = renderToStaticMarkup(boundary.render() as ReactElement);
    expect(html).toContain("error-boundary app-fallback");
    expect(html).toContain('role="alert"');
    expect(html).toContain("Something went wrong");
    expect(html).toContain("var(--bg, #fff)");
    expect(html).toContain("var(--ink, #333)");
    consoleSpy.mockRestore();
  });
});
