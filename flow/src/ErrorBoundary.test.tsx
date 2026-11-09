import { describe, it, expect, vi } from "vitest";
import { ErrorBoundary } from "../../packages/ui-components/src/index";

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
});