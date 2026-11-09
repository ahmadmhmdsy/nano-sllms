import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";
import { SearchInput } from "../src/index";

/* The repo has no @testing-library, so behavior tests navigate the
 * element tree directly (same white-box spirit as flow/'s ErrorBoundary
 * suite, which instantiates the class and pokes its state). */
function findInput(node: ReactElement): ReactElement | null {
  if (node.type === "input") return node;
  const children = node.props?.children;
  if (Array.isArray(children)) {
    for (const c of children) {
      if (c && typeof c === "object" && "type" in c) {
        const found = findInput(c as ReactElement);
        if (found) return found;
      }
    }
  } else if (children && typeof children === "object" && "type" in children) {
    return findInput(children as ReactElement);
  }
  return null;
}

describe("SearchInput", () => {
  it("associates a visible label with the input via htmlFor/id", () => {
    const html = renderToStaticMarkup(
      <SearchInput id="q" value="" onChange={() => {}} label="Find node" />,
    );
    expect(html).toContain('for="q"');
    expect(html).toContain('id="q"');
    expect(html).toContain("Find node");
  });

  it("falls back to an aria-label when no visible label is given", () => {
    const html = renderToStaticMarkup(
      <SearchInput id="q" value="" onChange={() => {}} ariaLabel="Find node" />,
    );
    expect(html).toContain('aria-label="Find node"');
  });

  it("defaults the accessible name to Search", () => {
    const html = renderToStaticMarkup(<SearchInput id="q" value="" onChange={() => {}} />);
    expect(html).toContain('aria-label="Search"');
  });

  it("renders value and placeholder", () => {
    const html = renderToStaticMarkup(
      <SearchInput id="q" value="abc" onChange={() => {}} placeholder="find node…" />,
    );
    expect(html).toContain('value="abc"');
    expect(html).toContain('placeholder="find node…"');
  });

  it("calls onChange with the typed value", () => {
    const onChange = vi.fn();
    const tree = SearchInput({ id: "q", value: "", onChange }) as ReactElement;
    const input = findInput(tree);
    expect(input).not.toBeNull();
    input!.props.onChange({ target: { value: "abc" } });
    expect(onChange).toHaveBeenCalledWith("abc");
  });

  it("forwards keydown events", () => {
    const onKeyDown = vi.fn();
    const tree = SearchInput({ id: "q", value: "", onChange: () => {}, onKeyDown }) as ReactElement;
    const input = findInput(tree);
    expect(input).not.toBeNull();
    input!.props.onKeyDown({ key: "Enter" });
    expect(onKeyDown).toHaveBeenCalledWith({ key: "Enter" });
  });

  it("applies the caller className to the wrapper", () => {
    const html = renderToStaticMarkup(
      <SearchInput id="q" value="" onChange={() => {}} className="toolbar-search" />,
    );
    expect(html).toContain("ui-search toolbar-search");
  });

  it("themes through CSS variables", () => {
    const html = renderToStaticMarkup(<SearchInput id="q" value="" onChange={() => {}} />);
    expect(html).toContain("var(--bg, #fff)");
    expect(html).toContain("var(--ink, #333)");
    expect(html).toContain("var(--line, #ccc)");
  });
});
