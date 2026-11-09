import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";
import { Tabs, type TabItem } from "../src/index";

const tabs: TabItem[] = [
  { id: "a", label: "Tab A", content: <div>Panel A</div> },
  { id: "b", label: "Tab B", content: <div>Panel B</div> },
];

/* The repo has no @testing-library, so behavior tests navigate the
 * element tree directly (same white-box spirit as flow/'s ErrorBoundary
 * suite, which instantiates the class and pokes its state). */
function findDescendant(node: ReactElement, pred: (el: ReactElement) => boolean): ReactElement | null {
  if (pred(node)) return node;
  const children = node.props?.children;
  if (Array.isArray(children)) {
    for (const c of children) {
      if (c && typeof c === "object" && "type" in c) {
        const found = findDescendant(c as ReactElement, pred);
        if (found) return found;
      }
    }
  } else if (children && typeof children === "object" && "type" in children) {
    return findDescendant(children as ReactElement, pred);
  }
  return null;
}

describe("Tabs", () => {
  it("renders a tablist with one tab button per item", () => {
    const html = renderToStaticMarkup(<Tabs tabs={tabs} active="a" onChange={() => {}} />);
    expect(html).toContain('role="tablist"');
    expect(html).toContain('role="tab"');
    expect(html).toContain("Tab A");
    expect(html).toContain("Tab B");
  });

  it("marks the active tab and renders only its panel", () => {
    const html = renderToStaticMarkup(<Tabs tabs={tabs} active="b" onChange={() => {}} />);
    expect(html).toContain('aria-selected="true"');
    expect(html).toContain('aria-selected="false"');
    expect(html).toContain("Panel B");
    expect(html).not.toContain("Panel A");
    expect(html).toContain('role="tabpanel"');
  });

  it("wires aria-controls / aria-labelledby between tab and panel", () => {
    const html = renderToStaticMarkup(<Tabs tabs={tabs} active="a" onChange={() => {}} />);
    expect(html).toContain('aria-controls="tabpanel-a"');
    expect(html).toContain('id="tabpanel-a"');
    expect(html).toContain('aria-labelledby="tab-a"');
    expect(html).toContain('id="tab-a"');
  });

  it("gives the active tab a 0 tabindex and the others -1 (roving tabindex)", () => {
    const html = renderToStaticMarkup(<Tabs tabs={tabs} active="a" onChange={() => {}} />);
    expect(html).toContain('tabindex="0"');
    expect(html).toContain('tabindex="-1"');
  });

  it("calls onChange with the clicked tab id", () => {
    const onChange = vi.fn();
    const tree = Tabs({ tabs, active: "a", onChange }) as ReactElement;
    const tabB = findDescendant(tree, (el) => el.props?.role === "tab" && el.props?.id === "tab-b");
    expect(tabB).not.toBeNull();
    tabB!.props.onClick();
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("applies the caller className to the wrapper", () => {
    const html = renderToStaticMarkup(
      <Tabs tabs={tabs} active="a" onChange={() => {}} className="main-tabs" />,
    );
    expect(html).toContain("ui-tabs main-tabs");
  });

  it("renders nothing when there are no tabs", () => {
    expect(Tabs({ tabs: [], active: "a", onChange: () => {} })).toBeNull();
  });

  it("themes the tab buttons through CSS variables", () => {
    const html = renderToStaticMarkup(<Tabs tabs={tabs} active="a" onChange={() => {}} />);
    expect(html).toContain("var(--panel, #f6f6f6)");
    expect(html).toContain("var(--tab-active, #e9e7e2)");
  });
});
