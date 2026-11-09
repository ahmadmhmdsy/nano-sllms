import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ReactFlowProvider, type NodeProps } from "@xyflow/react";
import ModelNode from "./ModelNode";
import type { ModelNodeData, ModelNode as ModelNodeType } from "./modelStore";

/* ------------------------------------------------------------------ */
/* Task M3.1: visual error indication on model canvas nodes. The      */
/* repo has no @testing-library; per the established pattern (see     */
/* confirm.test.tsx) the presentational contract is verified through   */
/* renderToStaticMarkup. ModelNode renders <Handle>, which needs the  */
/* xyflow store + handle-config contexts, so the render is wrapped in */
/* <ReactFlowProvider> (it provides both).                            */
/* ------------------------------------------------------------------ */

/** Hand-rolled NodeProps: xyflow v12 requires type/dragging/zIndex/
 * selectable/deletable/draggable/isConnectable/positionAbsoluteX/Y
 * on top of id/data. */
function makeProps(data: ModelNodeData): NodeProps<ModelNodeType> {
  return {
    id: "n1",
    type: "model",
    data,
    selected: false,
    dragging: false,
    zIndex: 0,
    selectable: true,
    deletable: true,
    draggable: true,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
  };
}

function makeData(overrides: Partial<ModelNodeData> = {}): ModelNodeData {
  return {
    kind: "linear",
    label: "Linear",
    props: {},
    inputs: [{ name: "in", label: "in" }],
    outputs: [{ name: "out", label: "out" }],
    shape: { b: 1, s: 2, d: 3 },
    params: 6,
    error: null,
    ...overrides,
  };
}

function renderNode(data: ModelNodeData): string {
  return renderToStaticMarkup(
    <ReactFlowProvider>
      <ModelNode {...makeProps(data)} />
    </ReactFlowProvider>,
  );
}

/** The wrapper is the first <div> in the markup (the providers render
 * no DOM); pull its exact class attribute for precise assertions. */
function wrapperClass(html: string): string {
  const m = html.match(/<div class="([^"]*)"/);
  if (!m) throw new Error("no wrapper div found in: " + html);
  return m[1];
}

describe("Task M3.1 ModelNode visual error indication", () => {
  it("node with an error gets model-node-error on the wrapper", () => {
    const html = renderNode(makeData({ error: "walk error: prop d must be a finite number" }));
    expect(wrapperClass(html)).toContain("model-node-error");
    expect(wrapperClass(html)).not.toContain("model-node-uncomputed");
  });

  it("node without an inferred shape gets model-node-uncomputed on the wrapper", () => {
    const html = renderNode(makeData({ shape: null }));
    expect(wrapperClass(html)).toContain("model-node-uncomputed");
    expect(wrapperClass(html)).not.toContain("model-node-error");
  });

  it("node with a shape and no error gets neither state class", () => {
    const html = renderNode(makeData());
    expect(wrapperClass(html)).toBe("model-node");
  });

  it("the full error message is the hover tooltip (wrapper title + error badge)", () => {
    const html = renderNode(makeData({ error: "walk error: prop d must be a finite number" }));
    expect(html).toContain('title="walk error: prop d must be a finite number"');
    expect(html).toContain("model-node-error-badge");
  });
});
