import { lazy, Suspense, useMemo, useState } from "react";
import { ErrorBoundary, Spinner, Tabs } from "../../packages/ui-components/src/index";
import { MODELS } from "./registry/models";
import { fmtP, pTotal } from "./engine/params";
import * as hyb from "./viz/kinds/hybridLinearMoe";
import * as dense from "./viz/kinds/denseGqaMoe";
import { Diagram, type SelRef } from "./viz/Diagram";
import { BlockInspector } from "./viz/BlockInspector";
import { KnowledgeMap } from "./viz/KnowledgeMap";
import { SpecTable, PurposeTable, SharedDna, Differences } from "./viz/CompareTables";

/* Task M6.2: code splitting — TokenFlowDemo (and its simulation engine) is
 * only needed in the Demo tab, so it lazy-loads as a separate chunk on
 * first tab switch instead of shipping in the initial bundle. It is a named
 * export, so the lazy() factory maps it onto default. */
const TokenFlowDemo = lazy(() =>
  import("./viz/TokenFlowDemo").then((m) => ({ default: m.TokenFlowDemo })),
);

type Tab = "stacks" | "compare" | "demo" | "about";

type Model = (typeof MODELS)[number];

export function App() {
  const [tab, setTab] = useState<Tab>("stacks");
  const [sel, setSel] = useState<SelRef | null>(null);
  const [live, setLive] = useState<SelRef | null>(null);
  const specs = useMemo(() => MODELS, []);

  return (
    <ErrorBoundary>
      <div>
        {/* Shared tab bar (packages/ui-components): full ARIA tablist / tab /
            tabpanel wiring; each tab's body mounts only while active, so
            per-tab state resets on switch exactly like the previous
            conditional rendering. */}
        <Tabs<Tab>
          active={tab}
          onChange={(id) => setTab(id)}
          tabs={[
            { id: "stacks", label: "Stacks", content: <Stacks specs={specs} sel={sel} setSel={setSel} live={live} setLive={setLive} /> },
            { id: "compare", label: "Compare", content: <Compare specs={specs} /> },
            {
              id: "demo",
              label: "Demo",
              content: (
                <Suspense fallback={<Spinner />}>
                  <div>
                  <section>
                    <h2 className="sec">Watch a prompt travel</h2>
                    <p className="secd">
                      A simulated forward pass: tokenize → embed → for every layer (norm → mixer → residual add → norm → FFN/experts → residual add) → final norm → LM head → softmax. Values are illustrative — the SIMULATED badge follows them everywhere.
                    </p>
                    {specs.map((spec) => (
                      <section key={spec.id} style={{ marginTop: 18 }}>
                        <h3 style={{ fontWeight: 600, color: spec.accent }}>{spec.name}</h3>
                        <div data-demo-model={spec.id}>
                          <TokenFlowDemo spec={spec} onLiveLayer={(l) => setLive(l == null ? null : { model: spec.id, block: "layer", layer: l })} />
                        </div>
                      </section>
                    ))}
                  </section>
                  </div>
                </Suspense>
              ),
            },
            { id: "about", label: "About", content: <About specs={specs} /> },
          ]}
        />
      </div>
    </ErrorBoundary>
  );
}

function Stacks({
  specs,
  sel,
  setSel,
  live,
  setLive,
}: {
  specs: Model[];
  sel: SelRef | null;
  setSel: (s: SelRef) => void;
  live: SelRef | null;
  setLive: (s: SelRef) => void;
}) {
  return (
    <div>
      <div className="grid">
        {/* registry-driven: maps over specs — nothing hard-coded (pitfall checklist) */}
        <div className="cols">
          {specs.map((spec) => {
            const isHyb = spec.archKind === "hybridLinearMoe";
            return (
              <div className="mcard" key={spec.id} data-model-col={spec.id}>
                <div className="mhead">
                  <h2>{spec.name}</h2>
                  <div className="sub">{spec.fullName}</div>
                  <div className="badges">
                    <span className="badge">{spec.shape.layers} layers</span>
                    <span className="badge">d_model {spec.shape.d}</span>
                    {isHyb ? (
                      <>
                        <span className="badge">{spec.hybrid?.layerPattern}</span>
                        <span className="badge">full: {spec.shape.heads}Q / {spec.shape.kv}KV</span>
                        <span className="badge">MoE {spec.hybrid?.experts?.total}e · {spec.hybrid?.experts?.active} active + 1 shared</span>
                      </>
                    ) : (
                      <>
                        <span className="badge">{spec.shape.heads}Q / {spec.shape.kv}KV heads</span>
                        <span className="badge">FFN {spec.shape.ffn}</span>
                      </>
                    )}
                    <span className="badge hot" style={{ background: spec.accent }}>{hybOrDenseBadge(spec)}</span>
                    <span className="badge">ctx {spec.shape.ctx}</span>
                    <span className="badge">vocab {Math.round(spec.shape.vocab / 1024) === spec.shape.vocab / 1024 ? spec.shape.vocab : fmtP(spec.shape.vocab)}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "var(--dim)", margin: "8px 0 2px" }}>{spec.blurb}</p>
                </div>
                <Diagram spec={spec} sel={sel} live={live} onPick={(s) => { setSel(s); setLive(s); }} />
              </div>
            );
          })}
        </div>
        <aside id="detail" className="detail">
          <div className="crumb">block inspector</div>
          <BlockInspector spec={specForSel(specs, sel)} sel={sel} />
          <div className="note">
            Every number is drawn from the real specs: <span className="mono">configs/target.yaml</span> + <span className="mono">src/model.py</span> for nano_SLMs; published <span className="mono">config.json</span> for the external models.
          </div>
        </aside>
      </div>
      <section className="block">
        <h2 className="sec">Where the learning and knowledge live</h2>
        <p className="secd">
          “Knowledge” is not a database — it is baked into the weights during training. Interpretability research (Geva et al. 2021: FFNs behave as key–value memories; Meng et al. 2022 “ROME”: factual edits succeed in mid-layer FFNs) keeps finding the same depth profile. Treat it as a rule-of-thumb, not a per-layer measurement.
        </p>
        <div className="two three">
          {specs.map((spec) => (
            <div className="mcard" key={"km-" + spec.id} data-kmap={spec.id}>
              <KnowledgeMap spec={spec} />
            </div>
          ))}
        </div>
        <div className="mcard" style={{ marginTop: 22, overflowX: "auto" }}>
          <PurposeTable specs={specs} />
        </div>
      </section>
      <section className="block">
        <h2 className="sec">Anything shared? — the two small ones: almost everything</h2>
        <p className="secd">
          The punchline: nano_SLMs and SmolLM2-135M are the same design (Llama-family GQA decoder) — they differ in scale, vocabulary, data and training, not in how they work. Qwen3.8-Flash-Next is the next branch of the same family tree: identical skeleton, evolved organs.
        </p>
        <div className="two">
          <div className="mcard" style={{ overflowX: "auto" }}>
            <SpecTable specs={specs} />
          </div>
          <div>
            <div className="mcard" style={{ marginBottom: 22 }}>
              <div className="ptit"><span>Shared DNA</span></div>
              <SharedDna />
            </div>
            <div className="mcard">
              <div className="ptit"><span>Where they differ</span></div>
              <Differences />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
function hybOrDenseBadge(spec: Model): string {
  return spec.archKind === "hybridLinearMoe" ? hyb.headBadge(spec) : dense.headBadge(spec);
}
function specForSel(specs: Model[], sel: SelRef | null): Model {
  return specs.find((m) => m.id === (sel ? sel.model : "")) ?? specs[0];
}
function Compare({ specs }: { specs: Model[] }) {
  return (
    <div>
      <section className="block">
        <h2 className="sec">Side-by-side specs</h2>
        <p className="secd">Every cell below is computed or looked up from the model YAML configs — nothing hand-maintained.</p>
        <div className="mcard" style={{ overflowX: "auto" }}>
          <SpecTable specs={specs} />
        </div>
        <div className="mcard" style={{ marginTop: 22, overflowX: "auto" }}>
          <PurposeTable specs={specs} />
        </div>
      </section>
    </div>
  );
}
function About({ specs }: { specs: Model[] }) {
  return (
    <section className="block">
      <h2 className="sec">About · honesty ledger</h2>
      <p className="secd">Where every number visible in this app comes from — enforced by the registry, not goodwill.</p>
      <div className="two three">
        {specs.map((spec) => (
          <div className="mcard" key={"about-" + spec.id}>
            <h3 style={{ fontWeight: 600 }}>{spec.name}</h3>
            <ul className="srcs">
              {spec.honesty.sources.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
            <p style={{ fontSize: 13.5, color: "var(--dim)" }}>
              <b>Trained on:</b> {spec.honesty.trainedOn}
              <br />
              <b>Speaks:</b> {spec.honesty.speaks}
              <br />
              <b>Params computed:</b> {fmtP(pTotal(spec))}
            </p>
            <div className="note">{spec.honesty.simulatedNote}</div>
          </div>
        ))}
      </div>
      <footer>
        All activation values are SIMULATED (seeded engine, plan D5) — shapes, vocab, layers and expert counts are real and cited per model in the About tab. Depth-role bands are rule-of-thumb heuristics (Geva et al. 2021 arXiv:2104.08696; Meng et al. 2022 arXiv:2202.05262).
        Config-driven: add a <span className="mono">models/&lt;id&gt;.yaml</span> and the model appears everywhere. React + Vite build; the static single-file explorer remains the offline export of the same data.
      </footer>
    </section>
  );
}