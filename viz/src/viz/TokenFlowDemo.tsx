// Simulated token-flow demo (plan D5). One seeded engine (engine/sim.ts),
// clear SIMULATED badges, per-model authored preds from YAML or neutral fallback.
import { useEffect, useRef, useState } from "react";
import type { ModelSpec } from "../schema/spec";
import { attRow, ffnSlots, residCurve, fallbackPreds } from "../engine/sim";
import { bandOf, heat } from "../engine/depth";
import { fmtP } from "../engine/params";
import { DEMO_PROMPTS } from "../registry/demoPrompts";

interface Step { s: string; l?: number }

function buildSteps(spec: ModelSpec): Step[] {
  const steps: Step[] = [{ s: "tokens" }, { s: "embed" }];
  for (let l = 1; l <= spec.shape.layers; l++) {
    steps.push({ s: "lnorm", l }, { s: "attn", l }, { s: "add", l }, { s: "lnorm2", l }, { s: "ffn", l }, { s: "add2", l });
  }
  steps.push({ s: "finalnorm" }, { s: "head" }, { s: "softmax" });
  return steps;
}

function stageLabel(spec: ModelSpec, st: Step): [string, string] {
  const M = spec.shape;
  const hybrid = spec.archKind === "hybridLinearMoe";
  const b = st.l ? bandOf(spec.bands, st.l) : "";
  const full = hybrid ? st.l! % (spec.hybrid?.fullEvery ?? 4) === 0 : false;
  switch (st.s) {
    case "tokens":
      return ["Tokenize", "Text → token IDs via the " + spec.tok + ". (Ġ = space, Ċ = newline.)"];
    case "embed":
      return ["Embed", "Each ID picks its row from the " + fmtP(M.vocab) + " × " + M.d + " table → a " + M.d + "-number meaning vector per token."];
    case "lnorm":
      return ["Layer " + st.l + " · pre-norm", "RMSNorm rescales the stream before attention reads it."];
    case "lnorm2":
      return ["Layer " + st.l + " · pre-norm (2nd)", "RMSNorm again before the FFN reads it."];
    case "attn":
      if (hybrid) {
        return full
          ? ["Layer " + st.l + " · gated attention (full, 1 in 4)", "The exact mixer: GQA " + M.heads + "Q/" + M.kv + "KV, head " + (spec.hybrid?.fullMixer?.headDim ?? 256) + ", partial RoPE — the only layers with a true token-to-token attention grid (shown opposite)."]
          : ["Layer " + st.l + " · GDN linear attention (state mixer)", "Compresses the past into a fixed-size state (" + spec.hybrid?.linearMixer?.vHeads + " value heads × " + spec.hybrid?.linearMixer?.headDim + ") that every token reads and writes — constant memory per token; how long context stays affordable. No attention grid exists here."];
      }
      return ["Layer " + st.l + " · GQA attention (" + b + " band)",
        b === "mid" ? "Middle-depth attention: routing is now semantic — linking related concepts, not just neighbors."
        : b === "early" ? "Early attention: broad, mostly local — spelling and short-range structure."
        : "Late attention: focused gathering of everything the stack produced."];
    case "ffn":
      if (hybrid)
        return ["Layer " + st.l + " · sparse MoE (" + b + " band)", "A router activates " + spec.hybrid?.experts?.active + " of " + spec.hybrid?.experts?.total + " expert key–value memories (+1 shared expert) and injects their combined update. In MoE models nearly ALL learned knowledge lives in these experts."];
      return ["Layer " + st.l + " · SwiGLU FFN (" + b + " band)",
        b === "mid" ? "Peak knowledge traffic: key–value memories fire and inject factual / semantic updates into the stream."
        : b === "early" ? "The FFN sharpens token-level patterns; little world knowledge is involved yet."
        : "The FFN converts accumulated features toward prediction-shaped ones."];
    case "add":
      return ["Layer " + st.l + " · residual add", "The attention update is ADDED onto the highway — the token keeps its full history."];
    case "add2":
      return ["Layer " + st.l + " · residual add", "The FFN update is added; the stream's norm grows as depth accumulates edits."];
    case "finalnorm":
      return ["Final RMSNorm", "One last rescale of the newest token's vector."];
    case "head":
      return hybrid
        ? ["LM head (untied)", "The newest vector is dotted against a SEPARATE " + M.d + " → " + fmtP(M.vocab) + " output matrix — scoring every possible next token."]
        : ["LM head (tied)", "Dot product against all " + fmtP(M.vocab) + " embedding rows — the same matrix that read the tokens in."];
    case "softmax":
      return ["Softmax → next token", "Top-5 probabilities below (SIMULATED values). The chosen token is appended and the whole loop repeats."];
    default:
      return ["", ""];
  }
}

function AttGrid({ spec, layer, promptIdx }: { spec: ModelSpec; layer: number | null; promptIdx: number }) {
  const p = DEMO_PROMPTS[promptIdx];
  if (layer == null) return null;
  const hybrid = spec.archKind === "hybridLinearMoe";
  const fullEvery = spec.hybrid?.fullEvery ?? 4;
  if (hybrid && layer % fullEvery !== 0) {
    return (
      <svg viewBox="0 0 320 132" width="100%" height="132">
        <text x={14} y={36} fontSize={11.5} fill="#e0a458" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>linear attention — no attention grid</text>
        <text x={14} y={58} fontSize={10.5} fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>GDN keeps a fixed-size recurrent state instead of</text>
        <text x={14} y={74} fontSize={10.5} fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>scoring every past token — O(1) memory per token,</text>
        <text x={14} y={90} fontSize={10.5} fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>{spec.hybrid?.linearMixer?.vHeads} value heads × {spec.hybrid?.linearMixer?.headDim} at layer {layer}.</text>
        <text x={14} y={106} fontSize={10.5} fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>Full attention returns every {fullEvery}th layer.</text>
      </svg>
    );
  }
  const n = p.toks.length;
  const raw = attRow(spec.shape.layers, n, layer, promptIdx + spec.shape.layers, spec.demo?.entropyBias ?? 0);
  const cell = Math.min(30, Math.floor(240 / n)), ox = 46, oy = 20;
  const q = n - 1;
  return (
    <svg viewBox="0 0 320 132" width="100%" height="132">
      {p.toks.map((t, j) => (
        <text key={j} x={ox + j * cell + cell / 2} y={14} textAnchor="middle" fontSize={8.5} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>{t.slice(0, 6)}</text>
      ))}
      {p.toks.map((_, j) => {
        if (j > q) return <rect key={"c" + j} x={ox + j * cell} y={oy} width={cell - 2} height={cell - 2} rx={2} fill="rgba(255,255,255,.02)" />;
        const v = raw[j].v;
        return (
          <g key={"cc" + j}>
            <rect x={ox + j * cell} y={oy} width={cell - 2} height={cell - 2} rx={2} fill={"rgba(224,164,88," + (0.08 + v * 0.85).toFixed(3) + ")"} />
            <text x={ox + j * cell + cell / 2 - 1} y={oy + cell + 10} textAnchor="middle" fontSize={8} fill={v > 0.25 ? "#e0a458" : "#54525e"} style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>{v.toFixed(2)}</text>
          </g>
        );
      })}
      <text x={6} y={oy + cell / 2 + 3} fontSize={8.5} fill="#e0a458" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>L{layer}</text>
      <text x={6} y={oy + cell + 20} fontSize={8} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>causal: q='{p.toks[q]}' sees only j ≤ q</text>
    </svg>
  );
}

function FfnBars({ spec, layer }: { spec: ModelSpec; layer: number | null }) {
  if (layer == null) return <svg viewBox="0 0 320 76" width="100%" height="76" />;
  const moe = spec.archKind === "hybridLinearMoe";
  const slots = ffnSlots(spec.shape.layers, layer, heat(spec.bands, layer), 1, moe && !!spec.demo?.moePeaks);
  return (
    <svg viewBox="0 0 320 76" width="100%" height="76">
      {slots.map((v, i) => (
        <rect key={i} x={0} y={6 + i * 8} width={290 * v} height={5} rx={2.5} fill={"rgba(167,139,220," + (0.25 + v * 0.65).toFixed(3) + ")"} />
      ))}
      <text x={296} y={40} fontSize={9} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>slot</text>
      <text x={296} y={52} fontSize={9} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>activity</text>
    </svg>
  );
}

function Spark({ spec, upToLayer }: { spec: ModelSpec; upToLayer: number }) {
  const pts = residCurve(spec.shape.layers);
  return (
    <svg viewBox="0 0 300 54" width="100%" height={54} preserveAspectRatio="none">
      <line x1={10} y1={50} x2={290} y2={50} stroke="#2a2a36" strokeWidth={1} />
      <text x={10} y={10} fontSize={9} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>depth → (SIMULATED)</text>
      {upToLayer > 1 && (
        <polyline
          points={pts.slice(0, upToLayer).map((p) => 10 + p.x * 280 + "," + (46 - p.y * 36)).join(" ")}
          fill="none" stroke="#7fc8a9" strokeWidth={2} strokeLinecap="round" />
      )}
      {upToLayer > 0 && (
        <circle cx={10 + pts[upToLayer - 1].x * 280} cy={46 - pts[upToLayer - 1].y * 36} r={3.5} fill="#7fc8a9" />
      )}
    </svg>
  );
}

export function tokenGlyph(tok: string): string {
  return tok.replace(/Ġ/g, "␣").replace(/Ċ/g, "⏎");
}

export function Top5({ spec, promptIdx, reached }: { spec: ModelSpec; promptIdx: number; reached: boolean }) {
  const authored: [string, number][] | undefined = spec.demo?.preds?.[DEMO_PROMPTS[promptIdx].id];
  const list: [string, number][] = authored ?? fallbackPreds(DEMO_PROMPTS[promptIdx].toks[0]);
  if (!reached) {
    return <div><span style={{ color: "var(--faint)", font: "12px var(--mono)" }}>— appears after the final layer —</span></div>;
  }
  const isFallback = !authored;
  return (
    <div className="pbar">
      {isFallback && (
        <div style={{ font: "10.5px var(--mono)", color: "#d8a24a", border: "1px dashed #6b5426", borderRadius: 6, padding: "4px 9px", marginBottom: 6 }}>
          GENERIC FALLBACK — this model ships no authored predictions; everything here is SIMULATED
        </div>
      )}
      {list.map(([tok, pct]) => (
        <div className="prow" key={tok}>
          <span className="lab">{tokenGlyph(tok)}</span>
          <span className="track"><i className="fill" data-w={pct} /></span>
          <span className="pct">{pct}%</span>
        </div>
      ))}
    </div>
  );
}

export function TokenFlowDemo({ spec, onLiveLayer }: { spec: ModelSpec; onLiveLayer?: (l: number | undefined) => void }) {
  const steps = buildSteps(spec);
  const [stepNum, setStepNum] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [promptIdx, setPromptIdx] = useState(0);
  const [speed, setSpeed] = useState(350);
  const timer = useRef<number | null>(null);
  const playingRef = useRef(playing);
  playingRef.current = playing;

  const st: Step = steps[Math.min(stepNum, steps.length - 1)];
  const [title, txt] = stageLabel(spec, st);
  const liveAttnLayer = st.s === "attn" ? st.l ?? null : null;
  const liveFfn = st.s === "ffn" ? st.l ?? null : null;
  const curLayer = st.l ?? 0;
  // notify the parent which layer is live so the Diagram can highlight it
  useEffect(() => {
    onLiveLayer?.(st.l);
  }, [st.l]);

  useEffect(() => {
    if (!playing) return;
    const t = window.setTimeout(() => {
      setStepNum((n) => n + 1 >= steps.length ? n : n + 1);
    }, speed);
    timer.current = t;
    return () => clearTimeout(t);
  }, [playing, stepNum, speed, steps.length]);
  useEffect(() => {
    if (stepNum + 1 >= steps.length) setPlaying(false);
  }, [stepNum, steps.length]);

  return (
    <div className="demo">
      <div className="ptit"><span>Run on</span><span className="chip on" style={{ borderColor: spec.accent }}>{spec.name}</span></div>
      <div className="ptit"><span>Prompt</span>
        <span className="chips">
          {DEMO_PROMPTS.map((p, i) => (
            <button key={p.id} className={"chip" + (i === promptIdx ? " on" : "")} onClick={() => { setPromptIdx(i); setStepNum(0); }}>
              {p.label}
            </button>
          ))}
        </span>
      </div>
      <div className="controls">
        <button data-testid="btn-play" onClick={() => { if (stepNum + 1 >= steps.length) setStepNum(0); setPlaying((v) => !v); }}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <button data-testid="btn-next" onClick={() => {
          setPlaying(false);
          let i = stepNum;
          while (true) {
            i++;
            if (i >= steps.length) { i = steps.length - 1; break; }
            if (steps[i].s === "attn") break;
          }
          setStepNum(i);
        }}>Next layer ⏭</button>
        <button onClick={() => { setPlaying(false); setStepNum(0); }}>↺ Reset</button>
        <select value={String(speed)} onChange={(e) => setSpeed(Number(e.target.value))}>
          <option value="700">1× slow</option>
          <option value="350">2× normal</option>
          <option value="160">4× fast</option>
        </select>
        <span className="simtag">SIMULATED ACTIVATIONS — NOT REAL INFERENCE</span>
      </div>
      <div className="demo-grid">
        <div className="tracebox">
          <h4>Trace</h4>
          <p data-testid="trace-step" style={{ font: "600 17px var(--serif)", margin: "0 0 6px" }}>{title}</p>
          <p style={{ color: "var(--dim)", fontSize: 14, margin: "0 0 10px" }}>{txt}{st.s === "softmax" ? "  ·  " + DEMO_PROMPTS[promptIdx].note : ""}</p>
          <div className="minis">
            <div className="mini">
              <h5>Residual stream ‖x‖ vs depth</h5>
              <Spark spec={spec} upToLayer={curLayer} />
            </div>
            <div className="mini">
              <h5>Top-5 next token</h5>
              <Top5 spec={spec} promptIdx={promptIdx} reached={st.s === "softmax"} />
            </div>
          </div>
        </div>
        <div className="tracebox">
          <h4>At this layer · attention pattern (query = newest token)</h4>
          <AttGrid spec={spec} layer={liveAttn(st)} promptIdx={promptIdx} />
          <h4 style={{ marginTop: 12 }}>FFN memory slots firing (8 of {spec.hybrid?.experts?.total ? fmtP(spec.hybrid.experts.total) + "+" : "8"} shown of thousands)</h4>
          <FfnBars spec={spec} layer={liveFfn ? st.l! : null} />
        </div>
      </div>
    </div>
  );
}
function liveAttn(st: Step): number | null {
  return st.s === "attn" ? st.l ?? null : null;
}
