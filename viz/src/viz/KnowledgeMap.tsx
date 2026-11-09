// Knowledge map card — depth-role zones + param-share bar (computed from engine).
import type { ModelSpec } from "../schema/spec";
import { fmtP, fmtB, pTotal } from "../engine/params";
import * as dense from "./kinds/denseGqaMoe";
import * as hyb from "./kinds/hybridLinearMoe";

export function KnowledgeMap({ spec }: { spec: ModelSpec }) {
  const M = spec;
  const L = M.shape.layers;
  const total = pTotal(M);
  const wEarly = ((M.bands.early[1] - M.bands.early[0] + 1) / L) * 100;
  const wMid = ((M.bands.mid[1] - M.bands.mid[0] + 1) / L) * 100;
  const wLate = ((M.bands.late[1] - M.bands.late[0] + 1) / L) * 100;
  const shares = M.archKind === "hybridLinearMoe" ? hyb.shareRows(M) : dense.shareRows(M);
  const pct = (n: number) => (n / total) * 100;
  return (
    <div>
      <div className="ptit">
        <span>{M.name} — depth roles</span>
        <span>{L} layers</span>
      </div>
      <div className="zones">
        <div className="zone" style={{ width: wEarly.toFixed(1) + "%", background: "rgba(86,200,216,.22)" }}>
          <b style={{ fontSize: 11 }}>INPUT · SYNTAX</b>
          <small>layers {M.bands.early[0]}–{M.bands.early[1]}</small>
        </div>
        <div className="zone" style={{ width: wMid.toFixed(1) + "%", background: "rgba(167,139,220,.30)" }}>
          <b style={{ fontSize: 11 }}>KNOWLEDGE BASE</b>
          <small>layers {M.bands.mid[0]}–{M.bands.mid[1]} · FFN peak</small>
        </div>
        <div className="zone" style={{ width: wLate.toFixed(1) + "%", background: "rgba(224,164,88,.22)" }}>
          <b style={{ fontSize: 11 }}>REASONING · PREDICT</b>
          <small>layers {M.bands.late[0]}–{M.bands.late[1]}</small>
        </div>
      </div>
      <div className="ptit">
        <span>Where the parameters (≈ learned material) sit</span>
        <span>{M.archKind === "hybridLinearMoe" ? "≈" + fmtB(total) + " (text stack)" : fmtP(total) + " params"}</span>
      </div>
      <div className="pshare">
        {shares.map(([label, p, color]) => (
          <div key={label} style={{ background: color, width: pct(p).toFixed(1) + "%" }} title={label + ": " + fmtP(p)}>
            {pct(p) > 9 ? pct(p).toFixed(0) + "%" : ""}
          </div>
        ))}
      </div>
      <div style={{ font: "11.5px var(--mono)", color: "var(--faint)" }}>
        {shares.map(([label, p, color], i) => (
          <span key={label}>
            {i > 0 && <span style={{ color: "var(--faint)" }}> · </span>}
            <span style={{ color }}>{pct(p).toFixed(1)}%</span>
          </span>
        ))}
      </div>
      <p style={{ fontSize: 13.5, color: "var(--dim)", margin: "12px 0 0" }}>
        Reading it: the <b style={{ color: "var(--ffn)" }}>middle-band FFNs</b> are where learned knowledge is exercised hardest — and where most parameters live. The <b style={{ color: "var(--embed)" }}>embedding</b> holds single-token meaning, <b style={{ color: "var(--attn)" }}>attention</b> moves information between tokens, and the <b style={{ color: "var(--attn)" }}>late band</b> turns accumulated features into a prediction.{" "}
        {M.archKind === "hybridLinearMoe" ? (
          <>In this model the recipe EVOLVED: attention has largely left the parameters — the mixers hold only ~1.6% of the weights — while <b style={{ color: "var(--ffn)" }}>≈97%</b> lives in {M.shape.layers * (M.hybrid?.experts?.total ?? 0)} routed expert key–value memories ({M.hybrid?.experts?.total} experts × {M.shape.layers} layers); the router fires just {M.hybrid?.experts?.active} + 1 shared per token. MoE = many small knowledge bases; the router picks the team per token.</>
        ) : (
          <>Same profile as any dense model, different proportions: SmolLM2 spends a bigger share on its 49k vocabulary (21% vs 15%), nano_SLMs on its wider FFNs (67% vs 59%).</>
        )}
      </p>
    </div>
  );
}
