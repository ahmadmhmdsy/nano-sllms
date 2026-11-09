// SVG diagram — direct port of the proven geometry in
// model_architecture_explorer.html renderDiagram() (plan P1: do NOT redesign).
import type { KeyboardEvent } from "react";
import { type ModelSpec } from "../schema/spec";
import { heat, mix } from "../engine/depth";
import { fmtP } from "../engine/params";
import * as dense from "./kinds/denseGqaMoe";
import * as hyb from "./kinds/hybridLinearMoe";

export interface SelRef {
  model: string;
  block: string;
  layer?: number;
}

interface Props {
  spec: ModelSpec;
  sel: SelRef | null;
  live: SelRef | null;
  onPick: (sel: SelRef) => void;
}

const SERIF = "Georgia, 'Iowan Old Style', 'Times New Roman', serif";
const MONO = "ui-monospace, Conservation, Consolas, monospace";
void MONO;

function kindOf(m: ModelSpec, l: number): "dense" | "linear" | "full" {
  if (m.archKind !== "hybridLinearMoe") return "dense";
  return hyb.isFullLayer(m, l) ? "full" : "linear";
}

export function Diagram({ spec: M, sel, live, onPick }: Props) {
  const Wd = 440, railX = 36, bx = 64, bw = 362;
  const midN = M.bands.mid[1] - M.bands.mid[0] + 1;
  const blockH = 230, bandH = 118;
  let y = 10; const tokY = y; y += 56; const embY = y; y += 64;
  const ys: Record<string, number> = {};
  ["b1", "b2", "band", "bN1", "bN"].forEach((st) => {
    ys[st] = y; y += st === "band" ? bandH : blockH; y += 26;
  });
  const fnY = y; y += 56; const headY = y; y += 66; const lgY = y; y += 56; const H = y + 16;

  const isSel = (block: string, layer?: number) =>
    !!sel && sel.model === M.id && sel.block === block && sel.layer === layer;
  const isLive = (block: string, layer?: number) =>
    !!live && live.model === M.id && live.block === block && live.layer === layer;
  const cls = (block: string, layer?: number) =>
    "blk" + (isSel(block, layer) ? " sel" : "") + (isLive(block, layer) ? " live" : "");
  const pick = (block: string, layer?: number) => () => onPick({ model: M.id, block, layer });

  /* Task M5.5 (a11y): keyboard activation for the interactive <g> groups.
   * They are focusable (tabIndex={0}) with role="button", but SVG elements
   * have no native key activation — Enter/Space must trigger the pick
   * explicitly (WCAG 2.1.1). */
  const activate = (fn: () => void) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fn();
    }
  };

  const hydra = M.archKind === "hybridLinearMoe";
  const toks = (M.facts?.tokens?.length ? M.facts.tokens : ["T0", "T1", "T2", "T3", "T4"]).slice(0, 5);

  function blockTitleText(idx: number, which: string): string {
    if (!hyb.isFullLayer(M, idx) && hydra) {
      const suf = which === "b1" ? " / " + M.shape.layers : which === "bN" ? " · last" : "";
      return "block " + idx + " · linear attention (GDN)" + suf;
    }
    if (hydra) {
      const suf = which === "b1" ? " / " + M.shape.layers : which === "bN" ? " · last" : "";
      return "block " + idx + " · gated attention (1 in 4)" + suf;
    }
    const suf = which === "b1" ? " / " + M.shape.layers : which === "bN" ? " last" : "";
    return "decoder block · layer " + idx + suf;
  }

  function DecoderBlock(idx: number, which: string, top: number) {
    return (
      <g key={which}>
        <rect x={bx} y={top} width={bw} height={blockH} rx={12} fill="rgba(255,255,255,.015)" stroke="#33333f" strokeWidth={1} />
        <g className={cls("layer", idx) + " blk"} onClick={pick("layer", idx)} onKeyDown={activate(pick("layer", idx))} tabIndex={0} role="button" data-model={M.id} data-block="layer" data-layer={idx}>
          <rect x={bx} y={top} width={bw} height={blockH} rx={12} fill="transparent" />
          <text x={bx + 12} y={top + 16} fontSize={12.5} fill="#9a97a3" style={{ fontFamily: SERIF }}>
            {blockTitleText(idx, which)}
          </text>
        </g>
        <line x1={railX} y1={top + 6} x2={railX} y2={top + 224} stroke="#7fc8a9" strokeWidth={2.2} opacity={0.8} />
        {[
          { yy: 24, ty: 37, line: 33 },
          { yy: 128, ty: 141, line: 137 },
        ].map((dd, ni) => (
          <g key={"ln" + ni}>
            <line x1={railX} y1={top + dd.line} x2={bx + 56} y2={top + dd.line} />
            <g className={cls("lnorm", idx) + " blk"} onClick={pick("lnorm", idx)} onKeyDown={activate(pick("lnorm", idx))} tabIndex={0} role="button" data-model={M.id} data-block="lnorm" data-layer={idx}>
              <rect x={bx + 58} y={top + dd.yy} width={110} height={18} rx={9} fill="rgba(138,143,152,.16)" stroke="#4a4a5a" />
              <text x={bx + 113} y={top + dd.ty} fontSize={9.5} fill="#b8bcc4" textAnchor="middle" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
                RMSNorm
              </text>
            </g>
          </g>
        ))}
        <line x1={bx + 113} y1={top + 42} x2={bx + 113} y2={top + 52} />
        <g className={cls("attn", idx) + " blk"} onClick={pick("attn", idx)} onKeyDown={activate(pick("attn", idx))} tabIndex={0} role="button" data-model={M.id} data-block="attn" data-layer={idx}>
          <rect x={bx + 58} y={top + 52} width={bw - 76} height={50} rx={7} fill="rgba(224,164,88,.10)" stroke="#e0a458" strokeWidth={1.3} />
          <text x={bx + 70} y={top + 73} fontSize={13.5} fill="#e9e7e2" style={{ fontFamily: SERIF }}>
            {kindOf(M, idx) === "linear" ? "Gated DeltaNet (linear attn)" : "Gated Attention · GQA"}
          </text>
          <text x={bx + 70} y={top + 92} fontSize={10} fill="#d8c49e" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
            {kindOf(M, idx) === "linear" ? hyb.attnBoxSub(M, false) : kindOf(M, idx) === "full" ? hyb.attnBoxSub(M, true) : dense.attnBoxSub(M)}
          </text>
          <text x={bx + bw - 70} y={top + 73} fontSize={10.5} fill="#9a97a3" textAnchor="end" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
            {kindOf(M, idx) === "linear" ? hyb.blockParamsAttn(M, false) : kindOf(M, idx) === "full" ? hyb.blockParamsAttn(M, true) : dense.blockParamsAttn(M)}
          </text>
        </g>
        <line x1={bx + 113} y1={top + 102} x2={bx + 113} y2={top + 116} />
        <line x1={bx + 113} y1={top + 116} x2={bx + 52} y2={top + 116} stroke="#e0a458" strokeWidth={1.4} />
        {/* add 1 */}
        <g className={cls("add", idx) + " blk"} onClick={pick("add", idx)} onKeyDown={activate(pick("add", idx))} tabIndex={0} role="button" data-model={M.id} data-block="add" data-layer={idx}>
          <circle cx={bx + 38} cy={top + 117} r={11} fill="rgba(127,200,169,.14)" stroke="#7fc8a9" strokeWidth={1.2} />
          <text x={bx + 38} y={top + 121} fontSize={13} textAnchor="middle" fill="#7fc8a9" style={{ fontFamily: SERIF }}>+</text>
        </g>
        <line x1={railX} y1={top + 116} x2={bx + 23} y2={top + 116} />
        <line x1={bx + 113} y1={top + 146} x2={bx + 113} y2={top + 156} />
        <g className={cls("ffn", idx) + " blk"} onClick={pick("ffn", idx)} onKeyDown={activate(pick("ffn", idx))} tabIndex={0} role="button" data-model={M.id} data-block="ffn" data-layer={idx}>
          <rect x={bx + 58} y={top + 156} width={bw - 76} height={50} rx={7} fill="rgba(167,139,220,.10)" stroke="#a78bdc" strokeWidth={1.3} />
          <text x={bx + 70} y={top + 177} fontSize={13.5} fill="#e9e7e2" style={{ fontFamily: SERIF }}>
            {hydra ? hyb.ffnBoxLabel(M) : dense.ffnBoxLabel(M)}
          </text>
          <text x={bx + 70} y={top + 196} fontSize={10} fill="#cbbde8" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
            {hydra ? hyb.ffnBoxSub(M) : dense.ffnBoxSub(M)}
          </text>
          <text x={bx + bw - 70} y={top + 177} fontSize={10.5} fill="#9a97a3" textAnchor="end" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
            {hydra ? hyb.blockParamsFfn(M) : dense.blockParamsFfn(M)}
          </text>
        </g>
        <line x1={bx + 113} y1={top + 206} x2={bx + 113} y2={top + 218} />
        <line x1={bx + 113} y1={top + 218} x2={bx + 52} y2={top + 218} stroke="#a78bdc" strokeWidth={1.4} />
        <g className={cls("add", idx) + " blk"} onClick={pick("add", idx)} onKeyDown={activate(pick("add", idx))} tabIndex={0} role="button" data-model={M.id} data-block="add" data-layer={idx}>
          <circle cx={bx + 38} cy={top + 219} r={11} fill="rgba(127,200,169,.14)" stroke="#7fc8a9" strokeWidth={1.2} />
          <text x={bx + 38} y={top + 223} fontSize={13} textAnchor="middle" fill="#7fc8a9" style={{ fontFamily: SERIF }}>+</text>
        </g>
        <line x1={railX} y1={top + 218} x2={bx + 23} y2={top + 218} />
      </g>
    );
  }

  return (
    <svg
      viewBox={"0 0 " + Wd + " " + H}
      style={{ display: "block", width: "100%", height: "auto" }}
      role="group"
      aria-label={M.name + " layer diagram"}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX={8} refY={5} markerWidth={8} markerHeight={8} orient="auto-start-reverse">
          <path d="M0,0L10,5L0,10z" fill="#5a5868" />
        </marker>
      </defs>
      {/* tokens */}
      {(() => {
        const tw = (bw - 4 * 8) / 5;
        return toks.map((t, i) => (
          <g key={"tk" + i} className={cls("tokens") + " blk"} onClick={pick("tokens")} onKeyDown={activate(pick("tokens"))} tabIndex={0} role="button" data-model={M.id} data-block="tokens">
            <rect x={bx + i * (tw + 8)} y={tokY + 1} width={tw} height={32} rx={16} fill="rgba(86,200,216,.10)" stroke="#4a4a5a" strokeWidth={1.3} />
            <text x={bx + i * (tw + 8) + tw / 2} y={tokY + 22} fontSize={10.5} textAnchor="middle" fill="#bfe6ec" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>{t}</text>
          </g>
        ));
      })()}
      <text x={Wd - 8} y={tokY + 22} fontSize={10} textAnchor="end" fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
        text → IDs
      </text>
      <Arrow x1={Wd / 2} y1={tokY + 36} y2={embY} />
      {/* embedding */}
      <g className={cls("embed") + " blk"} onClick={pick("embed")} onKeyDown={activate(pick("embed"))} tabIndex={0} role="button" data-model={M.id} data-block="embed">
        <rect x={bx} y={embY} width={bw} height={56} rx={9} fill="rgba(86,200,216,.08)" stroke="#56c8d8" strokeWidth={1.3} />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <rect key={"ec" + i} x={bx + 14 + i * 26} y={embY + 10} width={16} height={36} rx={2} fill={i % 2 ? "rgba(86,200,216,.25)" : "rgba(86,200,216,.12)"} />
        ))}
        <text x={bx + bw - 14} y={embY + 25} fontSize={14.5} textAnchor="end" fill="#e9e7e2" style={{ fontFamily: SERIF }}>
          Embedding table
        </text>
        <text x={bx + bw - 14} y={embY + 43} fontSize={10.5} textAnchor="end" fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
          {fmtP(M.shape.vocab)} × {M.shape.d} · one learned vector per token
        </text>
      </g>
      <Arrow x1={Wd / 2} y1={embY + 58} y2={ys.b1} />
      {DecoderBlock(1, "b1", ys.b1)}
      <Arrow x1={Wd / 2} y1={ys.b1 + blockH} y2={ys.b2} />
      {DecoderBlock(2, "b2", ys.b2)}
      <Arrow x1={Wd / 2} y1={ys.b2 + blockH} y2={ys.band} />
      {/* middle band */}
      <g className={cls("band") + " blk"} onClick={pick("band")} onKeyDown={activate(pick("band"))} tabIndex={0} role="button" data-model={M.id} data-block="band">
        <rect x={bx} y={ys.band} width={bw} height={bandH} rx={12} fill="rgba(167,139,220,.05)" stroke="#a78bdc" strokeWidth={1.2} strokeDasharray="5 4" />
        <text x={bx + 14} y={ys.band + 30} fontSize={12.5} fill="#cbbde8" style={{ fontFamily: SERIF }}>layers {M.bands.mid[0]}–{M.bands.mid[1]}</text>
        <text x={bx + 14} y={ys.band + 48} fontSize={12.5} fill="#cbbde8" style={{ fontFamily: SERIF }}>the knowledge</text>
        <text x={bx + 14} y={ys.band + 63} fontSize={12.5} fill="#cbbde8" style={{ fontFamily: SERIF }}>hotspot</text>
        <text x={bx + 14} y={ys.band + bandH - 14} fontSize={9.5} fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>one bar = one layer</text>
        {Array.from({ length: midN }, (_, k) => {
          const i = M.bands.mid[0] + k;
          const t = heat(M.bands, i);
          const ty2 = ys.band + 16 + ((k + 0.5) / midN) * 86;
          return (
            <g key={"hb" + i} className="tick" data-model={M.id} data-block="layer" data-layer={i} onClick={() => onPick({ model: M.id, block: "layer", layer: i })} onKeyDown={activate(() => onPick({ model: M.id, block: "layer", layer: i }))} tabIndex={0} role="button" aria-label={"layer " + i + " knowledge heat"}>
              <line x1={bx + 96} y1={ty2} x2={bx + 96 + (bw - 110) * t} y2={ty2} stroke={mix("#a78bdc", t)} strokeWidth={4} strokeLinecap="round" />
            </g>
          );
        })}
      </g>
      <Arrow x1={Wd / 2} y1={ys.band + bandH} y2={ys.bN1} />
      {DecoderBlock(M.shape.layers - 1, "bN1", ys.bN1)}
      <Arrow x1={Wd / 2} y1={ys.bN1 + blockH} y2={ys.bN} />
      {DecoderBlock(M.shape.layers, "bN", ys.bN)}
      <Arrow x1={Wd / 2} y1={ys.bN + blockH} y2={fnY} />
      {/* final norm */}
      <g className={cls("finalnorm") + " blk"} onClick={pick("finalnorm")} onKeyDown={activate(pick("finalnorm"))} tabIndex={0} role="button" data-model={M.id} data-block="finalnorm">
        <rect x={bx} y={fnY} width={bw} height={56} rx={9} fill="rgba(138,143,152,.10)" stroke="#8a8f98" strokeWidth={1.3} />
        <text x={bx + 16} y={fnY + 25} fontSize={14} fill="#e9e7e2" style={{ fontFamily: SERIF }}>Final RMSNorm</text>
        <text x={bx + 16} y={fnY + 43} fontSize={10.5} fill="#9a97a3" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>one last rescale before reading out</text>
      </g>
      <Arrow x1={Wd / 2} y1={fnY + 58} y2={headY} />
      {/* lm head */}
      <g className={cls("lmhead") + " blk"} onClick={pick("lmhead")} onKeyDown={activate(pick("lmhead"))} tabIndex={0} role="button" data-model={M.id} data-block="lmhead">
        <rect x={bx} y={headY} width={bw} height={60} rx={9} fill="rgba(232,99,90,.08)" stroke="#e8635a" strokeWidth={1.3} />
        <text x={bx + 16} y={headY + 26} fontSize={13.5} fill="#e9e7e2" style={{ fontFamily: SERIF }}>
          {hydra ? hyb.headTitle(M) : dense.headTitle(M)}
        </text>
        <text x={bx + 16} y={headY + 45} fontSize={10.5} fill="#e0a39d" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
          {hydra ? hyb.headSub(M) : dense.headSub(M)}
        </text>
      </g>
      <Arrow x1={Wd / 2} y1={headY + 62} y2={lgY} />
      {/* logits */}
      {[0, 1, 2, 3, 4].map((i) => {
        const lw = (bw - 40) / 5;
        const lx = bx + i * (lw + 10) - i;
        return (
          <g key={"lg" + i} className={cls("logits") + " blk"} onClick={pick("logits")} onKeyDown={activate(pick("logits"))} tabIndex={0} role="button" data-model={M.id} data-block="logits">
            <rect x={lx} y={lgY + 1} width={lw} height={32} rx={8} fill={i === 0 ? "rgba(232,99,90,.16)" : "rgba(232,99,90,.05)"} stroke="#4a4a5a" strokeWidth={1.3} />
            <text x={lx + lw / 2} y={lgY + 22} fontSize={12} textAnchor="middle" fill={i === 0 ? "#e8635a" : "#6a6776"}>
              {i === 0 ? "★" : "·"}
            </text>
          </g>
        );
      })}
      <text x={Wd - 8} y={lgY + 22} fontSize={10} textAnchor="end" fill="#6a6776" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
        softmax
      </text>
      <text x={bx} y={lgY + 52} fontSize={11.5} fill="#e8635a" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>
        top-1: {(M.facts?.top1 && M.facts.top1) || "—"}
      </text>
    </svg>
  );
}

function Arrow({ x1, y1, y2 }: { x1: number; y1: number; y2: number }) {
  return <line x1={x1} y1={y1} x2={x1} y2={y2 - 3} stroke="#5a5868" strokeWidth={1.4} markerEnd="url(#arr)" />;
}