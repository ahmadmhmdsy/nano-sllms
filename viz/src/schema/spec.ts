// ModelSpec — the one contract every *.yaml in ../models must satisfy (plan §4).
// Hand-typed interface (TS owns types); the zod schema mirrors it as the runtime
// validator used by the registry + doctor.
import { z } from "zod";

export type BandTuple = [number, number];
export type DemoPredEntry = [string, number];

export interface Shape {
  layers: number;
  d: number;
  heads: number;
  kv: number;
  ffn: number; // dense FFN width OR MoE expert inter width
  vocab: number;
  ctx: string;
  theta: string;
  eps: string;
  tied: boolean;
  precision: string;
  attentionImpl?: string;
}

export interface HybridCfg {
  fullEvery?: number;
  layerPattern?: string;
  linearMixer?: { kHeads: number; vHeads: number; qHeads: number; headDim: number; conv: number };
  fullMixer?: { partialRotary?: string; indexer?: string; headDim?: number };
  experts?: { total: number; active: number; shared: number; inter: number };
  extras?: { label: string; params: number };
}

export interface Facts {
  tokens?: string[];
  top1?: string;
  tokensNote?: string;
  blurb?: string;
  lnorm?: string;
  add?: string;
  addTech?: string;
  embed?: string;
  embedTech?: string;
  mixer?: string;
  mixerTech?: string;
  ffn?: string;
  ffnTech?: string;
  lmhead?: string;
  lmheadTech?: string;
  band?: string;
}

export interface DemoCfg {
  entropyBias?: number;
  moePeaks?: boolean;
  preds?: Record<string, DemoPredEntry[]>;
}

export interface Honesty {
  sources: string[];
  trainedOn: string;
  speaks: string;
  simulatedNote: string;
}

export interface Bands {
  early: BandTuple;
  mid: BandTuple;
  late: BandTuple;
}

export interface ModelSpec {
  id: string;
  name: string;
  fullName: string;
  accent: string;
  archKind: "denseGqaMoe" | "hybridLinearMoe";
  blurb: string;
  tok: string;
  data: string;
  domain: string;
  shape: Shape;
  hybrid?: HybridCfg;
  bands: Bands;
  facts?: Facts;
  demo?: DemoCfg;
  honesty: Honesty;
  overrides?: string;
}

// zod runtime mirror — types come from the interfaces above
export const modelSpecSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  fullName: z.string().min(1),
  accent: z.string().regex(/^#[0-9a-fA-F]{6}$/),
  archKind: z.enum(["denseGqaMoe", "hybridLinearMoe"]),
  blurb: z.string().min(1),
  tok: z.string(),
  data: z.string(),
  domain: z.string(),
  shape: z.object({
    layers: z.number().int().positive(),
    d: z.number().int().positive(),
    heads: z.number().int().positive(),
    kv: z.number().int().positive(),
    ffn: z.number().int().positive(),
    vocab: z.number().int().positive(),
    ctx: z.string(),
    theta: z.string(),
    eps: z.string(),
    tied: z.boolean(),
    precision: z.string(),
    attentionImpl: z.string().optional(),
  }),
  hybrid: z
    .object({
      fullEvery: z.number().int().min(2).optional(),
      layerPattern: z.string().optional(),
      linearMixer: z
        .object({
          kHeads: z.number().int().positive(),
          vHeads: z.number().int().positive(),
          qHeads: z.number().int().positive(),
          headDim: z.number().int().positive(),
          conv: z.number().int().positive(),
        })
        .optional(),
      fullMixer: z
        .object({
          partialRotary: z.string().optional(),
          indexer: z.string().optional(),
          headDim: z.number().int().positive().optional(),
        })
        .optional(),
      experts: z
        .object({
          total: z.number().int().positive(),
          active: z.number().int().positive(),
          shared: z.number().int(),
          inter: z.number().int().positive(),
        })
        .optional(),
      extras: z
        .object({
          label: z.string(),
          params: z.number(),
        })
        .optional(),
    })
    .optional(),
  bands: z.object({
    early: z.tuple([z.number().int().min(1), z.number().int().min(1)]),
    mid: z.tuple([z.number().int().min(1), z.number().int().min(1)]),
    late: z.tuple([z.number().int().min(1), z.number().int().min(1)]),
  }),
  facts: z
    .object({
      tokens: z.array(z.string()).optional(),
      top1: z.string().optional(),
      tokensNote: z.string().optional(),
      lnorm: z.string().optional(),
      add: z.string().optional(),
      addTech: z.string().optional(),
      embed: z.string().optional(),
      embedTech: z.string().optional(),
      mixer: z.string().optional(),
      mixerTech: z.string().optional(),
      ffn: z.string().optional(),
      ffnTech: z.string().optional(),
      lmhead: z.string().optional(),
      lmheadTech: z.string().optional(),
      band: z.string().optional(),
    })
    .optional(),
  demo: z
    .object({
      entropyBias: z.number().optional(),
      moePeaks: z.boolean().optional(),
      preds: z
        .record(z.string(), z.array(z.tuple([z.string(), z.number().min(0).max(100)])))
        .optional(),
    })
    .optional(),
  honesty: z.object({
    sources: z.array(z.string()).min(1),
    trainedOn: z.string(),
    speaks: z.string(),
    simulatedNote: z.string(),
  }),
  overrides: z.string().optional(),
});
