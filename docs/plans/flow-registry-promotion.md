# Plan — flow/ registry promotion (2026-09-13f)

User-approved from the MVP-retro next-phase menu. Promote each built-in node
from dict-entry to behavior-carrying definition; de-hardcode node knowledge;
keep the MVP linear-chain contract EXACTLY (shape-preserving refactor — no
feature changes). Plugin-discovery skeleton explicitly DEFERRED.

## Goals (user order of the day)
1. Built-ins become node definitions with behavior (validate_semantic,
   to_config, required_upstream declared per node).
2. No hardcoded node knowledge: CHAIN list, _DEPENDENCIES, PRESETS, knob
   checks, frontend FALLBACK_REGISTRY, kind==="infer"/"dataset" checks — all
   become registry data/methods.
3. Atomic flow save (temp + os.replace) folded in.

## Tasks

- T0. Golden-diff safety net: commit golden build_config outputs for the
  5 committed example/demo flows (flows/*.flow.json) under
  flow/tests/goldens/ + a unit test that build_config output for each is
  byte-equal to the golden (proves refactor, not behavior change).
- T1. Backend package flow/server/nodes/ (replaces nodes.py): base.py
  (NodeDef: kind, title, category, ports, PropSpec, label_semantic,
  features, gate, required_upstream), ports.py (compat rule named once),
  builtin/{dataset,prepare,tokenize,train,eval_job,infer}.py each owning its
  constants (PRESETS move into train.py), __init__.py registry API
  (register/get/all/NODES/PORTS/PROPS... compat surface re-exported).
  flow/server/nodes.py becomes the package dir; graph_schema/flows/app
  import surface unchanged.
- T2. config_gen de-hardcoding: chain derived from registered
  required_upstream + port graph (no CHAIN list, no _DEPENDENCIES);
  per-node validate_semantic replaces knob checks; PRESETS/SHARD_TOKENS/
  MAX_ROWS live in owning node modules. Error messages preserved.
- T3. Frontend: FALLBACK_REGISTRY -> src/nodes/registry.ts (fallback only,
  server snapshot authoritative); portTypes.ts compat rule; snapshot gains
  label_semantic + features; PipelineNode infer button + Inspector dataset
  placeholder driven by registry data (zero kind-conditionals in UI code).
- T4. flows.py atomic save (validate -> temp write -> fsync -> os.replace).
- T5. Verification: 114 server tests + new golden tests, vitest 85+new,
  pnpm build, browser drill (3 example flows validate + open modal closes),
  docs (flow/README architecture section, MEMORY lesson, AGENTS command
  table unchanged, ledger + HANDOFF).

## Out of scope
- F5 branching execution (interface stays stub), F2 model editor, plugin
  discovery directory, store.ts god-file split.
