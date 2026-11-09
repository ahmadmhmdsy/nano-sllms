# docs/plans — implementation and design plan index

Every plan/design document under docs/plans/, one row each. Statuses:
active (still governs work), completed (executed and closed by its
evidence), blocked (waiting on a user decision), historical (approved
baseline or handoff that no longer governs current work). Current repo
truth is always [CURRENT_STATE.md](../../CURRENT_STATE.md) plus
[AGENTS.md](../../AGENTS.md); plans never override the registries.

| path | scope | status | owner | last-verified | supersedes |
|---|---|---|---|---|---|
| docs/plans/2026-09-09-webui-u12-u13-model-tab.md | webui | completed | webui/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-10-model-viz-react-platform.md | viz | completed | viz/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-10_kt_ladder_handoff.md | core | historical | core | 2026-09-11 | none |
| docs/plans/2026-09-10_kt_ladder_plan.md | core | historical | core | 2026-09-11 | none |
| docs/plans/2026-09-10_kt_resume_handoff.md | core | historical | core | 2026-09-11 | none |
| docs/plans/2026-09-11-arabic-diacritization-specialist.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-11_mounting_design.md | core | historical | core | 2026-09-12 | docs/plans/2026-09-10_kt_ladder_plan.md |
| docs/plans/2026-09-11_mounting_impl_plan.md | core | historical | core | 2026-09-12 | docs/plans/2026-09-10_kt_ladder_handoff.md |
| docs/plans/2026-09-12-frozen-core-external-knn-memory-design.md | core | historical | core | UNKNOWN | none |
| docs/plans/2026-09-13-flow-editor-design.md | flow | completed | flow/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-13-flow-editor-mvp-plan.md | flow | completed | flow/AGENTS.md | 2026-09-13 | docs/plans/2026-09-13-flow-editor-design.md |
| docs/plans/2026-09-13-model-architecture-editor-design.md | viz | completed | viz/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-13-model-architecture-editor-plan.md | viz | completed | viz/AGENTS.md | 2026-09-13 | docs/plans/2026-09-13-model-architecture-editor-design.md |
| docs/plans/2026-09-13_diac_stage1_stage2_transfer_plan.md | diacritizer | historical | diacritizer | 2026-09-14 | none |
| docs/plans/2026-09-18-micro-expert-composition-plan.md | mex | completed | mex/AGENTS.md | 2026-09-19 | none |
| docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md | langid | completed | langid/AGENTS.md | 2026-09-19 | none |
| docs/plans/2026-09-19-mu1-composition-plan.md | mex | completed | mex/AGENTS.md | 2026-09-19 | docs/plans/2026-09-18-micro-expert-composition-plan.md |
| docs/plans/2026-09-19-mu2-growth-ladder-plan.md | mex | completed | mex/AGENTS.md | 2026-09-19 | docs/plans/2026-09-19-mu1-composition-plan.md |
| docs/plans/2026-09-21-dia-ladder-experts-plan.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-22 | docs/plans/2026-09-13_diac_stage1_stage2_transfer_plan.md |
| docs/plans/2026-09-22-agent-reflection-lessons.md | core | completed | core | 2026-09-22 | none |
| docs/plans/2026-09-22-e63-trial-bed-arms.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-23 | none |
| docs/plans/2026-09-22-shared-kv-per-layer-model.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-23 | none |
| docs/plans/2026-09-23-pipeline-feature-registry-plan.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-24 | none |
| docs/plans/2026-09-23-profile-store-reuse-sop.md | diacritizer | active | diacritizer/AGENTS.md | 2026-09-24 | none |
| docs/plans/2026-09-24-space-bunny-50k-data.md | ui | active | ui/AGENTS.md | 2026-09-24 | none |
| docs/plans/flow-registry-promotion.md | flow | completed | flow/AGENTS.md | 2026-09-13 | none |
| docs/plans/2026-09-28-remediation-plan.md | core | active | core | 2026-09-28 | none |
| docs/plans/2026-09-28-v2-fix-milestones.md | core | active | core | 2026-09-28 | none |

Notes (verification basis, recorded honestly):

- completed rows are backed by evidence artifacts owned by the line's
  scoped AGENTS.md (reports under research/, gate tables under
  research/gate_results/, ledgers under research/experiments.yaml).
- historical rows are approved baselines that were either completed and
  then extended by a superseding plan (the KT ladder was replaced by the
  mounting design), or never executed (the frozen-core/KNN-memory design
  stayed a design draft, so its last verified date is UNKNOWN).
- The two active rows govern ongoing work: the profile-store reuse SOP
  (diacritizer line) and the Space-Bunny 50k data build (ui line).
- New plans land here with an active row at creation time and move to
  completed/historical with a dated last-verified value when closed.
