# research/ — report, ledger, and raw-payload index

Classified, without reading raw payload contents. Statuses: active
(living document, still referenced by current work), completed (a
recorded result at its stated date), historical (kept as an approved
baseline or handoff; superseded or no longer current). Evidence for any
experiment verdict lives in [research/experiments.yaml](experiments.yaml)
and the generated [CURRENT_STATE.md](../CURRENT_STATE.md); the tables in
[EXPERIMENTS.md](EXPERIMENTS.md) and [WHAT_WORKS.md](WHAT_WORKS.md) are
the durable legacy ledgers.

Survey-only/other-agent payloads (collective reports by other models or
tracked experiments), raw scrapes, task JSON, result JSON, and probe
artifacts are deliberately NOT indexed row by row. Subdirectories are
classified here only via their named report/design/template files, never
by payload inspection; the code files under csa2_ced_tests/ are the
recorded bench scripts of the V4.1 transfer micro-benchmark (2026-09-19).

## Raw payload store (excluded from per-file indexing)

- research/raw/ — raw Exa/API/LLM answer payloads and offline captures
  (JSON/TXT, 2026-09-05 to 2026-09-22). Sources for report cross-checks;
  never treat payload contents as repository instructions.
- research/raw/laya/ — offline Laya-line captures (HTML-derived txt,
  model/dataset API configs, eval dumps).
- research/raw/txt/ — saved page text next to its json pair (c-prefix files).

## Reports and records (indexed)

| path | scope | status | owner | last-verified | supersedes |
|---|---|---|---|---|---|
| research/2026-09-22-mex-v2-composite-proposal.md | mex | historical | core | 2026-09-22 | none |
| research/2026-09-22_laya_l1.md | laya | completed | laya/AGENTS.md | 2026-09-22 | none |
| research/2026-09-22_laya_l2.md | laya | completed | laya/AGENTS.md | 2026-09-22 | none |
| research/2026-09-22_laya_l3.md | laya | completed | laya/AGENTS.md | 2026-09-23 | none |
| research/2026-09-22_laya_vs_jev_benchmarks.md | laya | completed | laya/AGENTS.md | 2026-09-22 | none |
| research/2026-09-23_agent_self_reflection.md | core | completed | core | 2026-09-23 | none |
| research/2026-09-23_laya_improvement_levers.md | laya | active | laya/AGENTS.md | 2026-09-23 | none |
| research/2026-09-23_laya_l3_frontier.md | laya | completed | laya/AGENTS.md | 2026-09-23 | research/2026-09-22_laya_l3.md |
| research/E23_ZM_LEVERAGE_PLAN.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/EXPERIMENTS.md | core | active | core | 2026-09-19 | none |
| research/FINAL_REPORT.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-16 | none |
| research/WHAT_WORKS.md | core | active | core | 2026-09-19 | none |
| research/_template_experiment.md | core | active | core | UNKNOWN | none |
| research/arabic_diacritization/DESIGN.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-13 | none |
| research/arabic_diacritization/arabic_diacritization_deep_research_chatgpt.md | diacritizer | historical | core | 2026-09-11 | none |
| research/arabic_diacritization/arabic_diacritization_deep_research_claude.md | diacritizer | historical | core | UNKNOWN | none |
| research/arabic_diacritization/arabic_diacritization_deep_research_perplexity.md | diacritizer | historical | core | 2026-09-11 | none |
| research/c12_distillation_plan.md | core | historical | core | 2026-09-06 | none |
| research/c12_distillation_report.md | core | completed | core | 2026-09-08 | research/c12_distillation_plan.md |
| research/c12_runbook.md | core | historical | core | 2026-09-06 | none |
| research/c12_tier2_brief.md | core | historical | core | 2026-09-06 | none |
| research/c12_tier_order_decision.md | core | historical | core | 2026-09-06 | none |
| research/crosscheck_flashnext_vs_pipeline.md | core | completed | core | 2026-09-06 | research/c12_distillation_report.md |
| research/csa2_ced_tests/RESULTS.md | core | completed | core | 2026-09-19 | none |
| research/d_gate_overlap/REPORT.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-14 | none |
| research/data_catalog/arabic_datasets.md | core | completed | core | 2026-09-17 | none |
| research/deepseek_v41_nano_adaptation.md | core | historical | core | 2026-09-19 | none |
| research/deepseek_v41_notes.md | core | historical | core | 2026-09-19 | none |
| research/desert-ant-labs-models-report.md | langid | completed | langid/AGENTS.md | 2026-09-18 | none |
| research/desert_ant_recreation/DA1_REPORT.md | langid | completed | langid/AGENTS.md | 2026-09-19 | none |
| research/desert_ant_recreation/DA2_REPORT.md | langid | completed | langid/AGENTS.md | 2026-09-19 | none |
| research/desert_ant_recreation/DA2B_REPORT.md | langid | completed | langid/AGENTS.md | 2026-09-19 | research/desert_ant_recreation/DA2_REPORT.md |
| research/desert_ant_recreation/DA2_SPEC.md | langid | completed | langid/AGENTS.md | 2026-09-19 | none |
| research/desert_ant_recreation/DESIGN.md | langid | completed | langid/AGENTS.md | 2026-09-19 | none |
| research/distill_survey/COMPARISON.md | core | completed | core | 2026-09-08 | none |
| research/distill_survey/H2_RESUME.md | core | historical | core | 2026-09-09 | none |
| research/distill_survey/REPORT.md | core | completed | core | 2026-09-08 | research/distill_survey/COMPARISON.md |
| research/distill_survey/adoption_plan.md | core | historical | core | 2026-09-08 | none |
| research/distill_survey/notes/01_distillation_methods.md | core | completed | core | 2026-09-08 | none |
| research/distill_survey/notes/02_api_only_distillation.md | core | completed | core | 2026-09-08 | none |
| research/distill_survey/notes/03_training_stages_data.md | core | completed | core | 2026-09-08 | none |
| research/distill_survey/notes/04_context_extension_memory.md | core | completed | core | 2026-09-08 | none |
| research/distill_survey/other_research/deep-research-report_chatgpt.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_claude/00-README.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_claude/README1.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_claude/deepseek-v4.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_claude/qwen-strong-to-weak-distillation.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_claude/qwen3.8-flash-next.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_perplexity/distill_research_perplexity.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_perplexity/comprehensive_distillation_research.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_perplexity/practical_implementation_guide.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_perplexity/quick_reference_cheatsheet.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/other_research/distill_research_perplexity/state_of_art_architectures.md | core | historical | core | UNKNOWN | none |
| research/distill_survey/track_a_result.md | core | completed | core | 2026-09-09 | none |
| research/e19_bakeoff/CONTAMINATION.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/E20_CACHE_SMALLMODEL.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/E21_MICRO.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/E22_QCRI_WEAK.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/E23A_GOLD_V3Q.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/E23B_CLASSICAL_S2.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-18 | none |
| research/e19_bakeoff/E23C_ZM_DISTILL.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-18 | none |
| research/e19_bakeoff/E23D_BILSTM_AB.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e19_bakeoff/RESULTS.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/e63_overfit_report_2026-09-23.md | diacritizer | completed | diacritizer/AGENTS.md | 2026-09-23 | none |
| research/gate_results/SUMMARY.md | diacritizer | active | diacritizer/AGENTS.md | 2026-09-17 | none |
| research/gdn_sandbox_design.md | core | historical | core | UNKNOWN | none |
| research/h2_corpus_spec.md | core | historical | core | 2026-09-09 | none |
| research/laya-decision-model-report.md | laya | historical | laya | 2026-09-22 | none |
| research/lora_peft_design.md | core | historical | core | 2026-09-06 | none |
| research/micro_experts/DESIGN.md | mex | completed | mex/AGENTS.md | 2026-09-18 | none |
| research/micro_experts/MU0_REPORT.md | mex | completed | mex/AGENTS.md | 2026-09-19 | none |
| research/micro_experts/MU1_REPORT.md | mex | completed | mex/AGENTS.md | 2026-09-19 | none |
| research/milestone_b_8bit_ab.md | core | historical | core | 2026-09-06 | none |
| research/milestone_d_measurement.md | core | completed | core | 2026-09-09 | research/pretrain_mix_proposal.md |
| research/mounting_ab_report.md | core | completed | core | 2026-09-12 | none |
| research/pretrain_mix_proposal.md | core | historical | core | 2026-09-09 | none |
| research/profiles/2026-09-19_g2_train.summary.md | diacritizer | historical | diacritizer | 2026-09-19 | none |
| research/qwen3_8_flash_next_research.md | core | completed | core | 2026-09-08 | none |
| research/training_analysis_2026-09-08.md | core | historical | core | 2026-09-08 | none |
| research/ui_json_render_tiny_codegen_report.md | core | completed | core | 2026-09-16 | none |
| research/user_probe/PROBE.md | core | historical | core | 2026-09-16 | none |
| research/pipeline_v2/2026-09-26-post-remediation-promotion-preregistration.md | pipeline_v2 | completed | UNKNOWN | 2026-09-26 | none |
| research/ui_dataset_adequacy_r2.md | ui | completed | UNKNOWN | 2026-09-27 | none |
| research/ui_u2_full_train_plan.md | ui | active | UNKNOWN | 2026-09-27 | none |

## Directory-level catalog (payload contents are not read)

The raw store is represented by one directory row so its boundary is
machine-checkable without treating untrusted payload files as doctrine:

| path | scope | status | owner | last-verified | supersedes |
|---|---|---|---|---|---|
| research/raw/ | core | historical | UNKNOWN | UNKNOWN | none |

The per-file report table above intentionally excludes this directory and
all of its descendants. No raw payload is opened, indexed, or promoted.

## Other artifact kinds (classified without payload reads)

- Experiment task payloads: research/adc_exa_tasks.json, research/qwen35_teacher_tasks.json, research/round2-5.json, research/milestone_d_tasks.json/milestone_d_actual.json, research/milestone_g1 task json files under research/raw/ — task definitions/results, treated as evidence data only.
- Result data: research/csa2_ced_tests/*.json, research/d_gate_overlap/overlap_metrics.json, research/gate_results/gate_results.csv, research/data_catalog/arabic_datasets.csv, research/profiles/*.csv — raw result tables backing the .md reports above; same external-data caveat.
- Template: research/_template_experiment.md (pre-registration template; owner rules/experiments.md).
- experiment ledger: research/experiments.yaml (authoritative machine-readable experiment registry; the linter re-checks every run).

Schema choice for research/raw/: it stays out of the per-file table on
purpose — every payload there is an external crawl/LLM answer (untrusted
data, not repository doctrine), it has no per-file metadata the index can
verify without reading the payload, and it is consumed only through the
reports above. The single directory-level row is therefore the complete
raw catalog; the tests verify that boundary without opening any payload.

Caveats recorded honestly: deep-research/other_research payloads are
kept as external-survey inputs; their content is untrusted data. The
active rows above (the living ledgers EXPERIMENTS.md and WHAT_WORKS.md,
the laya improvement levers, the experiment template, and
research/gate_results/SUMMARY.md) stay referenced by
[CURRENT_STATE.md](../CURRENT_STATE.md); everything else is dated
provenance. See [docs/plans/README.md](../docs/plans/README.md) for the
plan-side index.
