# Space Bunny 50k Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build an exactly-50,000-row, production-ready Space Bunny prompt-to-declarative-UI dataset while preserving the existing side corpus and passing the repository codec and validator on every retained row.

**Architecture:** Keep the eight legacy JSONL files immutable. Add a versioned production_50k directory containing a deterministic shared generator core, ten isolated domain modules, per-domain JSONL parts, a merged corpus, a manifest, and a validation report. Domain modules use the same small public interface—Profile, build_record, and write_part—so the parent can vary quotas without allowing agents to edit shared files.

**Tech Stack:** Python 3.12 in the repository uv venv, standard library only for generation, existing ui.src.chain codec, existing ui.src.validate validator, JSONL, SHA-256 fingerprints, deterministic seeds.

---

## Non-negotiable contracts

- Output row shape is exactly: {"arch_hint": string, "task": string, "spec": object, "split": string}.
- split is train, validation, or test; the default production split is deterministic and stratified by domain and archetype.
- Every retained row passes JSON parsing, spec_to_chain, chain_to_spec, validate_chain, and validate_spec with zero issues.
- Only the frozen UI1-2026-09-24-41c catalog is used. No invented component, prop, event, action, expression, or dot-notation state path is allowed.
- The v1 safe expression subset is $state, $bindState, $item, and $index. Nested state, repeat, visible conditions, and catalog actions are allowed only when they validate.
- A row is rejected if it contains Lorem ipsum, an unfilled placeholder, an unknown key, a missing required prop, an invalid pointer, an orphan, a cycle, or an invalid catalog value.
- Existing files under ui/data/space_bunny/side_space_bunny_*.jsonl are never overwritten.

## File map

- Create: docs/plans/2026-09-24-space-bunny-50k-data.md — this approved execution plan.
- Create: ui/data/space_bunny/production_50k/generator_core.py — shared deterministic engine, schema checks, JSONL writer, and CLI helpers.
- Create: ui/data/space_bunny/production_50k/generators/<domain>.py — one isolated domain vocabulary and CLI wrapper per worker; ten files total.
- Create: ui/data/space_bunny/production_50k/parts/<domain>.jsonl — generated domain parts; parent may regenerate these safely.
- Create: ui/data/space_bunny/production_50k/legacy_valid.jsonl — validated copy of retained legacy rows, without modifying legacy inputs.
- Create: ui/data/space_bunny/production_50k/corpus.jsonl — final exactly-50,000-row dataset.
- Create: ui/data/space_bunny/production_50k/manifest.json — counts, seeds, hashes, quotas, split counts, and generation environment.
- Create: ui/data/space_bunny/production_50k/validation_report.json — machine-readable validation and diversity evidence.
- Create: ui/data/space_bunny/production_50k/README.md — regeneration command, schema contract, and known limitations.

## Task 1: Freeze the production contract

- [x] Create generator_core.py with the exact row schema, catalog/validator imports, deterministic seed handling, JSONL serialization, and a reusable write_part(profile, output, count, seed) function.
- [x] Implement profile dataclass fields for key, organizations, subjects, roles, views, metrics, statuses, items, locations, and products.
- [x] Implement eight balanced archetype builders: dashboard_metrics, settings_form, todo_list, profile_card, notifications_center, pricing_cards, search_results, and orders_table.
- [x] Make every builder use depth no greater than four, siblings no greater than six, and total elements no greater than sixteen.
- [x] Add deterministic optional surfaces using only catalog components: Separator, Icon, Image, Skeleton, Spinner, Tooltip, Popover, Link, Toggle, DropdownMenu, ToggleGroup, ButtonGroup, Rating, Carousel, Tabs, Dialog, Drawer, Accordion, and Collapsible.
- [x] Add self-validation before every write: JSON round-trip, spec_to_chain, chain_to_spec, validate_chain, and validate_spec.
- [x] Add exact-duplicate retry logic that changes the deterministic seed without changing the requested count.
- [x] Add a CLI-compatible write_part API and a stable row fingerprint function.

## Task 2: Dispatch ten isolated domain workers

Each worker receives the approved plan, reads generator_core.py and ui/prompts/DATA_GEN_PROMPTS.md, and creates exactly one file: ui/data/space_bunny/production_50k/generators/<domain>.py. A worker must not edit the core, legacy files, plan, manifest, or another worker’s file.

- [x] retail_commerce.py: retail, marketplace, inventory, orders, pricing, customer support, and merchandising vocabulary.
- [x] healthcare.py: clinics, care teams, appointments, triage, lab results, public health, and patient-safe operational language.
- [x] education.py: schools, higher education, courses, learners, attendance, assessment, and family communication.
- [x] logistics.py: warehouses, carriers, routes, fleet, cold chain, dispatch, customs, and delivery exceptions.
- [x] finance.py: banking operations, budgeting, invoices, treasury, fraud review, subscriptions, and risk-aware non-sensitive examples.
- [x] gaming.py: game operations, clans, queues, events, leaderboards, inventory, community moderation, and player support.
- [x] developer_productivity.py: developer tools, incident response, release management, observability, documentation, and team workflows.
- [x] civic_nonprofit.py: public services, community programs, benefits intake, volunteer operations, shelters, and nonprofit program reporting.
- [x] hospitality_travel.py: hotels, property operations, travel planning, reservations, guest services, and itinerary management.
- [x] real_estate_workplace.py: property operations, leasing, facilities, workplace access, maintenance, and commercial teams.

Every module must define a module-level PROFILE, a build_record(seed) function returning a complete row, and a main() that calls write_part. The default count is 4,960 and the default seed is unique per domain. The task text must be natural and must not mention component internals or binding syntax.

## Task 3: Audit and retain legacy rows

- [x] Run a read-only audit over all eight existing JSONL files using the venv Python.
- [x] Parse every line as JSON and validate every row through the complete codec/validator gate.
- [x] Record invalid legacy rows and reasons in validation_report.json; do not edit the source files.
- [x] Copy only valid rows to legacy_valid.jsonl, preserving row order and recording source file/line metadata in the manifest.
- [x] Compute the remaining target: 50,000 minus the number of retained legacy rows.

## Task 4: Generate domain parts and fill the exact target

- [x] Run each domain module through write_part with deterministic, non-overlapping seed ranges.
- [x] Generate at least 4,960 rows per domain initially; trim or extend deterministically during integration so the final retained set is exactly 50,000.
- [x] Keep all domain part files separate until the parent has checked their counts and validation summaries.
- [x] Reject any part with a failed self-check, duplicate fingerprints, unexpected row shape, or catalog error.

## Task 5: Merge, deduplicate, and stratify

- [x] Read legacy_valid.jsonl and the domain parts in a fixed order.
- [x] Canonicalize each row with sorted JSON keys and SHA-256; retain the first occurrence of each exact fingerprint and record every rejected duplicate.
- [x] Select rows in a deterministic round-robin over domains and archetypes until the target is met.
- [x] Assign splits by a fixed hash or deterministic quota: 90% train, 6% validation, 4% test, with each domain and archetype represented where the count permits.
- [x] Ensure no split is empty for any sufficiently large domain/archetype cell.
- [x] Write corpus.jsonl only after the selected row count equals the target exactly.

## Task 6: Validate the final corpus

- [x] Count corpus.jsonl lines and assert exactly 50,000.
- [x] Parse every JSON line and assert the exact four-key row contract and valid split.
- [x] Round-trip and validate every row; fail the build if any issue is returned.
- [x] Check no exact duplicates, no placeholder text, no Lorem ipsum, no unknown component/prop/action/event, no invalid state pointer, no cycle, no orphan, and no unreachable element.
- [x] Compute depth, sibling, element-count, archetype, domain, component, prop, action, condition, repeat, binding, tone, and state-shape distributions.
- [x] Flag overconcentration; do not silently claim diversity. Any threshold breach is reported in validation_report.json and resolved by deterministic replacement before the final corpus is accepted.

## Task 7: Write the manifest, report, and regeneration instructions

- [x] Write manifest.json with target count, retained legacy count, per-domain quotas, seed ranges, source hashes, output hash, split counts, and generation timestamp.
- [x] Write validation_report.json with pass/fail gates, invalid legacy rows, duplicate counts, diversity statistics, and representative rejected examples.
- [x] Write README.md with the exact venv-only commands to audit, regenerate, validate, and reproduce the corpus.
- [x] Review all changed files, confirm no secrets or unrelated edits, and leave the legacy inputs untouched.

## Validation commands

Run from E:python_projects
ano_SLMs with the repository venv only:

~~~powershell
& .\.venv\Scripts\python.exe -m ui.scripts.build_space_bunny_50k --audit
& .\.venv\Scripts\python.exe -m ui.scripts.build_space_bunny_50k --generate
& .\.venv\Scripts\python.exe -m ui.scripts.build_space_bunny_50k --validate
~~~

The parent implementation may expose the same operations through a small script if the module path is not present; the final README must contain the actual working command. No GPU, network, external API, or model download is required.

## Review checkpoints

- After Task 1: run a one-row smoke generation and codec/validator check before dispatching workers.
- After Task 2: inspect all ten module files and confirm each changed only its assigned path.
- After Task 4: validate each domain part independently before merging.
- After Task 6: do not present the dataset as complete unless all hard gates pass and the line count is exactly 50,000.
