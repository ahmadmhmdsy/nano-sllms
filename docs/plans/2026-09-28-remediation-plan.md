# Remediation Plan — Critical Bugs & Test Failures

**Created:** 2026-09-28
**Scope:** Fix all critical bugs, test failures, and code hygiene issues identified in the project review
**Estimated total effort:** ~8-12 hours across sub-agents

---

## Milestone 1: Fix Corrupt YAML State Files (CRITICAL)

**Goal:** Make state/tasks.yaml and research/experiments.yaml parseable by any YAML parser.
**Priority:** P0 — blocks all state-dependent tooling
**Estimated effort:** 1-2 hours

### Task 1.1: Fix state/tasks.yaml YAML syntax error

**File:** state/tasks.yaml
**Problem:** Line 291 has "expected <block end>, but found '-'" — a mapping wasn't properly closed before the next list item.

**Steps:**
1. Read lines 280-300 of state/tasks.yaml
2. Identify the exact malformed entry (likely a missing key or incorrect indentation before the pipeline_v2/task-remediation-r1 entry)
3. Fix the YAML syntax — ensure all mapping keys are properly indented and list items are properly separated
4. Verify with: .\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('state/tasks.yaml')); print('OK')"

**Acceptance criteria:** The Python command above prints OK with no errors.

**Sub-agent prompt:**
Fix the YAML syntax error in state/tasks.yaml at line 291. The error is "expected <block end>, but found '-'". Read the file around lines 280-300, identify the malformed entry, fix the indentation/structure, and verify the file parses with yaml.safe_load(). Do NOT change any data values — only fix the syntax.

---

### Task 1.2: Fix research/experiments.yaml YAML syntax error

**File:** research/experiments.yaml
**Problem:** Line 709 has "mapping values are not allowed here" — likely a colon in an unquoted string or incorrect indentation.

**Steps:**
1. Read lines 700-714 of research/experiments.yaml
2. Identify the exact malformed entry (likely the E-008-redesign-proposal entry or the text before it)
3. Fix the YAML syntax — quote strings containing colons, fix indentation
4. Verify with: .\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('research/experiments.yaml')); print('OK')"

**Acceptance criteria:** The Python command above prints OK with no errors.

**Sub-agent prompt:**
Fix the YAML syntax error in research/experiments.yaml at line 709. The error is "mapping values are not allowed here". Read the file around lines 700-714, identify the malformed entry, fix the syntax (likely a colon in an unquoted string or incorrect indentation), and verify the file parses with yaml.safe_load(). Do NOT change any data values — only fix the syntax.

---

## Milestone 2: Fix Research Index and Linter Issues (HIGH)

**Goal:** Make the repository linter pass cleanly.
**Priority:** P1 — blocks CI/CD and agent context validation
**Estimated effort:** 1-2 hours

### Task 2.1: Add missing entries to research/README.md

**File:** research/README.md
**Problem:** At least 4 research markdown files are not indexed.

**Missing entries:**
- research/pipeline_v2/2026-09-26-post-remediation-promotion-preregistration.md
- research/ui_dataset_adequacy_r2.md
- research/ui_u2_full_train_plan.md
- research/distill_survey/other_research/distill_research_perplexity/Conduct a research about the distilled method that.md

**Steps:**
1. Read research/README.md to understand the table format
2. For each missing file, read the first few lines to determine: title, scope, status, date
3. Add rows to the index table in the correct format
4. Verify with: .\.venv\Scripts\python.exe -m pytest tests/agent_context/test_check_agent_context.py::DocIndexTests::test_research_index_covers_every_research_markdown_once -v

**Acceptance criteria:** The test passes.

**Sub-agent prompt:**
Add missing research files to the research/README.md index. The test test_research_index_covers_every_research_markdown_once fails because these files are not indexed: research/pipeline_v2/2026-09-26-post-remediation-promotion-preregistration.md, research/ui_dataset_adequacy_r2.md, research/ui_u2_full_train_plan.md, and research/distill_survey/other_research/distill_research_perplexity/Conduct a research about the distilled method that.md. Read the README to understand the table format, read each file first few lines to get metadata, and add the missing rows. Then verify the test passes.

---

### Task 2.2: Fix bare Python commands in rules/components/core.md

**File:** rules/components/core.md
**Problem:** Lines 29 and 40 use bare python instead of the venv path.

**Steps:**
1. Read rules/components/core.md lines 25-45
2. Replace bare python commands with the venv path: .\.venv\Scripts\python.exe
3. Verify with: .\.venv\Scripts\python.exe -m pytest tests/agent_context/test_check_agent_context.py::LintTreeHelperTestCase::test_real_repository_lints_clean_with_accepted_exceptions -v

**Acceptance criteria:** The test passes (0 issues, 4 accepted exceptions).

**Sub-agent prompt:**
Fix the bare Python commands in rules/components/core.md at lines 29 and 40. The project policy requires all Python commands to use the venv path (.\.venv\Scripts\python.exe). Replace the bare python commands with the venv path. Then verify the linter test passes.

---

## Milestone 3: Clean Up Code Hygiene (MEDIUM)

**Goal:** Clean the project root and fix code organization issues.
**Priority:** P2 — improves maintainability
**Estimated effort:** 2-3 hours

### Task 3.1: Move temporary files to scratch/

**Files to move:**
- _tmp_old_config_gen.py
- _tmp_rows.py
- _tmp_rows2.py
- ._probe.py
- scratch_merge_v3.py
- scratch_sadeed_dedup.py
- scratch_sota_search.py

**Steps:**
1. Create scratch/ if it does not exist
2. Move each file to scratch/
3. Verify no imports break (these are standalone scripts, not modules)

**Acceptance criteria:** No temporary files at project root; all moved files still exist under scratch/.

**Sub-agent prompt:**
Move temporary files from the project root to the scratch/ directory. Files to move: _tmp_old_config_gen.py, _tmp_rows.py, _tmp_rows2.py, ._probe.py, scratch_merge_v3.py, scratch_sadeed_dedup.py, scratch_sota_search.py. These are standalone scripts that should not clutter the project root.

---

### Task 3.2: Rename the weird research filename

**File:** research/distill_survey/other_research/distill_research_perplexity/Conduct a research about the distilled method that.md
**Problem:** The filename is a sentence, not a proper filename.

**Steps:**
1. Read the file to understand its content
2. Rename it to something proper like distill_research_perplexity.md
3. Update any references to it (check research/README.md and any other files that might link to it)
4. Verify the research index test still passes

**Acceptance criteria:** The file has a proper filename; all references are updated; the research index test passes.

**Sub-agent prompt:**
Rename the file research/distill_survey/other_research/distill_research_perplexity/Conduct a research about the distilled method that.md to a proper filename (e.g., distill_research_perplexity.md). Read the file first to understand its content, then rename it and update any references in research/README.md or other files. Verify the research index test still passes.

---

### Task 3.3: Remove or implement empty loop.py

**File:** pipeline_v2/slm_train/training/loop.py
**Problem:** The file is empty (0 bytes).

**Steps:**
1. Check if anything imports from loop.py (grep for "from.*loop import" or "import.*loop")
2. If nothing imports it, delete it
3. If something imports it, either implement the training loop or remove the import

**Acceptance criteria:** The empty file is gone or has content; no broken imports.

**Sub-agent prompt:**
The file pipeline_v2/slm_train/training/loop.py is empty (0 bytes). Check if anything imports from it (search for "from.*loop import" or "import.*loop" in the pipeline_v2 directory). If nothing imports it, delete the file. If something imports it, report what imports it and what the import expects.

---

## Milestone 4: Fix Pipeline_v2 Test Timeout (MEDIUM)

**Goal:** Make pytest pipeline_v2/tests/ complete within a reasonable time.
**Priority:** P2 — blocks CI/CD
**Estimated effort:** 2-3 hours

### Task 4.1: Diagnose pipeline_v2 test timeout

**Problem:** pytest pipeline_v2/tests/ times out after 120 seconds.

**Steps:**
1. Run the tests with --timeout=30 per test to identify which test hangs
2. If no per-test timeout is available, run tests one by one to identify the hanging test
3. Check if the hanging test is making network calls (HF hub, etc.)
4. Check if there is a deadlock or infinite loop

**Acceptance criteria:** Identify the specific test(s) that hang and the root cause.

**Sub-agent prompt:**
Diagnose why pytest pipeline_v2/tests/ times out after 120 seconds. Run the tests with a per-test timeout or one by one to identify which test hangs. Check if it is making network calls, has a deadlock, or has an infinite loop. Report the specific test name and root cause.

---

### Task 4.2: Fix the hanging test(s)

**Steps (depends on Task 4.1 findings):**
1. If it is a network call: mock it or add a skip marker for CI
2. If it is a deadlock: fix the concurrency issue
3. If it is an infinite loop: fix the loop condition
4. Verify with: .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/ -q (should complete in < 60 seconds)

**Acceptance criteria:** All pipeline_v2 tests complete within 60 seconds.

**Sub-agent prompt:**
Based on the diagnosis from Task 4.1, fix the hanging test(s) in pipeline_v2/tests/. The fix should make all tests complete within 60 seconds. If the test is making network calls, mock it. If it is a deadlock or infinite loop, fix the root cause. Verify with .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/ -q.

---

## Milestone 5: Verify All Fixes (HIGH)

**Goal:** Run the full test suite and confirm all tests pass.
**Priority:** P0 — final validation
**Estimated effort:** 30 minutes

### Task 5.1: Run full test suite

**Steps:**
1. Run: .\.venv\Scripts\python.exe -m pytest tests/ --tb=short -q
2. Run: .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/ -q
3. Confirm all tests pass (or only pre-existing skips remain)

**Acceptance criteria:** All tests pass; no failures.

**Sub-agent prompt:**
Run the full test suite and report results. Run both tests/ and pipeline_v2/tests/ directories. Report any failures with full traceback. The goal is to have 0 failures.

---

## Execution Order and Dependencies

Milestone 1 (YAML fixes) leads to Milestone 2 (Research index and linter) and Milestone 5 (Final verification).
Milestone 3 (Code hygiene) leads to Milestone 5 (Final verification).
Milestone 4 (Test timeout) leads to Milestone 5 (Final verification).

**Parallel execution:** Milestones 1, 3, and 4 can run in parallel. Milestone 2 depends on Milestone 1 (the linter checks YAML files). Milestone 5 depends on all others.

---

## Success Criteria

- state/tasks.yaml parses without errors
- research/experiments.yaml parses without errors
- research/README.md indexes all research files
- rules/components/core.md uses venv Python paths
- No temporary files at project root
- No empty loop.py file
- All tests/ pass (0 failures)
- All pipeline_v2/tests/ pass (0 failures, under 60 seconds)
- Linter test passes with 0 issues

---

## Notes for Sub-Agents

- All commands must use the venv: .\.venv\Scripts\python.exe
- Do NOT modify data values in YAML files — only fix syntax
- Do NOT delete any files that are tracked by git without explicit instruction
- When in doubt, ask before making destructive changes
- Each task should be completable in 1-2 hours
- Report back with: what was done, files changed, test results
