# Python code corpora survey + HuggingFace access test

Date: 2026-10-14. Owner: research/data (directive 2). Scope: enumerate candidate Python code
datasets/corpora on the net, test which ones THIS account can actually read, and recommend 2-3 for a
~200M-1B token Python fluency corpus. Research + access testing only - no full corpus downloaded.

## 0. BOTTOM LINE (corrected after the gated-access test)

- The bigcode repositories are GATED and THIS ACCOUNT DOES NOT HAVE DATA ACCESS. For every bigcode
  candidate the API returns gated="auto", README.md and metadata are readable WITHOUT accepting terms,
  but every real data file returns 403 GatedRepoError. The earlier inference "README downloaded, so the
  grant is present" was WRONG and is retracted. The user must click "Access repository" on each gated
  dataset page for the token to work on data.
- Obtainable RIGHT NOW without any access request (ungated, real data file byte-read succeeded):
  codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) (Python-only, whole-file), code-search-net/code_search_net Python config,
  Nan-Do/code-search-net-python, bigcode/python-stack-v1-functions-filtered-sc2, HuggingFaceTB/stack-edu
  (Python config), HuggingFaceTB/smollm-corpus (python-edu config), codeparrot/github-code(-clean).
- Recommended: codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) as PRIMARY; the ungated Python function corpora as
  supplements/coverage; HuggingFaceTB/stack-edu Python as the quality-filtered option (content via
  Software Heritage S3). Request bigcode/starcoderdata + the-stack-v2 access only if their size beats
  what the ungated set provides.

## 1. Method and account

- Interpreter: .venv/Scripts/python.exe (CPython 3.12.9, huggingface_hub 1.30.0), Test-Path verified.
- Token: HF_TOKEN read from the gitignored project .env and held in memory only. It was never printed,
  echoed, logged or written to any artifact. Auth headers are constructed in-process.
- Account: whoami name = ahmadsyy (type user).
- Checks per candidate: (a) HfApi.dataset_info; (b) raw GET /api/datasets/<id> with and without token
  (gated flag / existence); (c) for reachable repos HfApi.list_repo_tree plus a real hf_hub_download of
  README.md; (d) datasets-server /size for viewer bytes/rows; (e) a decisive gated test:
  HfApi.get_hf_file_metadata(url=..., token=TOK) and HfApi.file_exists(...) on a REAL data file
  (.parquet/.json.gz), which is the official gated-aware path; (f) direct authenticated HTTP Range GET
  of the first 512 bytes of a real data file via /resolve/.
- KEY METHOD FINDING: metadata, file listings and README.md are readable on gated:auto repos WITHOUT the
  terms grant; only real data files are blocked (403 GatedRepoError). So README success is NOT evidence
  of data access. The authoritative test is (e) on a data file, and (f) as a cross-check.

## 2. Gated-status test result (authoritative)

| Repo | gated | README/metadata | data file test | Verdict |
|---|---|---|---|---|
| bigcode/starcoderdata | auto | readable | GatedRepoError 403 (ada/train-00000-of-00001.parquet) | GATED - no access |
| bigcode/the-stack-v2 | auto | readable | GatedRepoError 403 (data/1C_Enterprise/train-00000-of-00001.parquet) | GATED - no access |
| bigcode/the-stack-v2-dedup | auto | readable | GatedRepoError 403 | GATED - no access |
| bigcode/the-stack-v2-train-full-ids | auto | readable | (same family) | GATED - no access |
| bigcode/the-stack-dedup | auto | readable | GatedRepoError 403 (data/abap/data-00000-of-00001.parquet) | GATED - no access |
| bigcode/the-stack | auto | readable | (same family) | GATED - no access |
| bigcode/the-stack-smol | auto | readable | (gated family; tree scan found no data file in 4000 entries) | GATED - no access |
| nampdn-ai/tiny-codes | auto | readable | GatedRepoError 403 (part_1_200000.parquet) | GATED - no access |
| nampdn-ai/tiny-textbooks, tiny-orca-textbooks | auto | readable | (same) | GATED - no access |
| codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) | false | readable | HTTP 206, 512 B, magic 1f8b (gzip json) | UNGATED - accessible |
| bigcode/python-stack-v1-functions-filtered-sc2 | false | readable | HTTP 206, 512 B, magic 50415231 (parquet) | UNGATED - accessible |
| HuggingFaceTB/stack-edu | false | readable | HTTP 206, 512 B, parquet | UNGATED - accessible |
| HuggingFaceTB/smollm-corpus | false | readable | HTTP 206, 512 B, parquet | UNGATED - accessible |
| codeparrot/code-search-net/code_search_net, Nan-Do/code-search-net-python, all Nvidia/SWE/OpenCoder/eval/instruction repos | false | readable | not individually range-tested | UNGATED - accessible |

Raw: research/raw/corpus_survey/hf_gated_check.json, hf_range_probe.json.

## 3. Master candidate table

Legend: Py = Python-only or Python subset. "Accessible?" = proven for THIS account now. Filter/dedup
state is from the dataset card.

| Candidate (hub id / URL) | Approx size | Py/mixed | Licence | Gated? | Accessible? | Filter / dedup state | Notes |
|---|---|---|---|---|---|---|---|
| codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) | README ~50 GB / 5,361,373 files; hub 56 files ~247 MB each ~= 13.8 GB gz | Python-only | not declared (GitHub source, per-file license column) | false | YES (data range 206) | exact-dedup + line-length/alphanum/auto-generated filters | Whole-file, pipeline public. PRIMARY - the volume ungated Python source. |
| codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.)-valid | 142 MB, 61,373 rows | Python-only | not declared | false | YES (metadata+README) | same as clean | Validation split. |
| bigcode/starcoderdata | 783 GB, ~250B tokens, 86 langs | Py subset via data_dir="python" | "other" (per-file licence; The Stack terms) | auto | NO - data 403 | near-dedup, PII removed, decontaminated; whole files | Best source IF access granted. Measured 2,345.7 tok/row (MEMORY). CONTENT IS INLINE. |
| bigcode/the-stack-v2 | 67.5 TB uncompressed, 600+ langs | Py config | per-file license_type; The Stack terms | auto | NO - data 403 | exact+near-dedup; licence inference GHArchive+ScanCode | CONTENT IS METADATA-ONLY (blob_id); bytes from SWH S3 anyway. |
| bigcode/the-stack-v2-dedup | 32.1 TB uncompressed / 2.9 TB dl | Py config | same | auto | NO - data 403 | ~40% permissive near-dups removed | S3 fetch + gated. |
| bigcode/the-stack-v2-train-full-ids | ~200B tokens train-full | all langs incl Py | same | auto | NO - data 403 | filtered/ID-resolved | S3 fetch + gated. |
| bigcode/the-stack-smol | 2.6 GB, 30 langs, 10k/lang | Py subset (10k files, ~27M tok) | The Stack terms | auto | NO - data 403 | 0.1% random sample | Would be ideal for smoke if granted. |
| bigcode/the-stack-smol-xl | viewer 463,457 rows | Py subset | The Stack terms | false | YES (metadata+README) | larger smol variant | Ungated. |
| bigcode/the-stack | ~6 TB raw / ~3.1 TB dedup, 30 langs | Py config | "other" | auto | NO - data 403 | v1 raw | Superseded. |
| bigcode/the-stack-dedup | ~3.1 TB, 30 langs | Py config | "other" | auto | NO - data 403 | exact dedup | v1 dedup. |
| bigcode/python-stack-v1-functions-filtered-sc2 | 138 MB, 248,934 functions | Python-only | not declared (The Stack subset) | false | YES (data range 206) | function extraction + filters | Clean, small; range-read verified (parquet PAR1). |
| code-search-net/code_search_net (also code_search_net) | 6.43 GB total, 6 langs | mixed; Py config | "other" | false | YES (metadata+README) | original CSN functions + docstrings | Python train 412,178 rows / 1.56 GB; the project's existing M3 source. |
| Nan-Do/code-search-net-python | 599 MB, 455,243 rows | Python-only | not declared | false | YES (metadata+README) | CSN Python extraction | Function-level supplement. |
| codeparrot/github-code | ~1.15 TB, ~115M files | mixed (all langs) | "other" | false | YES (metadata+README) | BigQuery GitHub dump, raw | datasets>=3 removed the loading script (MEMORY); filter Python from parquet. |
| codeparrot/github-code-clean | 11.03M rows | mixed | apache-2.0 | false | YES (metadata+README) | filter on top of github-code | Per-language configs. |
| HuggingFaceTB/stack-edu | 17.5 GB metadata total; Py config 4.95 GB / 25,286,019 files | per-language (Py config) | inherits The Stack v2 (per-file license_type) | false | YES (data range 206) | educational-score filtered (2025, arXiv 2502.02737) | CONTENT via SWH S3 (blob_id), not inline. Best quality signal. |
| HuggingFaceTB/smollm-corpus (config python-edu) | python-edu subset of 673 GB corpus | Python-only subset | odc-by | false | YES (data range 206) | Python files scored >=4 by HuggingFaceTB/python-edu-scorer | CONTENT via SWH S3 (blob_id). SmolLM2's Python mix. |
| nampdn-ai/tiny-codes | ~1.3 GB (11 parquet) | mixed (Python among many) | mit | auto | NO - data 403 | synthetic textbook-style code | Not real-GitHub; gated. |
| nvidia/OpenCodeInstruct | 19 GB, 5M examples | mixed (Python-heavy) | cc-by-4.0 | false | YES (metadata+README) | synthetic instruction + unit tests | Instruction, not pretraining. |
| nvidia/OpenCodeReasoning | 337,766 rows | Python/mixed | card-driven | false | YES (metadata+README) | reasoning traces | 2025 instruction/reasoning. |
| open-r1/codeforces-cots | 9.7 GB, 253,876 rows | Python-heavy | cc-by-4.0 | false | YES (metadata+README) | contest solutions + CoT, decontaminated variants | High-quality algorithmic Python (CoT shape). |
| SWE-bench/SWE-smith | 5.23 GB, 59,136 rows | Python-only | mit | false | YES (metadata+README) | repo-level SWE tasks | 2025. |
| SWE-Gym/SWE-Gym | 43 MB, 2,438 rows | Python-only | card | false | YES (metadata+README) | SWE trajectories | Small. |
| OpenCoder-LLM/opc-annealing-corpus | 11.6M rows | code/mixed | odc-by | false | YES (metadata+README) | synthetic + algorithmic | OpenCoder (ACL 2025) annealing. |
| OpenCoder-LLM/opc-sft-stage1 | 5.6 GB, 4.2M rows | code/mixed | odc-by | false | YES (metadata+README) | SFT stage 1 | Instruction. |
| livecodebench, deepmind/code_contests, mbpp, humaneval | small | Python/mixed | various | false | YES (metadata+README) | benchmark | EVAL ONLY - never train (project rule). |
| CodeAlpaca-20k, python_code_instructions_18k, Evol-Instruct-Code-80k | 8-121 MB | mixed/Py | various | false | YES (metadata+README) | instruction | Instruction seeds. |
| IBM Project CodeNet | ~14M samples, 55 langs (Python-heavy) | mixed, Python-heavy | Apache-2.0 (IBM repo/DAX) | n/a (not on HF) | not via HF | curated submissions | GitHub IBM/Project_CodeNet + IBM DAX. |
| GitHub public dumps: GH Archive, Google BigQuery github_repos, Software Heritage | TB-PB | mixed | per-repo licences | n/a | separate access | raw | Not HF; source of The Stack. |

Not found (404, no such HF repo): bigcode/stack-dedup-python, codeparrot/codeparrot (actual:
transformersbook/codeparrot), neulab/code_search_net, HuggingFaceTB/python-edu (a CONFIG of
HuggingFaceTB/smollm-corpus), HuggingFaceTB/fineweb-edu (actual HuggingFaceFW/fineweb-edu),
HuggingFaceTB/smollm3, ibm/Project_CodeNet, ibm-research/Project_CodeNet.

## 4. GATED datasets - ACCESS REQUESTS THE USER MUST MAKE

These are gated="auto"; this account currently has README/metadata but NOT data access. To use their
data, the user must open each page, accept the terms ("Access repository"), and the token then works
immediately (no re-auth):

- https://huggingface.co/datasets/bigcode/starcoderdata   (recommended - content inline, ~250B tokens)
- https://huggingface.co/datasets/bigcode/the-stack-v2
- https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
- https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids
- https://huggingface.co/datasets/bigcode/the-stack-smol
- https://huggingface.co/datasets/bigcode/the-stack
- https://huggingface.co/datasets/bigcode/the-stack-dedup
- https://huggingface.co/datasets/nampdn-ai/tiny-codes (and tiny-textbooks, tiny-orca-textbooks)

Everything else in the table is ungated (gated=false).

## 5. Recommendation for a ~200M-1B token Python fluency corpus (obtainable now)

1. codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) - PRIMARY. Ungated and byte-range-verified (206). Python-only, whole
   file, already exact-deduped and filtered for line-length/alphanumeric/auto-generated noise; ~50 GB
   uncompressed (~13.8 GB gz on the Hub) is far more than 200M-1B tokens, and the recipe is public.
   Caveat: licence not declared (a per-file license column exists) and it is a 2022 snapshot.
2. Ungated Python FUNCTION corpora as coverage/mix: code-search-net/code_search_net config python
   (412,178 rows / 1.56 GB; already the project's M3 source), Nan-Do/code-search-net-python (455,243
   rows / 599 MB), bigcode/python-stack-v1-functions-filtered-sc2 (248,934 functions / 138 MB; 206
   verified). Together these add ~200M tokens of clean, function-shaped Python and are the safest,
   smallest reproducible pulls. Caveat: function-level, not whole files, so they do not teach
   indentation/scope shape on their own.
3. HuggingFaceTB/stack-edu Python config - QUALITY OPTION. Ungated, 2025 educational-score filtered
   (arXiv 2502.02737), Python config 25.3M files / 4.95 GB metadata - the closest thing to the
   phi-1/SmolLM2 curation signal the approved A->C strategy calls for. Caveat: content is
   METADATA-ONLY on the Hub (blob_id), so file bytes require a Software Heritage S3 download pipeline.

If the user grants bigcode access, replace #1 with bigcode/starcoderdata python (content inline,
~250B tokens, per-file licence, measured 2,345.7 tok/row); the-stack-v2-dedup would then be the
best-provenance source but still needs the SWH S3 pipeline.

Practical plan: build the 200M-1B corpus from codeparrot-clean (+ function corpora), run the project's
quality filter + dedup and the battery decontamination check; request access to bigcode/starcoderdata +
the-stack-v2 in parallel so a larger/inline source is ready if the ungated volume proves insufficient.

## 6. Data-file Range-read results (raw log)

Second pass: first real data file located by list_repo_tree, then its first 512 bytes fetched with an
authenticated HTTP Range request (206 = partial content; 403 = gated).

  bigcode/starcoderdata | HTTPError 403 Forbidden (GATED)
  bigcode/the-stack-smol | HTTPError 403 Forbidden (GATED)
  bigcode/the-stack-v2 | HTTPError 403 Forbidden (GATED)
  bigcode/the-stack-dedup | HTTPError 403 Forbidden (GATED)
  bigcode/python-stack-v1-functions-filtered-sc2 | 206, 512 bytes, magic 50415231 (PAR1 parquet)
  codeparrot/codeparrot-clean (CONFIRMED 2026-10-14 with the project HF_TOKEN: dataset_info gated=False private=False,
56 files = 54 data shards + README/.gitattributes; the web UI prompts for login on the tree view but there is NO real
gate. whoami = ahmadsyy, type user. Downloaded file-000000000001.json.gz = 246,127,841 B (234.7 MiB) to
data/raw_codeparrot/ and decompressed it: records carry repo_name/path/copies/size/content with real Python bodies
(django/core/checks/registry.py etc). ~13.8 GB total across 54 shards. NOTE: the pipeline's HF source path is still
untested; feed the LOCAL .json.gz instead to keep step (a) off that untested path.) | 206, 512 bytes, magic 1f8b0808 (gzip)
  HuggingFaceTB/stack-edu | 206, 512 bytes, magic 50415231 (parquet)
  HuggingFaceTB/smollm-corpus | 206, 512 bytes, magic 50415231 (parquet)
  nampdn-ai/tiny-codes | HTTPError 403 Forbidden (GATED)

## 7. Raw evidence

- research/raw/corpus_survey/hf_access_probe.json - per-repo gated flag, api status, README download,
  viewer rows/bytes, top-level tree.
- research/raw/corpus_survey/hf_gated_check.json - authoritative GatedRepoError/file_exists test on real
  data files.
- research/raw/corpus_survey/hf_range_probe.json - real data-file paths + 512-byte range-read results.
- research/raw/corpus_survey/testfiles/<repo>/README.md - downloaded repo files (access proof for
  metadata; NOT proof of data access).
- research/raw/corpus_survey/probe_hf_access.py, probe_range3.py, probe_gated.py, probe_licenses.py,
  summarize_probe.py.
