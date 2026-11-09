"""
Benchmark Registry: declares all available benchmarks and their metadata.

Each benchmark is a static declaration: name, type, metrics, data source, and
adapter. New benchmarks are added here without changing the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
from pathlib import Path

__all__ = ["BenchmarkSpec", "BENCHMARKS", "benchmark_for", "list_benchmarks"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark declaration: what it is, what it measures, and how to run it."""
    
    name: str
    description: str
    benchmark_type: str  # "code", "diacritizer", "ui_chain", "custom"
    metrics: tuple  # tuple of metric names
    data_path: str  # path to benchmark data (relative to repo root)
    adapter: str  # adapter class name
    enabled: bool = True
    tags: tuple = ()  # optional tags for filtering
    
    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("a benchmark needs a non-empty name")
        if not self.metrics:
            raise ValueError("benchmark " + self.name + " needs at least one metric")


# Code LLM Benchmarks
HUMANEVAL = BenchmarkSpec(
    name="humaneval",
    description="OpenAI HumanEval: 164 programming problems with function signatures",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/humaneval/humaneval.jsonl",
    adapter="CausalLMAdapter",
    tags=("code", "generation", "function-level"),
)

MBPP = BenchmarkSpec(
    name="mbpp",
    description="Google MBPP: 500 crowd-sourced Python programming problems",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/mbpp/mbpp.jsonl",
    adapter="CausalLMAdapter",
    tags=("code", "generation", "beginner"),
)

BIGCODEBENCH = BenchmarkSpec(
    name="bigcodebench",
    description="BigCodeBench: 1,140 diverse code generation tasks",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/bigcodebench/bigcodebench.jsonl",
    adapter="CausalLMAdapter",
    tags=("code", "generation", "diverse"),
)

LIVECODEBENCH = BenchmarkSpec(
    name="livecodebench",
    description="LiveCodeBench: 880 contamination-free problems (release_v5)",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/livecodebench_repo/",
    adapter="CausalLMAdapter",
    tags=("code", "generation", "contamination-free"),
)

CODECONTESTS = BenchmarkSpec(
    name="codecontests",
    description="CodeContests: 16,534 competitive programming problems",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/codecontests/",
    adapter="CausalLMAdapter",
    enabled=False,  # Too big to download (25.17 GiB)
    tags=("code", "generation", "competitive"),
)

APPS = BenchmarkSpec(
    name="apps",
    description="APPS: 10,000 interview-style problems (3 difficulty levels)",
    benchmark_type="code",
    metrics=("pass@1", "pass@10", "pass@100"),
    data_path="data/benchmarks/apps/",
    adapter="CausalLMAdapter",
    enabled=False,  # Dataset scripts unsupported
    tags=("code", "generation", "interview"),
)

SMALL_CODE_MODELS = BenchmarkSpec(
    name="small-code-models",
    description="small-code-models: clone detection benchmarks for <220M models",
    benchmark_type="code",
    metrics=("accuracy", "f1", "precision", "recall"),
    data_path="data/benchmarks/small-code-models/",
    adapter="CausalLMAdapter",
    tags=("code", "clone-detection", "small-models"),
)

# Diacritizer Benchmarks
WIKINEWS2024 = BenchmarkSpec(
    name="wikinews2024",
    description="WikiNews 2024: 356 sentences with multi-reference diacritization",
    benchmark_type="diacritizer",
    metrics=("der", "der_nocase", "wer", "preserve"),
    data_path="data/diac/raw/wikinews/wikinews2024_multi_ref.diac",
    adapter="DiacritizerAdapter",
    tags=("diacritizer", "multi-ref", "arabic"),
)

FADEL_TEST = BenchmarkSpec(
    name="fadel_test",
    description="Fadel test set: standard Arabic diacritization benchmark",
    benchmark_type="diacritizer",
    metrics=("der", "der_nocase", "wer", "preserve"),
    data_path="data/diac/raw/fadel_test.diac",
    adapter="DiacritizerAdapter",
    tags=("diacritizer", "standard", "arabic"),
)

SADEED25 = BenchmarkSpec(
    name="sadeed25",
    description="Sadeed25: Arabic diacritization benchmark",
    benchmark_type="diacritizer",
    metrics=("der", "der_nocase", "wer", "preserve"),
    data_path="data/diac/raw/sadeed25.diac",
    adapter="DiacritizerAdapter",
    tags=("diacritizer", "standard", "arabic"),
)

# UI Chain Benchmarks
DESIGN2CODE = BenchmarkSpec(
    name="design2code",
    description="Design2Code: 485 webpages for front-end generation",
    benchmark_type="ui_chain",
    metrics=("visual_similarity", "text_similarity", "pass@1"),
    data_path="data/benchmarks/design2code/design2code.jsonl",
    adapter="UIChainAdapter",
    tags=("ui", "generation", "front-end"),
)

WEBGEN_BENCH = BenchmarkSpec(
    name="webgen_bench",
    description="WebGen-Bench: 6,667 interactive website generation examples",
    benchmark_type="ui_chain",
    metrics=("visual_similarity", "text_similarity", "pass@1"),
    data_path="data/benchmarks/webgen_bench/data.jsonl",
    adapter="UIChainAdapter",
    tags=("ui", "generation", "website"),
)

UI_BENCH = BenchmarkSpec(
    name="ui_bench",
    description="UI-Bench: 30 text-to-app design examples",
    benchmark_type="ui_chain",
    metrics=("visual_similarity", "text_similarity", "pass@1"),
    data_path="data/benchmarks/ui_bench/",
    adapter="UIChainAdapter",
    tags=("ui", "generation", "app"),
)

REACTBENCH = BenchmarkSpec(
    name="reactbench",
    description="ReactBench: 1,618 React component generation examples",
    benchmark_type="ui_chain",
    metrics=("visual_similarity", "text_similarity", "pass@1"),
    data_path="data/benchmarks/reactbench/data.parquet",
    adapter="UIChainAdapter",
    tags=("ui", "generation", "react"),
)

# Registry of all benchmarks
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "humaneval": HUMANEVAL,
    "mbpp": MBPP,
    "bigcodebench": BIGCODEBENCH,
    "livecodebench": LIVECODEBENCH,
    "codecontests": CODECONTESTS,
    "apps": APPS,
    "small-code-models": SMALL_CODE_MODELS,
    "wikinews2024": WIKINEWS2024,
    "fadel_test": FADEL_TEST,
    "sadeed25": SADEED25,
    "design2code": DESIGN2CODE,
    "webgen_bench": WEBGEN_BENCH,
    "ui_bench": UI_BENCH,
    "reactbench": REACTBENCH,
}


def benchmark_for(name: str) -> BenchmarkSpec:
    """Get one benchmark by name, or raise KeyError."""
    if name not in BENCHMARKS:
        available = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"unknown benchmark: {name}; available: {available}")
    return BENCHMARKS[name]


def list_benchmarks(
    benchmark_type: str | None = None,
    enabled_only: bool = False,
    tag: str | None = None,
) -> list[BenchmarkSpec]:
    """List benchmarks, optionally filtered by type, enabled status, or tag."""
    results = []
    for spec in BENCHMARKS.values():
        if benchmark_type and spec.benchmark_type != benchmark_type:
            continue
        if enabled_only and not spec.enabled:
            continue
        if tag and tag not in spec.tags:
            continue
        results.append(spec)
    return sorted(results, key=lambda s: s.name)
