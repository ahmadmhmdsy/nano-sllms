"""
Benchmark Runner: executes benchmarks on models and produces metrics.

The runner is the core of the bench module. It:
1. Loads a model using the appropriate adapter
2. Runs predictions on benchmark data
3. Computes metrics
4. Returns structured results

Usage:
    from bench.runner import BenchmarkRunner
    
    runner = BenchmarkRunner(config_path="bench/config.yaml")
    results = runner.run(
        model_path="runs/target/final/model.safetensors",
        benchmarks=["humaneval", "mbpp"]
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time
from dataclasses import dataclass, field

from bench.registry import BenchmarkSpec, benchmark_for, list_benchmarks
from bench.metrics import metric_for
from bench.adapters import CausalLMAdapter, DiacritizerAdapter, UIChainAdapter

__all__ = ["BenchmarkRunner", "BenchmarkResult", "RunResult"]


@dataclass
class BenchmarkResult:
    """Result of running one benchmark on one model."""
    
    benchmark_name: str
    model_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "model": self.model_path,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class RunResult:
    """Result of running multiple benchmarks on one model."""
    
    model_path: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "model": self.model_path,
            "total_duration_seconds": self.total_duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }


class BenchmarkRunner:
    """Runs benchmarks on models."""
    
    ADAPTER_MAP = {
        "CausalLMAdapter": CausalLMAdapter,
        "DiacritizerAdapter": DiacritizerAdapter,
        "UIChainAdapter": UIChainAdapter,
    }
    
    def __init__(self, config_path: str | None = None, config: dict | None = None):
        """Initialize the runner with a config file or dict."""
        if config is not None:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
    
    def _load_config(self, config_path: str) -> dict:
        """Load config from YAML file."""
        import yaml
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> dict:
        """Default configuration."""
        return {
            "device": "cuda",
            "batch_size": 1,
            "max_tokens": 512,
            "temperature": 0.0,
            "benchmarks": ["humaneval", "mbpp", "bigcodebench"],
        }
    
    def run(
        self,
        model_path: str,
        benchmarks: list[str] | None = None,
        benchmark_type: str | None = None,
    ) -> RunResult:
        """Run benchmarks on a model.
        
        Args:
            model_path: Path to model checkpoint
            benchmarks: List of benchmark names to run (None = all enabled)
            benchmark_type: Filter by benchmark type ("code", "diacritizer", "ui_chain")
        
        Returns:
            RunResult with all benchmark results
        """
        if benchmarks is None:
            specs = list_benchmarks(benchmark_type=benchmark_type, enabled_only=True)
            benchmarks = [s.name for s in specs]
        
        run_result = RunResult(model_path=model_path)
        start_time = time.time()
        
        for bench_name in benchmarks:
            try:
                spec = benchmark_for(bench_name)
                if not spec.enabled:
                    print(f"Skipping disabled benchmark: {bench_name}")
                    continue
                
                result = self._run_benchmark(model_path, spec)
                run_result.results.append(result)
                
                # Print progress
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
                print(f"  {bench_name}: {metrics_str} ({result.duration_seconds:.1f}s)")
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  {bench_name}: ERROR - {error_msg}")
                run_result.results.append(BenchmarkResult(
                    benchmark_name=bench_name,
                    model_path=model_path,
                    error=error_msg,
                ))
        
        run_result.total_duration_seconds = time.time() - start_time
        return run_result
    
    def _run_benchmark(self, model_path: str, spec: BenchmarkSpec) -> BenchmarkResult:
        """Run a single benchmark on a model."""
        start_time = time.time()
        
        # Get adapter
        adapter_class = self.ADAPTER_MAP.get(spec.adapter)
        if adapter_class is None:
            raise ValueError(f"unknown adapter: {spec.adapter}")
        
        # Create adapter
        adapter_config = {
            "device": self.config.get("device", "cuda"),
            "batch_size": self.config.get("batch_size", 1),
            "max_tokens": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 0.0),
        }
        adapter = adapter_class(model_path, adapter_config)
        
        # Generate predictions
        predictions = adapter.predict(spec.data_path)
        references = adapter.get_references(spec.data_path)
        
        # Compute metrics
        metrics = {}
        for metric_name in spec.metrics:
            metric_fn = metric_for(metric_name)
            try:
                score = metric_fn(predictions, references)
                metrics[metric_name] = score
            except Exception as e:
                metrics[metric_name] = float('nan')
                print(f"    Warning: metric {metric_name} failed: {e}")
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name=spec.name,
            model_path=model_path,
            metrics=metrics,
            predictions=predictions,
            references=references,
            duration_seconds=duration,
        )
    
    def save_results(self, result: RunResult, output_path: str) -> None:
        """Save results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {output_path}")
