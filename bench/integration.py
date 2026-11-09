"""
Pipeline v2 Integration: auto-bench models after training.

This module provides hooks for pipeline_v2 to automatically run benchmarks
after a training run completes. It can be called:
1. As a post-training hook in the pipeline
2. As a standalone command after training
3. As part of the external gates

Usage:
    python -m bench.integration --run-root runs/v2/target/final
    python -m bench.integration --model runs/target/final/model.safetensors --config bench/config.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import argparse

from bench.runner import BenchmarkRunner
from bench.config import load_config, get_benchmark_list

__all__ = ["run_benchmarks", "main"]


def run_benchmarks(
    model_path: str,
    config_path: str | None = None,
    config: dict | None = None,
    benchmarks: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run benchmarks on a model and save results.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config YAML file
        config: Config dict (alternative to config_path)
        benchmarks: List of benchmark names (overrides config)
        output_dir: Directory to save results
    
    Returns:
        Dict with benchmark results
    """
    # Load config
    if config is None:
        if config_path:
            config = load_config(config_path)
        else:
            config = {}
    
    # Get benchmark list
    if benchmarks is None:
        benchmarks = get_benchmark_list(config)
    
    # Create runner
    runner = BenchmarkRunner(config=config)
    
    # Run benchmarks
    print(f"Running benchmarks on {model_path}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print()
    
    result = runner.run(model_path, benchmarks=benchmarks)
    
    # Save results
    if output_dir is None:
        output_dir = config.get("output_dir", "runs/bench_results")
    
    output_path = Path(output_dir) / f"bench_{Path(model_path).parent.name}.json"
    runner.save_results(result, str(output_path))
    
    return result.to_dict()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run benchmarks on a model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument("--benchmarks", nargs="+", help="List of benchmark names")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Load config
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        if config is None:
            config = {}
        config["device"] = args.device
    
    # Run benchmarks
    run_benchmarks(
        model_path=args.model,
        config=config,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
