"""
Configuration loading and validation for the bench module.

Config is YAML-based and supports:
- Device selection (cuda, cpu)
- Batch size and generation parameters
- Benchmark selection (by name, type, or tags)
- Metric selection
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

__all__ = ["load_config", "validate_config", "get_benchmark_list"]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    
    return validate_config(config)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a config dict."""
    # Set defaults
    defaults = {
        "device": "cuda",
        "batch_size": 1,
        "max_tokens": 512,
        "temperature": 0.0,
        "benchmarks": [],
        "benchmark_type": None,
        "tags": [],
        "enabled_only": True,
        "output_dir": "runs/bench_results",
    }
    
    # Merge with defaults
    result = {**defaults, **config}
    
    # Validate device
    if result["device"] not in ("cuda", "cpu", "mps"):
        raise ValueError(f"invalid device: {result['device']}")
    
    # Validate batch_size
    if not isinstance(result["batch_size"], int) or result["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    
    # Validate max_tokens
    if not isinstance(result["max_tokens"], int) or result["max_tokens"] < 1:
        raise ValueError("max_tokens must be a positive integer")
    
    # Validate temperature
    if not isinstance(result["temperature"], (int, float)) or result["temperature"] < 0:
        raise ValueError("temperature must be a non-negative number")
    
    return result


def get_benchmark_list(config: dict[str, Any]) -> list[str]:
    """Get the list of benchmarks to run from config."""
    from bench.registry import list_benchmarks
    
    if config.get("benchmarks"):
        return config["benchmarks"]
    
    specs = list_benchmarks(
        benchmark_type=config.get("benchmark_type"),
        enabled_only=config.get("enabled_only", True),
        tag=config.get("tags")[0] if config.get("tags") else None,
    )
    return [s.name for s in specs]
