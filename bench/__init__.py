"""
Reusable Benchmark Module for nano_slMs

A scalable, customizable benchmarking system that integrates with pipeline_v2.
Supports multi-metric evaluation across code LLMs, diacritizers, and UI chain models.

Usage:
    python -m bench.runner --config bench/config.yaml --model runs/target/final/model.safetensors
    python -m bench.runner --config bench/config.yaml --model runs/diac/e63s_s4_12l160/final/model.pt
    python -m bench.runner --config bench/config.yaml --model runs/v2/ui/u2h_arm_eos_avg/weight-average.pt
"""

__version__ = "1.0.0"
__all__ = ["registry", "runner", "metrics", "config", "adapters", "integration"]
