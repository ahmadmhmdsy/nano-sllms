# Benchmark Module

A reusable, scalable, customizable benchmarking system for nano_slMs.

## Features

- **Reusable**: Bench any model, not just one-time use
- **Scalable**: Add new benchmarks and models without changing the runner
- **Customizable**: YAML-based configuration for benchmark selection
- **Multi-metrics**: Support for multiple metrics per benchmark
- **Pipeline Integration**: Auto-bench models after training with pipeline_v2

## Quick Start

### Run benchmarks on a model

```bash
# Run all enabled code benchmarks
python -m bench.integration --model runs/target/final/model.safetensors

# Run specific benchmarks
python -m bench.integration --model runs/target/final/model.safetensors --benchmarks humaneval mbpp

# Run with custom config
python -m bench.integration --model runs/target/final/model.safetensors --config bench/config.yaml

# Run on CPU
python -m bench.integration --model runs/target/final/model.safetensors --device cpu
```

### Run from Python

```python
from bench.runner import BenchmarkRunner

runner = BenchmarkRunner(config_path="bench/config.yaml")
result = runner.run(
    model_path="runs/target/final/model.safetensors",
    benchmarks=["humaneval", "mbpp"]
)

print(result.to_dict())
```

## Configuration

Edit `bench/config.yaml` to customize:

- **device**: cuda, cpu, or mps
- **batch_size**: Batch size for inference
- **max_tokens**: Maximum tokens to generate
- **temperature**: Sampling temperature (0 = greedy)
- **benchmarks**: List of benchmark names to run
- **benchmark_type**: Filter by type (code, diacritizer, ui_chain)
- **tags**: Filter by tags
- **enabled_only**: Only run enabled benchmarks

## Adding a New Benchmark

1. Add a `BenchmarkSpec` to `bench/registry.py`:

```python
MY_BENCHMARK = BenchmarkSpec(
    name="my_benchmark",
    description="My custom benchmark",
    benchmark_type="code",
    metrics=("pass@1", "pass@10"),
    data_path="data/benchmarks/my_benchmark/data.jsonl",
    adapter="CausalLMAdapter",
    tags=("code", "custom"),
)

BENCHMARKS["my_benchmark"] = MY_BENCHMARK
```

2. Place benchmark data at the specified path

3. Run: `python -m bench.integration --model <model> --benchmarks my_benchmark`

## Adding a New Adapter

1. Create a new adapter in `bench/adapters/`:

```python
from bench.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    def load_model(self) -> None:
        # Load your model
        pass
    
    def predict(self, data_path: str) -> list[str]:
        # Generate predictions
        pass
    
    def get_references(self, data_path: str) -> list[str]:
        # Get reference answers
        pass
```

2. Register the adapter in `bench/adapters/__init__.py`

3. Use in `BenchmarkSpec.adapter`

## Available Benchmarks

### Code LLM Benchmarks
- **humaneval**: 164 problems, pass@k
- **mbpp**: 500 problems, pass@k
- **bigcodebench**: 1,140 tasks, pass@k
- **livecodebench**: 880 problems, pass@k
- **codecontests**: 16,534 problems (disabled - too big)
- **apps**: 10,000 problems (disabled - scripts unsupported)
- **small-code-models**: clone detection, accuracy/f1

### Diacritizer Benchmarks
- **wikinews2024**: 356 sentences, DER/WER
- **fadel_test**: standard Arabic diacritization
- **sadeed25**: Arabic diacritization

### UI Chain Benchmarks
- **design2code**: 485 webpages, visual similarity
- **webgen_bench**: 6,667 examples, visual similarity
- **ui_bench**: 30 examples, visual similarity
- **reactbench**: 1,618 examples, visual similarity

## Available Metrics

- **pass@k**: pass@1, pass@10, pass@100
- **der**: Diacritic Error Rate
- **der_nocase**: DER (case-insensitive)
- **wer**: Word Error Rate
- **preserve**: Text preservation rate
- **accuracy**: Simple accuracy
- **f1**: F1 score

## Pipeline v2 Integration

The bench module integrates with pipeline_v2 to auto-bench models after training:

```python
from bench.integration import run_benchmarks

# After training completes
result = run_benchmarks(
    model_path="runs/v2/target/final/model.safetensors",
    config_path="bench/config.yaml",
)
```

## Output Format

Results are saved as JSON:

```json
{
  "model": "runs/target/final/model.safetensors",
  "total_duration_seconds": 123.45,
  "results": [
    {
      "benchmark": "humaneval",
      "model": "runs/target/final/model.safetensors",
      "metrics": {
        "pass@1": 0.25,
        "pass@10": 0.45,
        "pass@100": 0.65
      },
      "duration_seconds": 45.67,
      "error": null
    }
  ]
}
```
