
"""Download and measure benchmark sizes on disk."""
import os, sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

from datasets import load_dataset

benchmarks = {}

# 1. HumanEval
print("=== HumanEval ===", flush=True)
try:
    ds = load_dataset("openai/openai_humaneval", split="test")
    # Save to disk to measure size
    out_dir = REPO / "data" / "benchmarks" / "humaneval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_json(out_dir / "humaneval.jsonl")
    size = (out_dir / "humaneval.jsonl").stat().st_size
    benchmarks["HumanEval"] = {
        "problems": len(ds),
        "size_bytes": size,
        "size_mb": round(size / 1024 / 1024, 2),
        "source": "openai/openai_humaneval"
    }
    print(f"  Problems: {len(ds)}", flush=True)
    print(f"  Size: {round(size/1024/1024, 2)} MB", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)
    benchmarks["HumanEval"] = {"error": str(e)}

# 2. MBPP
print("\n=== MBPP ===", flush=True)
try:
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    out_dir = REPO / "data" / "benchmarks" / "mbpp"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_json(out_dir / "mbpp.jsonl")
    size = (out_dir / "mbpp.jsonl").stat().st_size
    benchmarks["MBPP"] = {
        "problems": len(ds),
        "size_bytes": size,
        "size_mb": round(size / 1024 / 1024, 2),
        "source": "google-research-datasets/mbpp"
    }
    print(f"  Problems: {len(ds)}", flush=True)
    print(f"  Size: {round(size/1024/1024, 2)} MB", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)
    benchmarks["MBPP"] = {"error": str(e)}

# 3. Design2Code
print("\n=== Design2Code ===", flush=True)
try:
    ds = load_dataset("SALT-NLP/Design2Code", split="test")
    out_dir = REPO / "data" / "benchmarks" / "design2code"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save as jsonl (images will be saved separately)
    ds.to_json(out_dir / "design2code.jsonl")
    # Also check total directory size
    total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    benchmarks["Design2Code"] = {
        "problems": len(ds),
        "size_bytes": total_size,
        "size_mb": round(total_size / 1024 / 1024, 2),
        "source": "SALT-NLP/Design2Code"
    }
    print(f"  Problems: {len(ds)}", flush=True)
    print(f"  Size: {round(total_size/1024/1024, 2)} MB", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)
    benchmarks["Design2Code"] = {"error": str(e)}

# 4. small-code-models (this is a framework, not a single dataset)
# Let's check the repo size
print("\n=== small-code-models ===", flush=True)
try:
    import subprocess
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/jorge-martinez-gil/small-code-models.git", 
         str(REPO / "data" / "benchmarks" / "small-code-models")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        scm_dir = REPO / "data" / "benchmarks" / "small-code-models"
        total_size = sum(f.stat().st_size for f in scm_dir.rglob("*") if f.is_file())
        benchmarks["small-code-models"] = {
            "problems": "Multiple clone-detection benchmarks",
            "size_bytes": total_size,
            "size_mb": round(total_size / 1024 / 1024, 2),
            "source": "jorge-martinez-gil/small-code-models"
        }
        print(f"  Size: {round(total_size/1024/1024, 2)} MB", flush=True)
    else:
        print(f"  Clone failed: {result.stderr[:200]}", flush=True)
        benchmarks["small-code-models"] = {"error": result.stderr[:200]}
except Exception as e:
    print(f"  Error: {e}", flush=True)
    benchmarks["small-code-models"] = {"error": str(e)}

# Save results
results_path = REPO / "data" / "benchmarks" / "benchmark_sizes.json"
results_path.write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
print(f"\nResults saved to {results_path}", flush=True)
