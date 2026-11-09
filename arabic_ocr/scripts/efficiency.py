"""Apply efficiency techniques to the best OCR model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.quantize import quantize_model_dynamic, benchmark_model
from src.training.prune import prune_model_unstructured, make_pruning_permanent, get_model_sparsity


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = ArabicOCRDataset(
        config['data']['train_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    val_dataset = ArabicOCRDataset(
        config['data']['val_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load best model
    model_config = config['model'].copy()
    model_config['vocab_size'] = len(train_dataset.char_to_idx)
    model = create_model(model_config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model: {model.count_params()/1e6:.1f}M params")
    print(f"Best CER during training: {checkpoint.get('best_cer', 'N/A')}")
    
    # Benchmark original model
    print("\n=== Original Model ===")
    bench = benchmark_model(model, device=device)
    print(f"  Latency: {bench['latency_ms']:.2f} ms")
    print(f"  Throughput: {bench['throughput']:.1f} img/s")
    
    # Apply dynamic quantization (CPU only for LSTM)
    print("\n=== Dynamic Quantization (INT8) ===")
    try:
        model_quant = quantize_model_dynamic(model)
        # Quantized models run on CPU
        bench_quant = benchmark_model(model_quant, device='cpu')
        print(f"  Latency: {bench_quant['latency_ms']:.2f} ms")
        print(f"  Throughput: {bench_quant['throughput']:.1f} img/s")
        print(f"  Note: Quantized model runs on CPU (LSTM quantization not supported on CUDA)")
    except Exception as e:
        print(f"  Quantization failed: {e}")
        print(f"  Skipping quantization (LSTM not supported on CUDA)")
        bench_quant = bench  # Use original as fallback
    
    # Apply pruning
    print("\n=== Pruning (30% unstructured) ===")
    model_pruned = prune_model_unstructured(model, amount=0.3)
    sparsity = get_model_sparsity(model_pruned)
    print(f"  Sparsity: {sparsity:.2%}")
    
    model_pruned = make_pruning_permanent(model_pruned)
    bench_pruned = benchmark_model(model_pruned, device=device)
    print(f"  Latency: {bench_pruned['latency_ms']:.2f} ms")
    print(f"  Throughput: {bench_pruned['throughput']:.1f} img/s")
    
    # Combined: pruning only (quantization not supported on CUDA for LSTM)
    print("\n=== Pruning Only (Quantization skipped for CUDA) ===")
    model_combined = model_pruned
    bench_combined = bench_pruned
    print(f"  Latency: {bench_combined['latency_ms']:.2f} ms")
    print(f"  Throughput: {bench_combined['throughput']:.1f} img/s")
    print(f"  Speedup: {bench_combined['throughput']/bench['throughput']:.2f}x")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Original:    {bench['latency_ms']:.2f} ms, {bench['throughput']:.1f} img/s")
    print(f"  Quantized:   {bench_quant['latency_ms']:.2f} ms, {bench_quant['throughput']:.1f} img/s")
    print(f"  Pruned:      {bench_pruned['latency_ms']:.2f} ms, {bench_pruned['throughput']:.1f} img/s")
    print(f"  Combined:    {bench_combined['latency_ms']:.2f} ms, {bench_combined['throughput']:.1f} img/s")


if __name__ == '__main__':
    main()
