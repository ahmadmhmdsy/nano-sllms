"""Benchmark evaluation script for Arabic OCR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import json
from torch.utils.data import DataLoader

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, compute_wer, decode_greedy


def evaluate_benchmark(model, benchmark_dir, config, train_dataset, device='cuda'):
    """Evaluate model on a benchmark dataset."""
    # Create dataset with training vocab
    dataset = ArabicOCRDataset(
        benchmark_dir,
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        char_to_idx=train_dataset.char_to_idx,
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            texts = batch['texts'].to(device)
            raw_texts = batch['raw_texts']
            
            output = model(images, text=texts)
            logits = output['logits']
            
            # Decode
            idx_to_char = {v: k for k, v in dataset.char_to_idx.items()}
            predictions = decode_greedy(logits, idx_to_char)
            
            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)
    
    # Compute metrics
    cer = compute_cer(all_predictions, all_targets)
    wer = compute_wer(all_predictions, all_targets)
    
    return {
        'cer': cer,
        'wer': wer,
        'n_samples': len(all_predictions),
        'predictions': all_predictions[:10],  # First 10 for inspection
        'targets': all_targets[:10],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark evaluation')
    parser.add_argument('--checkpoint', default='runs/vit_real/best.pt', help='Path to model checkpoint')
    parser.add_argument('--config', default='arabic_ocr/configs/vit_real.yaml', help='Path to config file')
    parser.add_argument('--max-seq-len', type=int, default=64, help='Max sequence length')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset to get vocab
    train_dataset = ArabicOCRDataset(
        config['data']['train_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    config['model']['vocab_size'] = len(train_dataset.char_to_idx)
    
    model = create_model(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded: {model.count_params()/1e6:.1f}M params")
    print(f"Device: {device}")
    print(f"Vocab size: {config['model']['vocab_size']}")
    
    # Evaluate on benchmarks
    benchmarks = [
        ('isippt', 'data/benchmark/isippt'),
        ('khatt', 'data/benchmark/khatt'),
    ]
    
    results = {}
    for name, path in benchmarks:
        print(f"\n=== Evaluating on {name} ===")
        result = evaluate_benchmark(model, path, config, train_dataset, device)
        results[name] = result
        print(f"  CER: {result['cer']:.4f} ({result['cer']*100:.2f}%)")
        print(f"  WER: {result['wer']:.4f} ({result['wer']*100:.2f}%)")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Sample predictions:")
        for i, (pred, target) in enumerate(zip(result['predictions'][:3], result['targets'][:3])):
            print(f"    {i+1}. Target: {target[:50]}")
            print(f"       Pred:   {pred[:50]}")
    
    # Save results
    output = {
        'model': 'ViT Decoder (7.8M params)',
        'benchmarks': results,
    }
    with open('runs/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to runs/benchmark_results.json")
    
    # Summary
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'Benchmark':<15} {'CER':<10} {'WER':<10} {'Samples':<10}")
    print("-" * 45)
    for name, result in results.items():
        print(f"{name:<15} {result['cer']:.4f}     {result['wer']:.4f}     {result['n_samples']}")


if __name__ == '__main__':
    main()
