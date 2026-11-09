"""Evaluation script for Arabic OCR."""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, compute_wer, decode_greedy, decode_ctc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='arabic_ocr/configs/smoke.yaml')
    parser.add_argument('--checkpoint', default='runs/smoke/best.pt')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    val_dataset = ArabicOCRDataset(
        config['data']['val_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Create model
    model_config = config['model'].copy()
    model_config['vocab_size'] = len(val_dataset.char_to_idx)
    model = create_model(model_config)
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Best CER during training: {checkpoint.get('best_cer', 'N/A')}")
    
    # Evaluate
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            texts = batch['texts'].to(device)
            raw_texts = batch['raw_texts']
            
            if hasattr(model, 'model'):  # Distilled model
                output = model(images, text=texts)
            else:
                output = model(images, text=texts)
            
            # Decode
            if config['model']['type'] == 'crnn':
                idx_to_char = {v: k for k, v in val_dataset.char_to_idx.items()}
                predictions = decode_ctc(output['logits'], idx_to_char)
            else:
                idx_to_char = {v: k for k, v in val_dataset.char_to_idx.items()}
                predictions = decode_greedy(output['logits'], idx_to_char)
            
            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)
    
    cer = compute_cer(all_predictions, all_targets)
    wer = compute_wer(all_predictions, all_targets)
    
    print(f"\nResults:")
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    
    # Show some examples
    print(f"\nExamples:")
    for i in range(min(5, len(all_predictions))):
        print(f"  Target: {all_targets[i]}")
        print(f"  Pred:   {all_predictions[i]}")
        print()


if __name__ == '__main__':
    main()
