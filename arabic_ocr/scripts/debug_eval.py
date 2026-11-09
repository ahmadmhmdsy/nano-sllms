"""Debug evaluation script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, compute_wer, decode_greedy


def main():
    # Load config
    with open('arabic_ocr/configs/combined.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print('Loading dataset...')
    dataset = ArabicOCRDataset(
        config['data']['val_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    print(f'Dataset size: {len(dataset)}')

    print('Loading model...')
    config['model']['vocab_size'] = len(dataset.char_to_idx)
    model = create_model(config['model'])
    checkpoint = torch.load('runs/combined/best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model loaded')

    print('Creating dataloader...')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print('Evaluating...')
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch['images']
            texts = batch['texts']
            raw_texts = batch['raw_texts']
            output = model(images, text=texts)
            logits = output['logits']
            idx_to_char = {v: k for k, v in dataset.char_to_idx.items()}
            predictions = decode_greedy(logits, idx_to_char)
            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)

    cer = compute_cer(all_predictions, all_targets)
    wer = compute_wer(all_predictions, all_targets)
    print(f'CER: {cer:.4f} ({cer*100:.2f}%)')
    print(f'WER: {wer:.4f} ({wer*100:.2f}%)')

    print('\nSample predictions:')
    for i in range(5):
        print(f'Target: {all_targets[i][:50]}')
        print(f'Pred:   {all_predictions[i][:50]}')
        print()


if __name__ == '__main__':
    main()
