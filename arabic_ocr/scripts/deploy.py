"""Deploy Arabic OCR model - ONNX export and demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np
from PIL import Image

from src.data.dataset import ArabicOCRDataset
from src.models import create_model
from src.training.eval import decode_ctc


def export_onnx(model, save_path, input_shape=(1, 3, 48, 256)):
    """Export model to ONNX format."""
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['images'],
        output_names=['logits'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {save_path}")


def load_model(checkpoint_path, config, device='cpu'):
    """Load trained model."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, dataset, device='cpu'):
    """Predict text from image."""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((dataset.image_width, dataset.image_height))
    img_array = np.array(img).transpose(2, 0, 1) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        logits = output['logits']
    
    # Decode
    idx_to_char = {v: k for k, v in dataset.char_to_idx.items()}
    text = decode_ctc(logits, idx_to_char)[0]
    
    return text


def main():
    # Load config
    with open('arabic_ocr/configs/full.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset (for vocab)
    dataset = ArabicOCRDataset(
        config['data']['val_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
    )
    
    # Update vocab size
    config['model']['vocab_size'] = len(dataset.char_to_idx)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('runs/full/best.pt', config['model'], device)
    
    print(f"Model loaded: {model.count_params()/1e6:.1f}M params")
    print(f"Device: {device}")
    
    # Export to ONNX (optional)
    try:
        onnx_path = 'runs/full/model.onnx'
        export_onnx(model, onnx_path)
    except ModuleNotFoundError:
        print("ONNX export skipped (onnxscript not installed)")
    
    # Test prediction
    test_images = list(Path(config['data']['val_dir']).glob('*.png'))[:5]
    print(f"\nTesting on {len(test_images)} images:")
    
    for img_path in test_images:
        # Load ground truth
        label_path = img_path.parent / 'labels.json'
        if label_path.exists():
            import json
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            # labels is a list of dicts with 'image' and 'text' keys
            gt = ''
            for item in labels:
                if item.get('image') == img_path.name:
                    gt = item.get('text', '')
                    break
        else:
            gt = ''
        
        # Predict
        pred = predict(model, img_path, dataset, device)
        
        print(f"  {img_path.name}:")
        print(f"    GT:   {gt}")
        print(f"    Pred: {pred}")
    
    # Benchmark
    print("\nBenchmarking...")
    from src.training.quantize import benchmark_model
    bench = benchmark_model(model, device=device, n_runs=100)
    print(f"  Latency: {bench['latency_ms']:.2f} ms")
    print(f"  Throughput: {bench['throughput']:.1f} img/s")
    
    print("\nDeployment complete!")


if __name__ == '__main__':
    main()
