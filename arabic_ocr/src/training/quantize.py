"""Quantization for Arabic OCR models."""
import torch
import torch.nn as nn


def quantize_model_int8(model, calib_loader, device='cuda'):
    """Post-training quantization to INT8.
    
    Args:
        model: model to quantize
        calib_loader: calibration data loader
        device: device
    
    Returns:
        quantized model
    """
    model.eval()
    model.to(device)
    
    # Fuse Conv-BN layers
    model = torch.quantization.fuse_modules(
        model,
        [['conv', 'bn', 'relu']] if hasattr(model, 'cnn') else [],
        inplace=True,
    )
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate
    with torch.no_grad():
        for batch in calib_loader:
            images = batch['images'].to(device)
            model_prepared(images)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized


def quantize_model_dynamic(model):
    """Dynamic quantization (easier, less accurate).
    
    Args:
        model: model to quantize
    
    Returns:
        quantized model
    """
    # Dynamic quantization for LSTM and Linear layers
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8,
    )
    
    return model_quantized


def benchmark_model(model, input_shape=(1, 3, 48, 256), device='cuda', n_runs=100):
    """Benchmark model inference speed.
    
    Args:
        model: model to benchmark
        input_shape: input tensor shape
        device: device
        n_runs: number of benchmark runs
    
    Returns:
        dict with latency and throughput
    """
    model.eval()
    model.to(device)
    
    x = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    latency_ms = elapsed_ms / n_runs
    throughput = 1000 / latency_ms  # images per second
    
    return {
        'latency_ms': latency_ms,
        'throughput': throughput,
        'n_runs': n_runs,
    }
