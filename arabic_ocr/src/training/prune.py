"""Pruning for Arabic OCR models."""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_model_structured(model, amount=0.3):
    """Structured pruning (remove entire channels/neurons).
    
    Args:
        model: model to prune
        amount: fraction of channels to prune (0-1)
    
    Returns:
        pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=1)
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    return model


def prune_model_unstructured(model, amount=0.3):
    """Unstructured pruning (magnitude-based).
    
    Args:
        model: model to prune
        amount: fraction of weights to prune (0-1)
    
    Returns:
        pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model


def make_pruning_permanent(model):
    """Make pruning permanent (remove pruned weights)."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    return model


def get_model_sparsity(model):
    """Get model sparsity (fraction of zero weights).
    
    Args:
        model: model to analyze
    
    Returns:
        sparsity as a float (0-1)
    """
    total = 0
    zeros = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    
    if total == 0:
        return 0.0
    return zeros / total
