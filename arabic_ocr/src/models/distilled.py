"""Option D: Distilled VLM for Arabic OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_decoder import CompactViTDecoder


class DistilledOCR(nn.Module):
    """Distilled OCR model — same architecture as ViT-Decoder with distillation loss."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = CompactViTDecoder(*args, **kwargs)
    
    def forward(self, images, text=None, teacher_logits=None):
        """Forward pass with optional distillation.
        
        Args:
            images: (N, C, H, W)
            text: (N, S) — target text
            teacher_logits: (N, S, C) — teacher logits for distillation
        
        Returns:
            dict with 'logits', 'loss', and optionally 'distill_loss'
        """
        result = self.model(images, text)
        
        if teacher_logits is not None and text is not None:
            # Distillation loss
            student_logits = result['logits']
            
            # CE loss
            logits_shifted = student_logits[:, :-1, :].contiguous()
            targets = text[:, 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            
            # KL divergence
            T = 2.0
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
            
            result['ce_loss'] = ce_loss
            result['distill_loss'] = kl_loss
            result['loss'] = 0.5 * ce_loss + 0.5 * kl_loss
        
        return result
    
    def generate(self, images, max_len=64):
        """Greedy generation."""
        return self.model.generate(images, max_len)
    
    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
