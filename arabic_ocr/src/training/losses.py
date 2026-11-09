"""Loss functions for Arabic OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    """CTC loss for CRNN."""
    
    def __init__(self, blank=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    
    def forward(self, logits, targets, target_lengths):
        """
        Args:
            logits: (T, N, C) — log probabilities
            targets: (N, S) or (sum(target_lengths),)
            target_lengths: (N,)
        """
        # logits: (T, N, C) -> log_probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Flatten targets if needed
        if targets.dim() == 2:
            targets = targets[:, :target_lengths.max()]
            # Flatten
            flat_targets = []
            for i, length in enumerate(target_lengths):
                flat_targets.append(targets[i, :length])
            targets = torch.cat(flat_targets)
        
        input_lengths = torch.full(
            (logits.size(1),), logits.size(0), dtype=torch.long
        )
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for transformer decoder."""
    
    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, S, C)
            targets: (N, S)
        """
        # Shift: input is targets[:-1], output is targets[1:]
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.pad_idx,
        )
        return loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""
    
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: (N, S, C)
            teacher_logits: (N, S, C)
            targets: (N, S)
        """
        # CE loss
        ce_loss = self.ce(student_logits, targets)
        
        # KL divergence loss
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
        
        return (1 - self.alpha) * ce_loss + self.alpha * kl_loss
