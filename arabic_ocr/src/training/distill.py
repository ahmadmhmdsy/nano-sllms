"""Knowledge distillation for Arabic OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationTrainer:
    """Trainer with knowledge distillation."""
    
    def __init__(self, student, teacher, config, device='cuda'):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()  # Teacher is always in eval mode
        self.config = config
        self.device = device
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer for student only
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 10),
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_step = 0
        self.best_cer = float('inf')
    
    def train_step(self, batch):
        """Single training step with distillation."""
        self.student.train()
        
        images = batch['images'].to(self.device)
        texts = batch['texts'].to(self.device)
        text_lens = batch['text_lens'].to(self.device)
        
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Student forward
            student_output = self.student(images, text=texts, text_lens=text_lens)
            student_logits = student_output['logits']
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = self.teacher(images, text=texts)
                teacher_logits = teacher_output['logits']
            
            # CE loss (student)
            ce_loss = student_output['loss']
            
            # KL divergence loss
            T = self.config.get('temperature', 2.0)
            alpha = self.config.get('alpha', 0.5)
            
            # Align sequence lengths
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]
            
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
            
            # Total loss
            loss = (1 - alpha) * ce_loss + alpha * kl_loss
        
        self.scaler.scale(loss).backward()
        
        max_norm = self.config.get('max_grad_norm', 1.0)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate student model."""
        self.student.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch in val_loader:
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            raw_texts = batch['raw_texts']
            
            output = self.student(images, text=texts, text_lens=batch['text_lens'].to(self.device))
            total_loss += output['loss'].item()
            
            # Decode
            from .eval import decode_ctc
            idx_to_char = batch.get('idx_to_char', {})
            predictions = decode_ctc(output['logits'], idx_to_char)
            
            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)
        
        from .eval import compute_cer, compute_wer
        cer = compute_cer(all_predictions, all_targets)
        wer = compute_wer(all_predictions, all_targets)
        
        return {
            'loss': total_loss / len(val_loader),
            'cer': cer,
            'wer': wer,
        }
    
    def fit(self, train_loader, val_loader, epochs=None):
        """Training loop with distillation."""
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        save_dir = Path(self.config.get('save_dir', 'runs/distilled'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                train_loss += metrics['loss']
                
                if (i + 1) % 50 == 0:
                    avg_loss = train_loss / (i + 1)
                    print(f"  Step {i + 1}/{len(train_loader)}: loss={avg_loss:.4f} (ce={metrics['ce_loss']:.4f}, kl={metrics['kl_loss']:.4f})")
            
            # Evaluation
            val_metrics = self.evaluate(val_loader)
            print(f"  Val: loss={val_metrics['loss']:.4f}, CER={val_metrics['cer']:.4f}, WER={val_metrics['wer']:.4f}")
            
            if val_metrics['cer'] < self.best_cer:
                self.best_cer = val_metrics['cer']
                self.save_checkpoint(save_dir / 'best.pt')
                print(f"  New best CER: {self.best_cer:.4f}")
            
            self.scheduler.step()
        
        self.save_checkpoint(save_dir / 'last.pt')
        print(f"\nDistillation complete. Best CER: {self.best_cer:.4f}")
    
    def save_checkpoint(self, path):
        """Save student checkpoint."""
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_cer': self.best_cer,
            'config': self.config,
        }, path)


from pathlib import Path
