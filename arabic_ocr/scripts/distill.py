"""Knowledge distillation training script for Arabic OCR."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, compute_wer, decode_greedy


class DistillationTrainer:
    """Knowledge distillation trainer."""
    
    def __init__(self, teacher_model, student_model, config, device='cuda'):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer for student
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['train']['epochs']
        )
        
        self.temperature = config.get('distill', {}).get('temperature', 2.0)
        self.alpha = config.get('distill', {}).get('alpha', 0.5)  # Weight for distillation loss
        
        self.best_cer = float('inf')
        self.save_dir = Path(config['train']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute distillation loss."""
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard target loss (cross-entropy)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=0  # PAD
        )
        
        # Combined loss
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return loss, kd_loss, ce_loss
    
    def train_epoch(self, loader):
        """Train for one epoch."""
        self.student.train()
        total_loss = 0
        total_kd_loss = 0
        total_ce_loss = 0
        
        for batch in loader:
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            
            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_output = self.teacher(images, text=texts)
                teacher_logits = teacher_output['logits']
            
            # Forward pass through student
            student_output = self.student(images, text=texts)
            student_logits = student_output['logits']
            
            # Compute loss
            loss, kd_loss, ce_loss = self.distillation_loss(student_logits, teacher_logits, texts)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config['train']['max_grad_norm'])
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_ce_loss += ce_loss.item()
        
        return {
            'loss': total_loss / len(loader),
            'kd_loss': total_kd_loss / len(loader),
            'ce_loss': total_ce_loss / len(loader),
        }
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate student model."""
        self.student.eval()
        all_predictions = []
        all_targets = []
        
        for batch in loader:
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            raw_texts = batch['raw_texts']
            
            output = self.student(images, text=texts)
            logits = output['logits']
            
            idx_to_char = {v: k for k, v in loader.dataset.char_to_idx.items()}
            predictions = decode_greedy(logits, idx_to_char)
            
            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)
        
        cer = compute_cer(all_predictions, all_targets)
        wer = compute_wer(all_predictions, all_targets)
        
        return {'cer': cer, 'wer': wer}
    
    def fit(self, train_loader, val_loader):
        """Train the student model."""
        history = []
        
        for epoch in range(self.config['train']['epochs']):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'kd_loss': train_metrics['kd_loss'],
                'ce_loss': train_metrics['ce_loss'],
                'val_cer': val_metrics['cer'],
                'val_wer': val_metrics['wer'],
            })
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['train']['epochs']}")
            print(f"  Train loss: {train_metrics['loss']:.4f} (KD: {train_metrics['kd_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
            print(f"  Val CER: {val_metrics['cer']:.4f}, WER: {val_metrics['wer']:.4f}")
            
            # Save best model
            if val_metrics['cer'] < self.best_cer:
                self.best_cer = val_metrics['cer']
                torch.save({
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_cer': self.best_cer,
                }, self.save_dir / 'best.pt')
                print(f"  Saved best model (CER: {self.best_cer:.4f})")
        
        # Save training history
        import json
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='arabic_ocr/configs/distill.yaml')
    parser.add_argument('--teacher', required=True, help='Path to teacher model checkpoint')
    parser.add_argument('--student', default=None, help='Path to student model checkpoint (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = ArabicOCRDataset(
        config['data']['train_dir'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        augment=True,
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
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Update vocab size
    config['model']['vocab_size'] = len(train_dataset.char_to_idx)
    config['teacher_model']['vocab_size'] = len(train_dataset.char_to_idx)
    
    # Create teacher model
    print("Loading teacher model...")
    teacher = create_model(config['teacher_model'])
    teacher_checkpoint = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.to(device)
    teacher.eval()
    print(f"Teacher loaded: {teacher.count_params()/1e6:.1f}M params")
    
    # Create student model
    print("Creating student model...")
    student = create_model(config['model'])
    if args.student:
        student_checkpoint = torch.load(args.student, map_location=device)
        student.load_state_dict(student_checkpoint['model_state_dict'])
        print("Loaded student checkpoint")
    student.to(device)
    print(f"Student: {student.count_params()/1e6:.1f}M params")
    
    # Create trainer
    trainer = DistillationTrainer(teacher, student, config, device)
    
    # Train
    print("\nStarting distillation training...")
    history = trainer.fit(train_loader, val_loader)
    
    print(f"\nTraining complete! Best CER: {trainer.best_cer:.4f}")


if __name__ == '__main__':
    main()
