"""Training loop for Arabic OCR."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .eval import compute_cer, compute_wer, decode_greedy, decode_ctc, decode_token_ids


class OCRTrainer:
    """Trainer for Arabic OCR models."""

    def __init__(self, model, config, device='cuda', char_to_idx=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.char_to_idx = char_to_idx or {}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0.01),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 10),
        )

        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.global_step = 0
        self.best_cer = float('inf')
        self.history = []

    def train_step(self, batch):
        """Single training step."""
        self.model.train()

        images = batch['images'].to(self.device)
        texts = batch['texts'].to(self.device)
        text_lens = batch['text_lens'].to(self.device)

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            model_type = self.config.get('model_type', 'vit_decoder')
            if model_type in ('crnn', 'vit_hybrid'):
                output = self.model(images, text=texts, text_lens=text_lens)
            else:
                output = self.model(images, text=texts)
            loss = output.get('loss', output.get('logits', torch.tensor(0.0)).sum() * 0)

        self.scaler.scale(loss).backward()

        # Gradient clipping
        max_norm = self.config.get('max_grad_norm', 1.0)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1

        # Log CE and CTC separately. If CTC collapses to the all-blank path,
        # zero_infinity turns its loss into exactly 0 and the total silently
        # becomes pure CE -- a collapse that is invisible in the total alone.
        metrics = {'loss': loss.item()}
        for k in ('ce_loss', 'ctc_loss'):
            if k in output:
                metrics[k] = float(output[k].detach())
        return metrics

    @torch.no_grad()
    def eval_step(self, batch):
        """Single evaluation step."""
        self.model.eval()

        images = batch['images'].to(self.device)
        texts = batch['texts'].to(self.device)
        text_lens = batch['text_lens'].to(self.device)
        raw_texts = batch['raw_texts']

        model_type = self.config.get('model_type', 'vit_decoder')
        if model_type in ('crnn', 'vit_hybrid'):
            output = self.model(images, text=texts, text_lens=text_lens)
        else:
            output = self.model(images, text=texts)

        logits = output['logits']

        # Decode predictions
        if self.config.get('model_type') == 'crnn':
            predictions = decode_ctc(logits, self.idx_to_char)
        else:
            predictions = decode_greedy(logits, self.idx_to_char)

        # Compute metrics
        cer = compute_cer(predictions, raw_texts)
        wer = compute_wer(predictions, raw_texts)

        return {
            'loss': output['loss'].item(),
            'cer': cer,
            'wer': wer,
            'predictions': predictions,
        }

    def fit(self, train_loader, val_loader, epochs=None):
        """Training loop."""
        if epochs is None:
            epochs = self.config.get('epochs', 10)

        save_dir = Path(self.config.get('save_dir', 'runs/smoke'))
        save_dir.mkdir(parents=True, exist_ok=True)

        log_interval = self.config.get('log_interval', 10)
        eval_interval = self.config.get('eval_interval', 10)

        # Interruptions happen (external kills, driver resets). Resuming from
        # the per-epoch checkpoint keeps a multi-hour run from restarting at 0.
        start_epoch = self.maybe_resume(save_dir)
        if start_epoch:
            print("Resumed from %s at epoch %d" % (save_dir / 'last.pt', start_epoch))

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss = 0.0
            ce_sum = 0.0
            ctc_sum = 0.0
            for i, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                train_loss += metrics['loss']
                ce_sum += metrics.get('ce_loss', 0.0)
                ctc_sum += metrics.get('ctc_loss', 0.0)

                if (i + 1) % log_interval == 0:
                    avg_loss = train_loss / (i + 1)
                    print(f"  Step {i + 1}/{len(train_loader)}: loss={avg_loss:.4f}"
                          f" ce={ce_sum / (i + 1):.4f} ctc={ctc_sum / (i + 1):.4f}")

            # Evaluation
            val_metrics = None
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                val_metrics = self.evaluate(val_loader)
                extra = ''
                if 'ctc_cer' in val_metrics:
                    extra = f", CTC_CER={val_metrics['ctc_cer']:.4f}"
                print(f"  Val: loss={val_metrics['loss']:.4f}, CER={val_metrics['cer']:.4f}, WER={val_metrics['wer']:.4f}{extra}")

                # Save best model
                if val_metrics['cer'] < self.best_cer:
                    self.best_cer = val_metrics['cer']
                    self.save_checkpoint(save_dir / 'best.pt')
                    print(f"  New best CER: {self.best_cer:.4f}")

            self.scheduler.step()
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss / max(len(train_loader), 1),
                'val_loss': val_metrics['loss'] if val_metrics else None,
                'cer': val_metrics['cer'] if val_metrics else None,
                'wer': val_metrics['wer'] if val_metrics else None,
            })

            # Checkpoint every epoch: an interruption costs one epoch, not the run.
            self.save_checkpoint(save_dir / 'last.pt', epoch=epoch + 1)
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

        # Save final model
        self.save_checkpoint(save_dir / 'last.pt', epoch=epochs)

        print(f"\nTraining complete. Best CER: {self.best_cer:.4f}")
        return self.history

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_ctc = []

        for batch in val_loader:
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            text_lens = batch['text_lens'].to(self.device)
            raw_texts = batch['raw_texts']

            model_type = self.config.get('model_type', 'vit_decoder')
            if model_type in ('crnn', 'vit_hybrid'):
                output = self.model(images, text=texts, text_lens=text_lens)
            else:
                output = self.model(images, text=texts)

            total_loss += output['loss'].item()

            # Decode. Autoregressive generation is the honest measure because
            # teacher forcing is never available at inference time.
            if self.config.get('model_type') == 'crnn':
                predictions = decode_ctc(output['logits'], self.idx_to_char)
            elif self.config.get('eval_generate', True):
                tokens = self.model.generate(images, max_len=texts.size(1))
                predictions = decode_token_ids(tokens, self.idx_to_char)
            else:
                predictions = decode_greedy(output['logits'], self.idx_to_char)

            all_predictions.extend(predictions)
            all_targets.extend(raw_texts)

            # Diagnostic: CTC decoding of the encoder frames. If this is far
            # better than the autoregressive CER, the encoder reads the image
            # and only the decoder is failing (and vice versa).
            # CRNN has no ctc_head attribute at all (its whole output is CTC),
            # while the ViT hybrid sets ctc_head=None when CTC is disabled.
            if hasattr(self.model, 'ctc_log_probs') and getattr(self.model, 'ctc_head', True) is not None:
                lp = self.model.ctc_log_probs(self.model.encode(images))
                all_ctc.extend(decode_ctc(lp, self.idx_to_char))

        cer = compute_cer(all_predictions, all_targets)
        wer = compute_wer(all_predictions, all_targets)

        out = {
            'loss': total_loss / len(val_loader),
            'cer': cer,
            'wer': wer,
        }
        if all_ctc:
            out['ctc_cer'] = compute_cer(all_ctc, all_targets)
        return out

    def save_checkpoint(self, path, epoch=None):
        """Save model checkpoint."""
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_cer': self.best_cer,
            'config': self.config,
            'history': self.history,
        }
        if epoch is not None:
            payload['epoch'] = epoch
        torch.save(payload, path)

    def maybe_resume(self, save_dir):
        """Restore last.pt if present. Returns the epoch index to start from."""
        path = save_dir / 'last.pt'
        if not self.config.get('resume', True) or not path.exists():
            return 0
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt.get('global_step', 0)
        self.best_cer = ckpt.get('best_cer', self.best_cer)
        self.history = ckpt.get('history', [])
        return int(ckpt.get('epoch', 0))

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_cer = checkpoint['best_cer']
