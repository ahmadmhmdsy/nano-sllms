"""CNN + BiLSTM + CTC for line-level Arabic OCR.

CTC is used deliberately. A seq2seq decoder conditioned on the target prefix
can minimise the loss with a character language model alone, and on this data
it does: generation collapsed to fluent Arabic that was identical for
different images. CTC has no prefix to condition on, so the encoder must
actually read the image.

The CNN collapses height to one row and downsamples width by 4, so the CTC
time axis follows the writing direction (T = width/8 = 128 steps here).
Keeping T close to the target length matters: with T far above it the model
falls into the all-blank CTC solution and never aligns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + batch norm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CRNN(nn.Module):
    """CNN + BiLSTM + CTC for OCR."""

    def __init__(
        self,
        vocab_size=256,
        image_height=64,
        image_width=1024,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.cnn = nn.Sequential(
            ConvBlock(3, 64), nn.MaxPool2d((2, 2)),      # H/2,  W/2
            ConvBlock(64, 128), nn.MaxPool2d((2, 2)),    # H/4,  W/4
            ConvBlock(128, 256), nn.MaxPool2d((2, 2)),   # H/8,  W/8
            ConvBlock(256, 256), nn.MaxPool2d((2, 1)),   # H/16, W/4
            ConvBlock(256, 384), nn.MaxPool2d((2, 1)),   # H/32, W/4
            ConvBlock(384, 384), nn.MaxPool2d((2, 1)),   # H/64, W/4
        )

        feat_h = max(1, image_height // 64)
        self.rnn = nn.LSTM(
            384 * feat_h,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size)

    def encode(self, images):
        """Encode images to (N, T, features), T following the writing direction."""
        x = self.cnn(images)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)
        x, _ = self.rnn(x)
        return x

    def decode(self, features, text=None):
        """Features to logits shaped (T, N, vocab) for CTC."""
        logits = self.output_head(features)
        return logits.transpose(0, 1)

    def ctc_log_probs(self, features):
        """Per-frame log-softmax shaped (T, N, vocab).

        Mirrors the ViT hybrid's method so Trainer.evaluate() can report a
        CTC-decoded CER for both model families.
        """
        return F.log_softmax(self.decode(features).float(), dim=-1)

    def forward(self, images, text=None, text_lens=None):
        """Forward pass. Returns logits and, when text is given, the CTC loss."""
        features = self.encode(images)
        logits = self.decode(features)
        result = {"logits": logits}

        if text is not None:
            if text_lens is None:
                text_lens = (text != 0).sum(dim=1)
            # Stored targets are [SOS, chars..., EOS, PAD...]; CTC wants the
            # characters only, so SOS and EOS must be stripped.
            targets, target_lengths = [], []
            for i in range(text.size(0)):
                length = int(text_lens[i].item())
                chars = text[i, 1:max(2, length - 1)]
                targets.append(chars)
                target_lengths.append(chars.numel())
            target_lengths = torch.tensor(
                target_lengths, dtype=torch.long, device=text.device)
            # CTC needs T >= target length + adjacent repeats. A sample that
            # violates this makes the loss infinite, and zero_infinity then
            # zeroes BOTH its loss and its gradient -- it trains on nothing
            # while looking healthy. Drop such samples explicitly and report
            # the count so the loss of data is visible rather than silent.
            T = logits.size(0)
            nonempty = target_lengths > 0
            keep = nonempty & (target_lengths <= T)
            result["ctc_dropped"] = int((nonempty & ~keep).sum())
            if bool(keep.any()):
                flat = torch.cat([t for t, k in zip(targets, keep.tolist()) if k])
                input_lengths = torch.full(
                    (int(keep.sum()),), logits.size(0),
                    dtype=torch.long, device=text.device)
                # fp16 CTC loses precision over 54 classes x 256 frames, and
                # zero_infinity then silently drops the damaged samples, so the
                # loss is computed in fp32.
                log_probs = F.log_softmax(logits.float(), dim=-1)
                result["loss"] = F.ctc_loss(
                    log_probs[:, keep], flat, input_lengths, target_lengths[keep],
                    blank=0, zero_infinity=True,
                )
        return result

    def generate(self, images, max_len=64):
        """Greedy CTC decoding. Returns (N, T) indices."""
        logits = self.decode(self.encode(images))
        return logits.argmax(dim=-1).transpose(0, 1)

    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
