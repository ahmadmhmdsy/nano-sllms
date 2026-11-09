"""Option A: Compact ViT + Transformer Decoder for Arabic OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_channels=3, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (N, C, H, W) -> (N, embed_dim, H/P, W/P) -> (N, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer (pre-norm).

    Post-norm (norm(x + sub(x))) re-normalises the residual stream at every
    layer, which rank-collapses all patch tokens onto nearly the same vector
    as depth grows. Measured 2026-10-01: after 6 post-norm layers, patch 1 and
    patch 100 of the same image already differed only in the third decimal at
    initialisation, and CTC training drove the encoder output to *exactly*
    constant within 50 steps -- leaving the CTC head and the decoder nothing
    to read. Pre-norm keeps the residual stream unnormalised.
    """

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with causal self- and cross-attention."""

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, memory, attn_mask=None):
        # Self-attention must be causal: a position may not attend to future
        # target tokens, otherwise teacher forcing leaks the answer.
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out

        h = self.norm2(x)
        cross_out, _ = self.cross_attn(h, memory, memory)
        x = x + cross_out

        x = x + self.ffn(self.norm3(x))
        return x


class CompactViTDecoder(nn.Module):
    """Compact ViT encoder + Transformer decoder for OCR."""

    def __init__(
        self,
        vocab_size=256,
        image_height=48,
        image_width=256,
        patch_size=16,
        vit_layers=4,
        vit_hidden=256,
        vit_heads=4,
        vit_ffn=1024,
        decoder_layers=4,
        decoder_hidden=256,
        decoder_heads=4,
        max_seq_len=64,
        dropout=0.1,
        pad_loss_weight=0.1,
        ctc_weight=0.0,
        blank_idx=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_loss_weight = pad_loss_weight
        self.ctc_weight = ctc_weight
        self.blank_idx = blank_idx

        self.patch_embed = PatchEmbed(3, patch_size, vit_hidden)
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, vit_hidden) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_hidden) * 0.02)

        self.vit_layers = nn.ModuleList([
            TransformerEncoderLayer(vit_hidden, vit_heads, vit_ffn, dropout)
            for _ in range(vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_hidden)

        self.projector = nn.Linear(vit_hidden, decoder_hidden)

        self.token_embed = nn.Embedding(vocab_size, decoder_hidden)
        self.pos_embed_dec = nn.Parameter(torch.randn(1, max_seq_len, decoder_hidden) * 0.02)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(decoder_hidden, decoder_heads, decoder_hidden * 4, dropout)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_hidden)
        self.output_head = nn.Linear(decoder_hidden, vocab_size)

        # Auxiliary CTC head on the ENCODER. The seq2seq decoder can minimise
        # its loss from the target prefix alone (a character LM), which lets it
        # ignore the image entirely -- the failure mode measured on 2026-10-01.
        # CTC has no prefix to condition on, so this loss can only fall if the
        # encoder genuinely reads the image. That makes it both a regulariser
        # and a verifiable diagnostic.
        self.ctc_head = nn.Linear(vit_hidden, vocab_size) if ctc_weight > 0 else None

    def encode(self, images):
        """Encode images to features. Returns (N, num_patches + 1, vit_hidden)."""
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.vit_layers:
            x = layer(x)
        return self.vit_norm(x)

    def _decode_tokens(self, features, input_tokens):
        """Run the decoder over raw input token ids (no internal shifting)."""
        x = self.token_embed(input_tokens)
        x = x + self.pos_embed_dec[:, :x.size(1), :]
        memory = self.projector(features)

        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.decoder_layers:
            x = layer(x, memory, attn_mask=causal_mask)
        x = self.decoder_norm(x)
        return self.output_head(x)

    def decode(self, features, text=None):
        """Teacher-forced logits.

        text is the full stored target [SOS, chars..., EOS, PAD...]. The decoder
        input is that target shifted left by one, so logits[t] predicts text[t+1].
        The stored target already starts with SOS, so no SOS is prepended here.
        """
        if text is not None:
            input_tokens = text[:, :-1]
        else:
            input_tokens = torch.ones(features.size(0), 1, dtype=torch.long, device=features.device)
        return self._decode_tokens(features, input_tokens)

    def ctc_log_probs(self, features):
        """Per-frame log-softmax over patch tokens. Returns (T, N, C)."""
        if self.ctc_head is None:
            raise RuntimeError("model was built without a CTC head")
        x = features[:, 1:, :]                      # drop the CLS token
        logits = self.ctc_head(x)
        return F.log_softmax(logits.float(), dim=-1).transpose(0, 1)

    def ctc_loss(self, features, text, text_lens):
        """CTC loss over encoder frames. T must exceed target length + repeats."""
        log_probs = self.ctc_log_probs(features)
        T, N, _ = log_probs.shape
        # text is [SOS, chars..., EOS, PAD...]; strip both sentinels.
        tgt_lens = (text_lens - 2).clamp(min=1)
        keep = tgt_lens <= T
        if not bool(keep.any()):
            return log_probs.sum() * 0.0
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        flat = torch.cat([text[i, 1:1 + int(tgt_lens[i])] for i in idx.tolist()])
        input_lens = torch.full((idx.numel(),), T, dtype=torch.long, device=log_probs.device)
        return F.ctc_loss(
            log_probs[:, keep], flat, input_lens, tgt_lens[keep],
            blank=self.blank_idx, zero_infinity=True,
        )

    def forward(self, images, text=None, text_lens=None):
        """Forward pass. Returns logits plus CE/CTC losses when text is given."""
        features = self.encode(images)
        logits = self.decode(features, text)

        result = {"logits": logits}
        if text is not None:
            # logits has length S-1 and already aligns with text[:, 1:]
            targets = text[:, 1:].contiguous()
            # PAD is down-weighted rather than ignored. Ignoring it leaves the
            # model with no gradient telling it where a line ends, so it emits
            # characters across the whole padded window and never learns EOS.
            weight = torch.ones(logits.size(-1), device=logits.device, dtype=logits.dtype)
            weight[0] = self.pad_loss_weight
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                weight=weight,
            )
            result["ce_loss"] = loss
            if self.ctc_head is not None and text_lens is not None:
                ctc = self.ctc_loss(features, text, text_lens)
                result["ctc_loss"] = ctc
                loss = (1.0 - self.ctc_weight) * loss + self.ctc_weight * ctc
            result["loss"] = loss
        return result

    def generate(self, images, max_len=None):
        """Greedy autoregressive generation. Returns (N, L) token ids."""
        max_len = max_len or self.max_seq_len
        features = self.encode(images)
        batch_size = images.size(0)

        tokens = torch.ones(batch_size, 1, dtype=torch.long, device=images.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            logits = self._decode_tokens(features, tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(2)  # EOS
            if bool(finished.all()):
                break
        return tokens

    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
