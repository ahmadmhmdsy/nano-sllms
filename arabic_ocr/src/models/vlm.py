"""Option C: Tiny Vision-Language Model for Arabic OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    
    def __init__(self, in_channels=3, patch_size=16, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer."""
    
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
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TinyVLM(nn.Module):
    """Tiny Vision-Language Model for OCR."""
    
    def __init__(
        self,
        vocab_size=256,
        image_height=48,
        image_width=256,
        patch_size=16,
        vit_layers=6,
        vit_hidden=384,
        vit_heads=6,
        lm_layers=4,
        lm_hidden=384,
        lm_heads=6,
        max_seq_len=64,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # ViT encoder
        self.patch_embed = PatchEmbed(3, patch_size, vit_hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, (image_height // patch_size) * (image_width // patch_size) + 1, vit_hidden) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_hidden) * 0.02)
        
        self.vit_layers = nn.ModuleList([
            TransformerLayer(vit_hidden, vit_heads, vit_hidden * 4, dropout)
            for _ in range(vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_hidden)
        
        # Projector
        self.projector = nn.Linear(vit_hidden, lm_hidden)
        
        # LM decoder
        self.token_embed = nn.Embedding(vocab_size, lm_hidden)
        self.pos_embed_lm = nn.Parameter(torch.randn(1, max_seq_len, lm_hidden) * 0.02)
        
        self.lm_layers = nn.ModuleList([
            TransformerLayer(lm_hidden, lm_heads, lm_hidden * 4, dropout)
            for _ in range(lm_layers)
        ])
        self.lm_norm = nn.LayerNorm(lm_hidden)
        
        # Output head
        self.output_head = nn.Linear(lm_hidden, vocab_size)
    
    def encode(self, images):
        """Encode images to features."""
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        for layer in self.vit_layers:
            x = layer(x)
        
        x = self.vit_norm(x)
        return x
    
    def decode(self, features, text=None):
        """Decode features to logits."""
        batch_size = features.size(0)
        
        if text is not None:
            input_tokens = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.long, device=text.device),
                text[:, :-1]
            ], dim=1)
        else:
            input_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=features.device)
        
        x = self.token_embed(input_tokens)
        x = x + self.pos_embed_lm[:, :x.size(1), :]
        
        # Project features as memory
        memory = self.projector(features)
        
        # Concatenate memory and text embeddings
        x = torch.cat([memory, x], dim=1)
        
        # LM layers
        for layer in self.lm_layers:
            x = layer(x)
        
        x = self.lm_norm(x)
        
        # Only take text positions for output
        x = x[:, memory.size(1):, :]
        
        logits = self.output_head(x)
        return logits
    
    def forward(self, images, text=None):
        """Forward pass."""
        features = self.encode(images)
        logits = self.decode(features, text)
        
        result = {'logits': logits}
        
        if text is not None:
            logits_shifted = logits[:, :-1, :].contiguous()
            targets = text[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            result['loss'] = loss
        
        return result
    
    def generate(self, images, max_len=64):
        """Greedy generation."""
        features = self.encode(images)
        batch_size = images.size(0)
        
        tokens = torch.ones(batch_size, 1, dtype=torch.long, device=images.device)
        
        for _ in range(max_len - 1):
            logits = self.decode(features, tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if (next_token == 2).all():
                break
        
        return tokens
    
    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
