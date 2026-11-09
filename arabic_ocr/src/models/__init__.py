"""Model factory for Arabic OCR."""
from .vit_decoder import CompactViTDecoder
from .crnn import CRNN
from .vlm import TinyVLM
from .distilled import DistilledOCR


def create_model(config):
    """Create a model from config.
    
    Args:
        config: dict with model configuration
    
    Returns:
        model instance
    """
    model_type = config.get('type', 'vit_decoder')
    
    if model_type in ('vit_decoder', 'vit_hybrid'):
        return CompactViTDecoder(
            vocab_size=config.get('vocab_size', 256),
            image_height=config.get('image_height', 48),
            image_width=config.get('image_width', 256),
            patch_size=config.get('patch_size', 16),
            vit_layers=config.get('vit_layers', 4),
            vit_hidden=config.get('vit_hidden', 256),
            vit_heads=config.get('vit_heads', 4),
            vit_ffn=config.get('vit_ffn', 1024),
            decoder_layers=config.get('decoder_layers', 4),
            decoder_hidden=config.get('decoder_hidden', 256),
            decoder_heads=config.get('decoder_heads', 4),
            max_seq_len=config.get('max_seq_len', 64),
            dropout=config.get('dropout', 0.1),
            pad_loss_weight=config.get('pad_loss_weight', 0.1),
            ctc_weight=config.get('ctc_weight', 0.0),
            blank_idx=config.get('blank_idx', 0),
        )
    elif model_type == 'crnn':
        return CRNN(
            vocab_size=config.get('vocab_size', 256),
            image_height=config.get('image_height', 48),
            image_width=config.get('image_width', 256),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
        )
    elif model_type == 'vlm':
        return TinyVLM(
            vocab_size=config.get('vocab_size', 256),
            image_height=config.get('image_height', 48),
            image_width=config.get('image_width', 256),
            patch_size=config.get('patch_size', 16),
            vit_layers=config.get('vit_layers', 6),
            vit_hidden=config.get('vit_hidden', 384),
            vit_heads=config.get('vit_heads', 6),
            lm_layers=config.get('lm_layers', 4),
            lm_hidden=config.get('lm_hidden', 384),
            lm_heads=config.get('lm_heads', 6),
            max_seq_len=config.get('max_seq_len', 64),
            dropout=config.get('dropout', 0.1),
        )
    elif model_type == 'distilled':
        return DistilledOCR(
            vocab_size=config.get('vocab_size', 256),
            image_height=config.get('image_height', 48),
            image_width=config.get('image_width', 256),
            patch_size=config.get('patch_size', 16),
            vit_layers=config.get('vit_layers', 4),
            vit_hidden=config.get('vit_hidden', 256),
            vit_heads=config.get('vit_heads', 4),
            vit_ffn=config.get('vit_ffn', 1024),
            decoder_layers=config.get('decoder_layers', 4),
            decoder_hidden=config.get('decoder_hidden', 256),
            decoder_heads=config.get('decoder_heads', 4),
            max_seq_len=config.get('max_seq_len', 64),
            dropout=config.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
