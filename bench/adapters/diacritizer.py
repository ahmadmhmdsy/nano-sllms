"""
Adapter for Diacritizer models (Arabic diacritization).

Supports models trained with the diacritizer pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import torch
import importlib.util

from bench.adapters.base import BaseAdapter

__all__ = ["DiacritizerAdapter"]


class DiacritizerAdapter(BaseAdapter):
    """Adapter for Arabic diacritizer models."""
    
    def __init__(self, model_path: str, config: dict[str, Any] | None = None):
        super().__init__(model_path, config)
        self.model_config = self.config.get("model_config", {})
        self.batch_size = self.config.get("batch_size", 64)
    
    def load_model(self) -> None:
        """Load the model from checkpoint."""
        # Handle directory paths: look for best.pt or final/model.pt
        model_file = self._resolve_model_path()
        if model_file is None:
            raise FileNotFoundError(f"model checkpoint not found in: {self.model_path}")
        
        # Load model config if available
        config_path = self._resolve_config_path()
        if config_path and config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.model_config = yaml.safe_load(f)
        
        # Build model
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "diacritizer" / "src"))
        from model import build_from_config
        import tokenizer as TK
        
        model_cfg = self.model_config.get("model", self.model_config)
        self.model = build_from_config(model_cfg, TK.VOCAB_SIZE).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def _resolve_model_path(self) -> Path | None:
        """Resolve model file path from directory or file path."""
        if self.model_path.is_file():
            return self.model_path
        
        # If directory, look for best.pt or final/model.pt
        if self.model_path.is_dir():
            # Try best.pt first
            best_pt = self.model_path / "best.pt"
            if best_pt.exists():
                return best_pt
            
            # Try final/model.pt
            final_pt = self.model_path / "final" / "model.pt"
            if final_pt.exists():
                return final_pt
            
            # Try any .pt file in the directory
            for pt_file in self.model_path.glob("*.pt"):
                return pt_file
        
        return None
    
    def _resolve_config_path(self) -> Path | None:
        """Resolve config file path from directory or file path."""
        if self.model_path.is_file():
            return self.model_path.parent / "config.yaml"
        
        # If directory, look for config.yaml or final/config.yaml
        if self.model_path.is_dir():
            config_path = self.model_path / "config.yaml"
            if config_path.exists():
                return config_path
            
            final_config = self.model_path / "final" / "config.yaml"
            if final_config.exists():
                return final_config
        
        return None
    
    def predict(self, data_path: str) -> list[str]:
        """Generate predictions on benchmark data."""
        if self.model is None:
            self.load_model()
        
        # Load data
        data = self._load_diacritizer_data(data_path)
        predictions = []
        
        # Process in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_preds = self._predict_batch(batch)
            predictions.extend(batch_preds)
        
        return predictions
    
    def get_references(self, data_path: str) -> list[str]:
        """Get reference answers from benchmark data."""
        data = self._load_diacritizer_data(data_path)
        references = []
        
        for item in data:
            ref = item.get("reference", item.get("diacritized", ""))
            references.append(ref)
        
        return references
    
    def _load_diacritizer_data(self, data_path: str) -> list[dict]:
        """Load diacritizer benchmark data.
        
        WikiNews 2024 format: each line is a sentence with diacritized words.
        Words with multiple valid alternatives are separated by "/".
        The bare text is obtained by taking the first alternative of each word
        and removing diacritics.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"benchmark data not found: {data_path}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # The line is already diacritized text with possible "/" separators
                # Create bare text by taking first alternative of each word and stripping diacritics
                bare = self._make_bare_text(line)
                
                data.append({
                    "bare": bare,
                    "diacritized": line,
                })
        
        return data
    
    def _make_bare_text(self, diacritized_text: str) -> str:
        """Create bare text from diacritized text with multi-reference support.
        
        For words with "/" separators, take the first alternative.
        Then strip all diacritics.
        """
        words = diacritized_text.split()
        bare_words = []
        for word in words:
            # Take first alternative if multi-reference
            if "/" in word:
                word = word.split("/")[0]
            bare_words.append(word)
        return self._strip_diacritics(" ".join(bare_words))
    
    def _strip_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text."""
        # Arabic diacritics range: U+064B to U+065F, U+0670, U+06D6 to U+06ED
        diacritics = re.compile(r'[ً-ٰٟۖ-ۭ]')
        return diacritics.sub('', text)
    
    def _predict_batch(self, batch: list[dict]) -> list[str]:
        """Predict a batch of diacritized text."""
        # Load the diacritizer bench module using importlib
        bench_path = Path(__file__).resolve().parent.parent.parent / "diacritizer" / "scripts" / "bench.py"
        spec = importlib.util.spec_from_file_location("diacritizer_bench", bench_path)
        diac_bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diac_bench)
        
        # Use the predict_bares function
        results = diac_bench.predict_bares(
            self.model, 
            [(item["bare"], []) for item in batch], 
            self.model_config.get("model", {}).get("ctx", 128), 
            self.device, 
            batch=len(batch)
        )
        return results
