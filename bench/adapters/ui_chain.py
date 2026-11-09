"""
Adapter for UI Chain models (UI component generation).

Supports models trained with the UI chain pipeline (U2H, U2I, U5, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import torch

from bench.adapters.base import BaseAdapter

__all__ = ["UIChainAdapter"]


class UIChainAdapter(BaseAdapter):
    """Adapter for UI chain models."""
    
    def __init__(self, model_path: str, config: dict[str, Any] | None = None):
        super().__init__(model_path, config)
        self.model_config = self.config.get("model_config", {})
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.0)
    
    def load_model(self) -> None:
        """Load the model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {self.model_path}")
        
        # Load model config if available
        config_path = self.model_path.parent / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                self.model_config = yaml.safe_load(f)
        
        # Build model
        from slm_train.models.causal_lm import build_causal_lm
        
        model_cfg = self.model_config.get("model", self.model_config)
        tokenizer_cfg = self.model_config.get("tokenizer", {})
        
        self.model = build_causal_lm(
            model_config=model_cfg,
            capabilities=[],
            data_context=model_cfg.get("ctx", 512),
            vocab_size=tokenizer_cfg.get("vocab_size", 50257),
            seed=42,
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, data_path: str) -> list[str]:
        """Generate predictions on benchmark data."""
        if self.model is None:
            self.load_model()
        
        # Load data (supports both JSONL and Parquet)
        path = Path(data_path)
        if path.suffix == ".parquet":
            data = self.load_parquet(data_path)
        else:
            data = self.load_jsonl(data_path)
        
        predictions = []
        
        for item in data:
            prompt = item.get("prompt", item.get("text", item.get("instruction", "")))
            if not prompt:
                predictions.append("")
                continue
            
            # Generate
            with torch.no_grad():
                from slm_train.data.tokenizer import Tokenizer
                tokenizer = Tokenizer.from_json(self.model_path.parent / "tokenizer.json")
                tokens = tokenizer.encode(prompt)
                
                input_ids = torch.tensor([tokens], device=self.device)
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
                
                generated = tokenizer.decode(output[0].tolist())
                predictions.append(generated)
        
        return predictions
    
    def get_references(self, data_path: str) -> list[str]:
        """Get reference answers from benchmark data."""
        path = Path(data_path)
        if path.suffix == ".parquet":
            data = self.load_parquet(data_path)
        else:
            data = self.load_jsonl(data_path)
        
        references = []
        
        for item in data:
            ref = item.get("canonical_solution", item.get("answer", item.get("reference", item.get("html", ""))))
            references.append(ref)
        
        return references
