"""
Adapter for Causal LM models (code LLMs, text generation models).

Supports models trained with the nano_slMs pipeline (S -> P -> T).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import torch

from bench.adapters.base import BaseAdapter

__all__ = ["CausalLMAdapter"]


class CausalLMAdapter(BaseAdapter):
    """Adapter for causal language models."""
    
    def __init__(self, model_path: str, config: dict[str, Any] | None = None):
        super().__init__(model_path, config)
        self.model_config = self.config.get("model_config", {})
        self.max_tokens = self.config.get("max_tokens", 512)
        self.temperature = self.config.get("temperature", 0.0)
        self.batch_size = self.config.get("batch_size", 1)
    
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
        
        # Build model using the pipeline_v2 model factory
        from slm_train.models.causal_lm import build_causal_lm
        
        # Extract model config
        model_cfg = self.model_config.get("model", {})
        if not model_cfg:
            # Try to get from checkpoint
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if "config" in checkpoint:
                model_cfg = checkpoint["config"].get("model", {})
        
        # Get vocab size from tokenizer
        vocab_size = 50257  # Default GPT-2 vocab size
        tokenizer_path = self.model_path.parent / "tokenizer.json"
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_cfg = json.load(f)
                vocab_size = tokenizer_cfg.get("vocab_size", vocab_size)
        
        # Build model
        self.model = build_causal_lm(
            model_config=model_cfg,
            capabilities=[],
            data_context=model_cfg.get("ctx", 512),
            vocab_size=vocab_size,
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
        
        data = self.load_jsonl(data_path)
        predictions = []
        
        for item in data:
            prompt = item.get("prompt", item.get("text", ""))
            if not prompt:
                predictions.append("")
                continue
            
            # Generate
            with torch.no_grad():
                # Tokenize
                from slm_train.data.tokenizer import Tokenizer
                tokenizer = Tokenizer.from_json(self.model_path.parent / "tokenizer.json")
                tokens = tokenizer.encode(prompt)
                
                # Generate
                input_ids = torch.tensor([tokens], device=self.device)
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
                
                # Decode
                generated = tokenizer.decode(output[0].tolist())
                predictions.append(generated)
        
        return predictions
    
    def get_references(self, data_path: str) -> list[str]:
        """Get reference answers from benchmark data."""
        data = self.load_jsonl(data_path)
        references = []
        
        for item in data:
            ref = item.get("canonical_solution", item.get("answer", item.get("reference", "")))
            references.append(ref)
        
        return references
    
    def load_jsonl(self, data_path: str) -> list[dict]:
        """Load a JSONL file with UTF-8 encoding."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"benchmark data not found: {data_path}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
