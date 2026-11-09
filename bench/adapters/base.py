"""
Base adapter interface for all benchmark adapters.

Each adapter is responsible for:
1. Loading a model from a checkpoint
2. Generating predictions on benchmark data
3. Returning predictions in a format the metrics can evaluate
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json

__all__ = ["BaseAdapter"]


class BaseAdapter(ABC):
    """Abstract base class for all benchmark adapters."""
    
    def __init__(self, model_path: str, config: dict[str, Any] | None = None):
        self.model_path = Path(model_path)
        self.config = config or {}
        self.model = None
        self.device = self.config.get("device", "cuda")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from checkpoint."""
        pass
    
    @abstractmethod
    def predict(self, data_path: str) -> list[str]:
        """Generate predictions on benchmark data.
        
        Args:
            data_path: Path to benchmark data file
            
        Returns:
            List of predictions (one per sample)
        """
        pass
    
    @abstractmethod
    def get_references(self, data_path: str) -> list[str]:
        """Get reference answers from benchmark data.
        
        Args:
            data_path: Path to benchmark data file
            
        Returns:
            List of reference answers (one per sample)
        """
        pass
    
    def load_jsonl(self, data_path: str) -> list[dict]:
        """Load a JSONL file."""
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
    
    def load_parquet(self, data_path: str) -> list[dict]:
        """Load a Parquet file."""
        import pandas as pd
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"benchmark data not found: {data_path}")
        
        df = pd.read_parquet(path)
        return df.to_dict('records')
