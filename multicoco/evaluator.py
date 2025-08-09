from types import SimpleNamespace
from typing import Dict

import torch


class MultimodalCoconutEvaluator:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        # In a fuller version we would load from checkpoint and reuse trainer setup
        self.device = torch.device(getattr(self.config.hardware, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    def evaluate(self) -> Dict[str, float]:
        # Placeholder evaluator to complete CLI; real eval is in Trainer._evaluate
        return {"accuracy": 0.0}

    def generate_samples(self):
        # Placeholder generation method
        pass

