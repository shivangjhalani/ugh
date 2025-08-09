import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch


def setup_logging(config: SimpleNamespace) -> None:
    os.makedirs(getattr(config, "output_dir", "./outputs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

