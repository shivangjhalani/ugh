import argparse
import os
import yaml
from copy import deepcopy
from types import SimpleNamespace

from multicoco.trainer import MultimodalCoconutTrainer
from multicoco.evaluator import MultimodalCoconutEvaluator
from multicoco.utils import setup_logging, set_seed


def _dict_to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_ns(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k == "base_config":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Handle base_config chaining
    if isinstance(data, dict) and "base_config" in data and data["base_config"]:
        base_path = data["base_config"]
        # Resolve relative to the current config file directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_ns = load_config(base_path)
        base_dict = base_ns.__dict__
        merged = _deep_merge(base_dict, data)
        return _dict_to_ns(merged)

    return _dict_to_ns(data)


def main():
    parser = argparse.ArgumentParser(description="MultiCoCo: Multimodal Coconut Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "generate"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    config.debug = args.debug

    setup_logging(config)
    set_seed(getattr(config, "seed", 42))

    if args.mode == "train":
        trainer = MultimodalCoconutTrainer(config)
        trainer.train_all_stages()
    elif args.mode == "eval":
        evaluator = MultimodalCoconutEvaluator(config)
        results = evaluator.evaluate()
        print(f"Evaluation Results: {results}")
    else:
        evaluator = MultimodalCoconutEvaluator(config)
        evaluator.generate_samples()

    print(f"MultiCoCo {args.mode} completed successfully!")


if __name__ == "__main__":
    main()

