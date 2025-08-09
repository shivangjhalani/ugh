import argparse
import yaml
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


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
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

