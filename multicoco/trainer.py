from types import SimpleNamespace
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from tqdm import tqdm

from .model import MultimodalCoconut
from .dataset import MultimodalProcessor
from .collator import MultimodalCoconutCollator


class MultimodalCoconutTrainer:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        device_str = (
            getattr(getattr(self.config, "hardware", SimpleNamespace(device=None)), "device", None)
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(device_str)
        self._setup_model_and_data()

    def _setup_model_and_data(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.model.name, trust_remote_code=True)

        base_model = AutoModel.from_pretrained(
            self.config.model.name,
            torch_dtype=(torch.bfloat16 if getattr(self.config.training, "bf16", False) else torch.float32),
            low_cpu_mem_usage=getattr(self.config.model, "low_cpu_mem_usage", True),
            trust_remote_code=getattr(self.config.model, "trust_remote_code", True),
        )
        base_model.eval()

        # Add special tokens
        coconut_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        internvl_tokens = ["<img>", "</img>", "<IMG_CONTEXT>"]
        existing = set(self.tokenizer.get_vocab().keys())
        to_add = [t for t in coconut_tokens + internvl_tokens if t not in existing]
        num_added = 0
        if to_add:
            num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})

        # Build model wrapper
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_context_token_id == self.tokenizer.unk_token_id:
            raise ValueError("IMG_CONTEXT token not found after add_special_tokens")

        self.model = MultimodalCoconut(
            base_internvl_model=base_model,
            tokenizer=self.tokenizer,
            latent_token_id=self.tokenizer.convert_tokens_to_ids("<|latent|>"),
            start_latent_id=self.tokenizer.convert_tokens_to_ids("<|start-latent|>"),
            end_latent_id=self.tokenizer.convert_tokens_to_ids("<|end-latent|>"),
            eos_token_id=self.tokenizer.eos_token_id,
            img_context_token_id=img_context_token_id,
            config=self.config,
        ).to(self.device)

        # Resize embeddings on LM only when new tokens were added
        if num_added > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))
            inp = self.model.language_model.get_input_embeddings().weight.data
            avg_vec = inp[:-num_added].mean(dim=0, keepdim=True)
            inp[-num_added:] = avg_vec
            if hasattr(self.model.language_model, "get_output_embeddings"):
                out = self.model.language_model.get_output_embeddings()
                if out is not None and not getattr(self.model.language_model.config, "tie_word_embeddings", True):
                    out.weight.data[-num_added:] = out.weight.data[:-num_added].mean(dim=0, keepdim=True)

        # Derive num_image_token from config/model
        if hasattr(base_model, "config") and hasattr(base_model.config, "vision_config"):
            image_size = getattr(base_model.config, "force_image_size", 448) or base_model.config.vision_config.image_size
            patch = base_model.config.vision_config.patch_size
            downsample = getattr(base_model.config, "downsample_ratio", 0.5)
            self.config.data.num_image_token = int((image_size // patch) ** 2 * (downsample ** 2))
        else:
            self.config.data.num_image_token = getattr(self.config.data, "num_image_token", 256)

        self.data_processor = MultimodalProcessor(self.tokenizer, self.image_processor, self.config)

    def _build_loaders(self, stage: int):
        from datasets import load_dataset

        train_ds = load_dataset(self.config.dataset.hf_dataset_id, split=self.config.dataset.train_split, trust_remote_code=True)
        val_ds = load_dataset(self.config.dataset.hf_dataset_id, split=self.config.dataset.val_split, trust_remote_code=True)

        train_ds.set_transform(lambda s: self.data_processor.process_sample(s, stage))
        val_ds.set_transform(lambda s: self.data_processor.process_sample(s, stage))

        collator = MultimodalCoconutCollator(self.tokenizer, self.tokenizer.convert_tokens_to_ids("<|latent|>"))
        train_loader = DataLoader(
            train_ds, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=collator, num_workers=self.config.training.num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.training.batch_size, shuffle=False, collate_fn=collator, num_workers=self.config.training.num_workers
        )
        return train_loader, val_loader

    def train_all_stages(self) -> None:
        best_acc = 0.0
        for stage in range(self.config.coconut.max_latent_stage + 1):
            train_loader, val_loader = self._build_loaders(stage)
            if self.config.coconut.reset_optimizer or stage == 0:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
            acc = self._train_single_stage(train_loader, val_loader, optimizer, stage)
            if acc > best_acc:
                best_acc = acc

    def _train_single_stage(self, train_loader, val_loader, optimizer, stage: int) -> float:
        epochs_this_stage = (
            getattr(self.config.coconut, "epochs_stage0", self.config.coconut.epochs_per_stage) if stage == 0 else self.config.coconut.epochs_per_stage
        )
        for _ in range(epochs_this_stage):
            self.model.train()
            optimizer.zero_grad()
            progress = tqdm(train_loader, desc=f"Stage {stage}")
            for step, batch in enumerate(progress):
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out["loss"] / self.config.training.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        return self._evaluate(val_loader)

    @torch.no_grad()
    def _evaluate(self, val_loader) -> float:
        self.model.eval()
        total = 0
        correct = 0
        for batch in tqdm(val_loader, desc="Eval"):
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Minimal evaluation: next-token logits proxy (placeholder)
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_flags=batch["image_flags"],
            )
            # Placeholder accuracy increment to keep pipeline functional
            total += 1
            correct += 0
        return (correct / total) if total else 0.0

