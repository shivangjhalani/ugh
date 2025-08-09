from typing import Any, Dict, Optional

import torch


class MultimodalProcessor:
    """Process VQA samples into model-ready tensors with InternVL token pattern."""

    def __init__(self, tokenizer, image_processor, config) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

        # Coconut tokens (ensure added by trainer setup)
        self.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    def load_image_with_dynamic_tiling(self, image, input_size=448, max_num=12):
        try:
            from internvl.train.dataset import dynamic_preprocess, build_transform
        except Exception:
            from internvl_chat.internvl.train.dataset import dynamic_preprocess, build_transform

        images = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        transform = build_transform(input_size=input_size)
        pixel_values = [transform(tile) for tile in images]
        return torch.stack(pixel_values)

    def _apply_coconut_stage_training(self, reasoning_steps, stage: int, has_visual_context: bool = False) -> str:
        if stage == 0:
            if has_visual_context and reasoning_steps:
                enhanced = []
                for i, s in enumerate(reasoning_steps):
                    if i == 0:
                        enhanced.append(f"Looking at the image, {s.lower()}")
                    else:
                        enhanced.append(s)
                return "\n".join(enhanced)
            return "\n".join(reasoning_steps)

        c_thought = getattr(self.config.coconut, "c_thought", 2)
        n_latent_tokens = stage * c_thought
        latent_seq = (
            "<|start-latent|>" + ("<|latent|>" * n_latent_tokens) + "<|end-latent|>"
        )
        remaining = reasoning_steps[stage:] if stage < len(reasoning_steps) else []
        return latent_seq + ("\n" + "\n".join(remaining) if remaining else "")

    def process_sample(self, sample: Dict[str, Any], stage: int = 0) -> Dict[str, torch.Tensor]:
        image = sample["image"].convert("RGB")
        pixel_values = self.load_image_with_dynamic_tiling(image, input_size=448, max_num=8)
        num_patches = pixel_values.shape[0]

        choices_formatted = "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(sample["choices"]) if i < 4
        )
        question_text = (
            f"<image>\nQuestion: {sample['question']}\n\n"
            f"Choices:\n{choices_formatted}\n\n"
            f"Let me think step by step:\n"
        )

        reasoning_steps = sample.get("rationales", sample.get("steps", [])) or []
        processed_reasoning = self._apply_coconut_stage_training(
            reasoning_steps, stage, has_visual_context=True
        )

        if "correct_choice_idx" in sample:
            idx = int(sample["correct_choice_idx"]) if sample["correct_choice_idx"] is not None else 0
            idx = max(0, min(3, idx))
            answer_text = f"The answer is {chr(65+idx)}."
        else:
            answer_text = "The answer is unknown"

        full_text = question_text + processed_reasoning + "\n" + answer_text
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * (self.config.data.num_image_token * num_patches)) + "</img>"
        full_text = full_text.replace("<image>", image_tokens, 1)

        tok = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_sequence_length,
        )

        q_with_img = question_text.replace("<image>", image_tokens, 1)
        q_len = len(self.tokenizer(q_with_img)["input_ids"])
        labels = tok["input_ids"].clone()
        labels[:, :q_len] = -100

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": torch.tensor([1] * num_patches, dtype=torch.long),
            "choices": sample.get("choices", None),
        }

