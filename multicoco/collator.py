from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class MultimodalCoconutCollator:
    def __init__(self, tokenizer, latent_id: int, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.latent_id = latent_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"].squeeze(0) for item in batch]
        attention_masks = [item["attention_mask"].squeeze(0) for item in batch]
        labels = [item["labels"].squeeze(0) for item in batch]
        pixel_values_list = [item["pixel_values"] for item in batch]
        image_flags_list = [item["image_flags"] for item in batch]

        # Align earliest latent across batch
        earliest_latents = [
            ids.tolist().index(self.latent_id) for ids in input_ids if (ids == self.latent_id).any()
        ]
        if earliest_latents:
            latest_earliest = max(earliest_latents)
            for i, ids in enumerate(input_ids):
                if (ids == self.latent_id).any():
                    pad_needed = latest_earliest - ids.tolist().index(self.latent_id)
                    if pad_needed > 0:
                        pad_tokens = torch.full((pad_needed,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                        input_ids[i] = torch.cat([pad_tokens, ids])
                        attention_masks[i] = torch.cat([torch.zeros(pad_needed, dtype=attention_masks[i].dtype), attention_masks[i]])
                        labels[i] = torch.cat([torch.full((pad_needed,), self.label_pad_token_id, dtype=labels[i].dtype), labels[i]])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

        # Concatenate tiles across batch
        try:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_flags = torch.cat(image_flags_list, dim=0)
        except Exception:
            pixel_values = None
            image_flags = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "choices": [item.get("choices", None) for item in batch],
        }

