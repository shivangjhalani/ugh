from typing import Dict, Optional

import torch
import torch.nn as nn


class MultimodalCoconut(nn.Module):
    """Core multimodal Coconut model that delegates vision/text to InternVL3.

    This implements:
    - IMG_CONTEXT token replacement using InternVL's gradient-safe pattern
    - Sequential position_ids generation after padding
    - Multi-pass latent token processing (one latent per pass)
    """

    def __init__(
        self,
        base_internvl_model: nn.Module,
        tokenizer,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        img_context_token_id: int,
        config,
    ) -> None:
        super().__init__()

        if base_internvl_model is None:
            raise ValueError("base_internvl_model cannot be None")
        for attr in ("vision_model", "language_model", "mlp1"):
            if not hasattr(base_internvl_model, attr):
                raise ValueError(f"Invalid InternVL model: missing {attr}")

        self.base_internvl_model = base_internvl_model
        self.vision_model = base_internvl_model.vision_model
        self.language_model = base_internvl_model.language_model
        self.visual_projector = base_internvl_model.mlp1

        self.tokenizer = tokenizer
        self.config = config

        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.img_context_token_id = img_context_token_id

        # Generation forward counter for potential FSDP sync usage
        self.gen_forward_cnt = 0

    @torch.no_grad()
    def extract_feature(self, pixel_values: Optional[torch.FloatTensor]):
        if pixel_values is None:
            return None
        return self.base_internvl_model.extract_feature(pixel_values)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Step 1: text embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Step 2: vision features + IMG_CONTEXT replacement
        if pixel_values is not None and image_flags is not None:
            # extract features
            vit_embeds = self.extract_feature(pixel_values)
            if vit_embeds is not None:
                # filter by flags
                image_flags = image_flags.view(-1)
                vit_embeds = vit_embeds[image_flags == 1]

                B, N, C = input_embeds.shape
                input_ids_flat = input_ids.reshape(B * N)
                input_embeds_flat = input_embeds.reshape(B * N, C)
                selected = input_ids_flat == self.img_context_token_id
                try:
                    input_embeds_flat[selected] = (
                        input_embeds_flat[selected] * 0.0 + vit_embeds.reshape(-1, C)
                    )
                except Exception:
                    vit_flat = vit_embeds.reshape(-1, C)
                    n_token = selected.sum()
                    input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_flat[:n_token]
                input_embeds = input_embeds_flat.view(B, N, C)

        # Step 3: sequential position ids
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        return self._coconut_continuous_reasoning(
            input_embeds, input_ids, attention_mask, labels, position_ids
        )

    def _coconut_continuous_reasoning(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        position_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        # find latent positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        if latent_indices.numel() == 0:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )
            return {"loss": outputs.loss, "logits": outputs.logits, "inputs_embeds": inputs_embeds}

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max(len(lst) for lst in latent_lists)
        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache
                ]
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (b_idx, mask_list[pass_idx]) for b_idx, mask_list in enumerate(latent_lists) if len(mask_list) > pass_idx
            ]
            if filling_indices:
                tensor_list = [
                    [inputs_embeds[b, pos, :] for pos in range(inputs_embeds.shape[1])]
                    for b in range(inputs_embeds.shape[0])
                ]
                for b_idx, token_pos in filling_indices:
                    if token_pos > 0:
                        tensor_list[b_idx][token_pos] = hidden_states[b_idx, token_pos - 1 - hidden_states_offset, :]
                inputs_embeds = torch.stack([torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])])

            # compute range update
            if pass_idx + 1 < max_n_latents:
                next_positions = [mask_list[pass_idx + 1] for mask_list in latent_lists if len(mask_list) > pass_idx + 1]
                next_compute_range = (next_compute_range[1], min(next_positions) if next_positions else input_ids.shape[1])
            else:
                next_compute_range = (next_compute_range[1], input_ids.shape[1])

        final_outputs = self.language_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            labels=labels[:, next_compute_range[0] : next_compute_range[1]] if labels is not None else None,
            output_hidden_states=True,
        )
        self.gen_forward_cnt += max_n_latents + 1
        return {"loss": final_outputs.loss, "logits": final_outputs.logits, "inputs_embeds": inputs_embeds}

