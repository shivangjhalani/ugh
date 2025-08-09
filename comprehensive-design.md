# MultiCoCo
# Multimodal Latent Space Reasoning Extension: Integrating Coconut Framework with InternVL3-1B for Visual Question Answering : MultiCoCo

## Executive Summary

This design document presents a comprehensive extension of Meta's Coconut latent-space reasoning framework to support multimodal visual question answering through integration with InternVL3-1B-Pretrained. Based on extensive analysis of the Coconut codebase[1] and InternVL3 architecture[2][3], this project combines continuous thought reasoning with state-of-the-art multimodal capabilities to create a novel system that performs visual reasoning through latent space rather than traditional token-based processing[4][5][1].

The key innovation lies in extending Coconut's continuous thought mechanism from purely textual reasoning to multimodal contexts, enabling the model to reason about visual content through hidden state representations while maintaining the breadth-first search capabilities (mentioned in Coconut research paper as emergent patterns after coconut training) that make Coconut superior for complex reasoning tasks[4][6].

## Background Reading

1. Coconut paper and codebase:
– Paper: https://arxiv.org/abs/2412.06769
– Code: https://github.com/facebookresearch/coconut (present in reference folder : reference/coconut)

2. InternVL repository and pretrained weights:
– Repo: https://github.com/OpenGVLab/InternVL.git (present in reference folder : reference/InternVL)
– Model: InternVL3-1B-Pretrained
– Paper: https://arxiv.org/html/2504.10479v1

3. A-OKVQA (HuggingFaceM4/A-OKVQA)


## InternVL3-1B Built-in Special Tokens

InternVL3-1B-Pretrained includes several built-in special tokens for multimodal processing:

**Core Image Tokens:**
- `<image>`: Basic image placeholder token used in prompts
- `<IMG_CONTEXT>`: Context token that gets replaced with actual visual features
- `<img>`: Image start token
- `</img>`: Image end token

**Additional Special Tokens:**
- `<im_patch>`: Image patch token for fine-grained visual processing
- `<im_start>`: Alternative image start token
- `<im_end>`: Alternative image end token

**Spatial Reasoning Tokens:**
- `<quad>`, `</quad>`: Quadrilateral/bounding box tokens
- `<ref>`, `</ref>`: Reference tokens for object referencing
- `<box>`, `</box>`: Box coordinate tokens

**Token Processing Pattern:**
Note on naming: In code we use literal tokens (`<img>`, `<IMG_CONTEXT>`, `</img>`). Conceptually these map 1:1 to InternVL’s start/context/end image tokens.
The InternVL3 tokenizer processes images by replacing `<image>` placeholders with structured token sequences:
```
<image> → <img><IMG_CONTEXT><IMG_CONTEXT>...<IMG_CONTEXT></img>
```
256 `<IMG_CONTEXT>` tokens per tile (1-12 tiles supported for dynamic resolution).

**InternVL3-1B-Pretrained Integration Requirements:**

1. **Special Token Addition**: MUST add InternVL3 tokens (`<img>`, `</img>`, `<IMG_CONTEXT>`) in addition to Coconut tokens
2. **IMG_CONTEXT Token ID Assignment**: MUST set `model.img_context_token_id` on the model instance after initialization
3. **Model Component Verification**: MUST verify `mlp1`, `vision_model`, `language_model` attributes exist before use
4. **Configuration Validation**: MUST check `downsample_ratio` and `ps_version` in model config
5. **Token Replacement Pattern**: Follow exact InternVL3 pattern for IMG_CONTEXT token replacement
6. **Dynamic Resolution Support**: Handle variable tile counts per batch item correctly
7. **Pixel Shuffle Implementation**: Use exact InternVL3 pixel_shuffle with ps_version handling

**Critical Special Token Management Protocol:**

For InternVL3-1B-Pretrained models, special tokens MUST be added to the `language_model` component, not the top-level InternVL model. The InternVL model lacks `set_output_embeddings` method required by transformers' `resize_token_embeddings`.

```python
# CORRECT: Add tokens to language model component
if hasattr(model, 'language_model'):
    target_model = model.language_model
    target_model.resize_token_embeddings(len(tokenizer))
else:
    # For standalone language models
    model.resize_token_embeddings(len(tokenizer))
```

**Required Token Set:**
- Coconut tokens: `<|start-latent|>`, `<|end-latent|>`, `<|latent|>`
- InternVL3 tokens: `<img>`, `</img>`, `<IMG_CONTEXT>` (may already exist)

**Embedding Initialization Strategy:**
Initialize new token embeddings using averaged embeddings from common reference tokens ("the", "a", "an", etc.) to provide meaningful starting representations rather than random initialization.

## 1. Background Analysis and Technical Foundation

### 1.1 Coconut Framework Deep Dive

#### 1.1.1 Core Architecture Analysis

The Coconut framework represents a paradigm shift from language-space reasoning to continuous latent space reasoning[4][5]. Based on comprehensive codebase analysis[1], the architecture consists of:

**Primary Components:**

- `Coconut` class: Wrapper around base causal language models with latent reasoning capability
- **Special Token System**: Three critical tokens manage continuous reasoning:
    - `<|start-latent|>`: Marks beginning of latent thought mode
    - `<|end-latent|>`: Marks end of latent thought mode
    - `<|latent|>`: Represents individual continuous thoughts
- **Multi-pass Forward Mechanism**: Alternates between language and latent modes through multiple forward passes[1][7]

AI IMPLEMENTATION NOTE: Critical Coconut Architecture Details
1. **Token Replacement Logic**: The core mechanism is `tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]`
2. **KV Cache Management**: Essential for multi-pass efficiency - must extract and reuse past_key_values correctly
3. **Generation Counter**: `gen_forward_cnt` tracks forward passes for FSDP synchronization in distributed training
4. **Embedding Access**: Use `self.base_causallm.get_input_embeddings()` for GPT2, `self.base_causallm.get_input_embeddings()` for Llama
5. **Batch Size Limitation**: Original Coconut only supports batch_size=1 for generation - this MUST be extended for multimodal use

**Continuous Thought Implementation:**

```python
# From coconut.py analysis
class Coconut(nn.Module):
    def forward(self, input_ids, attention_mask, labels, position_ids):
        # Process latent tokens through multiple passes
        for pass_idx in range(max_n_latents):
            # Extract hidden states as continuous thoughts
            hidden_states = outputs.hidden_states[-1]
            # Replace latent tokens with preceding hidden states
            tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - offset, :]
```


#### 1.1.2 Stage-Based Training Methodology

Coconut employs a sophisticated curriculum learning approach with distinct training stages[1]:

**Stage 0**: Full Chain-of-Thought reasoning in natural language
**Stage 1**: Replace first reasoning step with `c_thought` continuous thoughts
**Stage 2**: Replace first two reasoning steps with latent thoughts
**Stage N**: Replace N reasoning steps with latent representations

**Key Training Parameters:**

- `c_thought = 2`: Number of continuous thoughts per reasoning step[1]
- `epochs_per_stage = 3`: Training epochs per stage
- `max_latent_stage = 4`: Maximum number of training stages
- `pad_latent_to_max = True`: Padding strategy for consistent latent positioning

AI IMPLEMENTATION NOTE: Critical Stage-Based Training Details
1. **Curriculum Progression**: Each stage replaces more reasoning steps with latent tokens: `n_latent_tokens = current_stage * c_thought`
2. **Latent Token Sequence**: Format is `<|start-latent|>` + `<|latent|>` * n_latent_tokens + `<|end-latent|>`
3. **Token Initialization**: New special tokens must be initialized with existing token embeddings (use "the" token as reference)
4. **Optimizer Reset**: `reset_optimizer=True` between stages is crucial for stable training
5. **Padding Strategy**: `pad_latent_to_max=True` ensures consistent latent positioning across batch items for KV cache efficiency


#### 1.1.3 Data Processing Pipeline

The Coconut data processing follows a specific format derived from codebase analysis[1]:

```python
# Required JSON format
{
    "image_path": "path/to/image.jpg",
    "question": "Step-by-step math problem...",
    "answer": "42",
    "steps": [
        "First, identify the given values",
        "Next, apply the formula",
        "Finally, calculate the result"
    ]
}
```

The `MyCollator` class handles sophisticated padding to ensure consistent latent token positioning across batch items, critical for maintaining reasoning coherence[1].

AI IMPLEMENTATION NOTE: Critical Data Processing Details (Adjusted for InternVL3)
1. **Latent Token Alignment**: The collator finds the latest earliest latent token position across batch and pads all sequences to align latent tokens for KV cache efficiency
2. **Padding Strategy**: Uses `tokenizer.pad_token_id` for input padding and `label_pad_token_id=-100` for label padding
3. **Position IDs**: Do not pad `position_ids` in the collator; generate sequential `position_ids` dynamically in forward (0..seq_len-1) after padding (InternVL3-compatible)
4. **Attention Mask**: Padded positions get attention_mask=0 to prevent attention to padding tokens
5. **Tokenization Format**: Question ends with "\n", steps end with "\n", answer starts with "### " and ends with EOS token

### 1.2 InternVL3-1B Architecture Analysis

#### 1.2.1 Architectural Framework

InternVL3-1B follows the established "ViT-MLP-LLM" paradigm with significant improvements[2][3][8]:

**Vision Component**:

- **InternViT-300M-448px-V2_5**: Vision transformer optimized for 448×448 image tiles
- **Pixel Unshuffle Operation**: Reduces visual tokens by 75% (from 1024 to 256 tokens per tile)
- **Dynamic Resolution Support**: Handles variable image sizes through tiling

**Language Component**:

- **Qwen2.5-0.5B**: Compact but powerful base language model (not instruction-tuned)
- **Native Multimodal Pre-training**: Joint training on vision-language and text-only data[2]

**Fusion Mechanism**:

- **Randomly Initialized MLP Projector**: Maps visual features to language embedding space
- **Standard Position Encoding**: Uses sequential `position_ids` for all tokens

AI IMPLEMENTATION NOTE: Critical InternVL3 Architecture Details from Codebase
1. **Token Replacement Pattern**: `input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)` - MUST zero out original embeddings first
2. **CLS Token Removal**: Always remove first token from ViT output: `vit_embeds = vit_embeds[:, 1:, :]`
3. **Pixel Shuffle Logic**: Critical for reducing tokens - follow exact implementation with ps_version handling
4. **MLP Projector**: Sequential layers: LayerNorm → Linear → GELU → Linear
5. **Dynamic Batching**: `pixel_values` are concatenated across batch, not stacked - shape [total_tiles, 3, 448, 448]
6. **Error Handling**: Always include try-catch for shape mismatches with fallback logic


#### 1.2.2 Native Multimodal Pre-training

InternVL3's key innovation is Native Multimodal Pre-training, which consolidates language and vision learning into a single stage rather than post-hoc adaptation[2]. This approach:

- Interleaves multimodal data with text-only corpora
- Learns linguistic and multimodal representations simultaneously
- Eliminates complex alignment challenges in traditional pipelines


### 1.3 Visual Question Answering Dataset Characteristics

#### 1.3.1 Dataset Structure and Complexity

Visual Question Answering datasets present unique challenges for multimodal reasoning. By using the `datasets` library, we can directly load and process various VQA datasets without manual downloads.

**Common VQA Dataset Characteristics:**

- Require broad commonsense and world knowledge beyond image content
- Questions cannot be answered through simple knowledge base queries
- Multiple reasoning types: visual, logical, commonsense, and factual

**Typical Data Fields:**

- `image`: Visual content (PIL.Image object or image path)
- `question_id`: Unique identifier
- `question`: Natural language question
- `choices`: Multiple choice options (when applicable)
- `correct_choice_idx`: Index of correct answer
- `direct_answers`: Alternative answer formulations
- `rationales`: Human-provided reasoning explanations

**Question Categories:**

- Object recognition: "What objects are in the image?"
- Spatial reasoning: "Where is the object located?"
- Causal reasoning: "Why is this happening?"
- Predictive reasoning: "What will happen next?"
- Knowledge-based: "What is this used for?"


## 2. Comprehensive System Design

### 2.4 Implementation Decisions

- Padding policy: use right-padding everywhere (tokenizer and collator). Do not provide `position_ids` from the collator; generate sequential `position_ids` in forward (0..N-1) after padding. This matches InternVL3 and avoids left-pad complexity.
- Model: `OpenGVLab/InternVL3-1B-Pretrained` on a single GPU, bf16, trust_remote_code=true, use_flash_attn=true when available.
- Dataset policy (A-OKVQA):
  - Multiple-choice only. Skip samples without rationales (to preserve reasoning supervision quality).
  - Single image per question (no multi-image support in this release).
- Vision tiling: dynamic preprocess with max_num=8 tiles (balanced for VRAM/perf); use thumbnail; cap at 8 to avoid diminishing returns.
- Special tokens:
  - Add Coconut (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`) and InternVL (`<img>`, `</img>`, `<IMG_CONTEXT>`) as additional special tokens via `add_special_tokens`.
  - Resize on `language_model` only; initialize new embeddings by average of existing embeddings (input and output if untied). This mirrors Coconut’s neutral init and avoids semantic bias; better than copying a single token.
- Training schedule:
  - Follow Coconut curriculum: c_thought=2, epochs_per_stage=3, max_latent_stage=4, pad_latent_to_max=true, reset_optimizer=true between stages.
  - Optimizer: AdamW; Scheduler: cosine w/ warmup; gradient clipping max_grad_norm=1.0; bf16; gradient checkpointing enabled.
- Generation behavior:
  - Use greedy decoding for evaluation; temperature/top_p optional for demos.
  - Early stop on EOS; max_new_tokens=64 for A-OKVQA; regex-based choice extraction (A–D).
  - Dynamic latent tokens supported via last-hidden reuse path (implemented) with KV cache; no extra passes.
- Logging & checkpoints:
#### WandB and Local Logging Integration

- Initialize wandb with `project` and `run_name` from config. Log per-step loss (smoothed), per-epoch metrics (val accuracy, loss), learning rate, grad norm, and throughput.
- Write local `metrics_epoch_{k}.json` with the same scalars for offline inspection.
- Save checkpoints each epoch as `epoch_{k}.pt` and a `best.pt` when val accuracy improves. Record artifact references in wandb.

#### A-OKVQA Multiple-Choice Formatting

- Choices are rendered as:
  - `A. <choice_0>`\n`B. <choice_1>`\n`C. <choice_2>`\n`D. <choice_3>`
- Prompt includes image tokens (expanded) before question; skip samples without rationales; constrain to single image.
- Answer extraction uses regex `(answer is|Answer:?\s*)([A-D])` (case-insensitive). If no match, default to A and log a warning.

#### Vision Tiling Policy

- Use InternVL3 dynamic tiling with `max_num=8` and `use_thumbnail=True` for balanced cost/benefit.
- Concatenate tiles across batch (not stacked) and set `image_flags` accordingly. Assert `<IMG_CONTEXT>` count equals `tiles * 256` per sample.

  - Integrate Weights & Biases (wandb) and local logging.
  - Single-GPU only (no FSDP). Save epoch-wise checkpoints; keep best by validation accuracy as well.

### 2.1 Unified Architecture Overview

The corrected multimodal Coconut system follows InternVL3's exact token replacement pattern:

```
Input Processing:
[Image] → InternVL3 Vision Encoder → Visual Features (256 tokens/tile)
[Text with <IMG_CONTEXT> tokens] → Tokenizer → Token Sequence

Token-Level Multimodal Fusion (InternVL3 Pattern):
[Token Sequence] → Replace <IMG_CONTEXT> with Visual Features → Multimodal Sequence

Continuous Reasoning:
[Multimodal Sequence] → Coconut Multi-Pass Reasoning → Latent Thoughts
                                                    ↓
[Final Sequence] → Language Model → Generated Answer
```


### 2.2 Core Implementation Components

#### 2.2.1 Multimodal Coconut Architecture

```python
# Required imports for MultimodalCoconut implementation
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer
from collections import namedtuple
from typing import Optional, List, Union, Tuple

# Define output structure following Coconut pattern
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class MultimodalCoconut(nn.Module):
    """Extended Coconut model supporting vision-language continuous reasoning

    Based on analysis of coconut.py and InternVL3 architecture, this implementation
    maintains Coconut's multi-stage reasoning while following InternVL's exact
    token replacement pattern for multimodal fusion.

    AI IMPLEMENTATION NOTE: This class is the core of the multimodal reasoning system.
    Key implementation requirements:
    1. MUST support variable batch sizes (not restricted to batch_size=1)
    2. MUST handle variable number of image tiles per batch item
    3. MUST maintain exact Coconut reasoning logic with proper tensor operations
    4. MUST use InternVL3's exact token replacement pattern for IMG_CONTEXT tokens
    5. All tensor operations must be batch-aware and handle edge cases gracefully

    Critical tensor shapes to maintain:
    - `input_ids`: [batch_size, seq_len]
    - `pixel_values`: [total_tiles_in_batch, 3, 448, 448] (concatenated, not stacked)
    - `image_flags`: [total_tiles_in_batch] (boolean mask for valid tiles)
    - `input_embeds`: [batch_size, seq_len, hidden_size]
    """

    def __init__(self,
                 base_internvl_model,           # Complete InternVL3 model for delegation
                 tokenizer,                     # Tokenizer with special tokens
                 config,                        # Configuration object with model parameters
                 latent_token_id=None,          # Token ID for <|latent|> tokens
                 start_latent_id=None,          # Token ID for <|start-latent|> tokens
                 end_latent_id=None,            # Token ID for <|end-latent|> tokens
                 eos_token_id=None,             # Token ID for end-of-sequence
                 img_context_token_id=None,     # Token ID for <IMG_CONTEXT> tokens
                 **kwargs):
        super().__init__()

        # CRITICAL: Model loading validation following InternVL pattern
        if base_internvl_model is None:
            raise ValueError("base_internvl_model cannot be None")

        if not hasattr(base_internvl_model, 'vision_model'):
            raise ValueError("Invalid InternVL model: missing vision_model component")

        if not hasattr(base_internvl_model, 'language_model'):
            raise ValueError("Invalid InternVL model: missing language_model component")

        if not hasattr(base_internvl_model, 'mlp1'):
            raise ValueError("Invalid InternVL model: missing mlp1 (visual projector) component")

        # Tokenizer validation
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")

        # Configuration validation
        if config is None:
            raise ValueError("config cannot be None")

        # Store configuration and tokenizer
        self.config = config
        self.tokenizer = tokenizer

        # Store reference to base InternVL model for delegation
        self.base_internvl_model = base_internvl_model

        # Extract individual components with error handling
        try:
            self.vision_model = base_internvl_model.vision_model
            self.visual_projector = base_internvl_model.mlp1  # InternVL uses mlp1
            self.language_model = base_internvl_model.language_model
        except AttributeError as e:
            raise ValueError(f"Failed to extract model components: {e}. Ensure base_internvl_model is a valid InternVL model.")

        # Compute token IDs from tokenizer with validation (following Coconut pattern)
        try:
            if latent_token_id is None:
                latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                if latent_token_id == tokenizer.unk_token_id:
                    print("Warning: <|latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")

            if start_latent_id is None:
                start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
                if start_latent_id == tokenizer.unk_token_id:
                    print("Warning: <|start-latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")

            if end_latent_id is None:
                end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
                if end_latent_id == tokenizer.unk_token_id:
                    print("Warning: <|end-latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")

            if eos_token_id is None:
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id is None:
                    raise ValueError("Tokenizer missing eos_token_id. This is required for generation.")

            if img_context_token_id is None:
                img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                if img_context_token_id == tokenizer.unk_token_id:
                    print("Warning: <IMG_CONTEXT> token not found in tokenizer, using UNK token ID. Consider adding InternVL special tokens.")

        except Exception as e:
            raise ValueError(f"Failed to compute token IDs from tokenizer: {e}. Ensure tokenizer has required special tokens.")

        # Coconut reasoning components (from coconut.py analysis)
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id

        # InternVL3 configuration parameters (with proper defaults)
        self.img_context_token_id = img_context_token_id
        self.num_image_token = getattr(config, 'num_image_token', 256)  # Default 256 tokens per tile
        self.downsample_ratio = getattr(config, 'downsample_ratio', 0.5)  # Default 0.5
        self.ps_version = getattr(config, 'ps_version', 'v2')  # Default v2
        self.select_layer = getattr(config, 'select_layer', -1)  # Default -1 (last layer)
        self.template = getattr(config, 'template', 'internvl2_5')  # Conversation template
        self.max_dynamic_patch = getattr(config, 'max_dynamic_patch', 12)  # Max tiles
        self.min_dynamic_patch = getattr(config, 'min_dynamic_patch', 1)   # Min tiles
        self.use_thumbnail = getattr(config, 'use_thumbnail', True)  # Thumbnail processing
        self.pad2square = getattr(config, 'pad2square', False)  # Padding strategy
        self.dynamic_image_size = getattr(config, 'dynamic_image_size', True)  # Dynamic resolution
        self.force_image_size = getattr(config, 'force_image_size', 448)  # Image size

        # Calculate patch size from config
        if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'patch_size'):
            self.patch_size = config.vision_config.patch_size
        else:
            self.patch_size = 14  # Default InternVL3 patch size

        # Generation counter for FSDP synchronization (from Coconut)
        # Critical for distributed training - tracks forward passes across GPUs
        self.gen_forward_cnt = 0

        # Memory optimization features
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            self.vision_model.gradient_checkpointing_enable()
            self.language_model.gradient_checkpointing_enable()

    def extract_feature(self, pixel_values):
        """
        Delegate to InternVL's built-in extract_feature method for visual feature extraction.

        This approach ensures exact compatibility with InternVL's behavior, including proper
        select_layer support, spatial reshaping, ps_version handling, and scale_factor parameters.
        """
        if pixel_values is None:
            return None

        # Delegate directly to the base InternVL model's tested extract_feature method
        # This ensures we get exact InternVL behavior with all edge cases handled correctly
        return self.base_internvl_model.extract_feature(pixel_values)

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                position_ids=None,
                pixel_values=None,
                image_flags=None,
                **kwargs):
        """Forward pass with multimodal continuous reasoning and variable tile support

        Follows InternVL3's exact token replacement pattern with proper handling
        of variable-sized `pixel_values` from dynamic tiling.

        AI IMPLEMENTATION NOTE: This is the main forward method. Critical requirements:
        1. MUST handle batch processing with variable image tile counts per batch item
        2. Input tensor shapes:
           - `input_ids`: [batch_size, seq_len]
           - `pixel_values`: [total_tiles_in_batch, 3, 448, 448] (concatenated across batch)
           - `image_flags`: [total_tiles_in_batch] (boolean mask for valid tiles)
        3. MUST follow InternVL3's exact IMG_CONTEXT token replacement pattern
        4. MUST maintain Coconut's continuous reasoning logic with proper batching
        5. Error handling for mismatched tensor shapes is essential
        """

        # Step 1: Get initial text embeddings (following InternVL3)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Step 2: Process visual inputs using InternVL's native image_flags mechanism
        if pixel_values is not None and image_flags is not None:
            try:
                # Validate input shapes
                if len(pixel_values.shape) != 4:
                    raise ValueError(f"pixel_values must be 4D tensor [total_tiles, 3, 448, 448], got shape {pixel_values.shape}")

                if len(image_flags.shape) != 1:
                    raise ValueError(f"image_flags must be 1D tensor [total_tiles], got shape {image_flags.shape}")

                if pixel_values.shape[0] != image_flags.shape[0]:
                    raise ValueError(f"pixel_values and image_flags must have same first dimension, got {pixel_values.shape[0]} vs {image_flags.shape[0]}")

                # Process all tiles through vision encoder (InternVL approach)
                vit_embeds = self.extract_feature(pixel_values)  # [total_tiles, num_image_token, hidden_size]

                if vit_embeds is None:
                    print("Warning: extract_feature returned None, skipping visual processing")
                    return self._coconut_continuous_reasoning(input_embeds, input_ids, attention_mask, labels, position_ids)

                # CRITICAL: Use InternVL's native image_flags filtering mechanism
                # This elegantly handles variable tile counts, text-only samples, and padding
                image_flags = image_flags.squeeze(-1)  # Remove any extra dimensions
                vit_embeds = vit_embeds[image_flags == 1]  # Filter out padding/text-only patches

            except Exception as e:
                print(f"Error in visual feature extraction: {e}. Skipping visual processing.")
                return self._coconut_continuous_reasoning(input_embeds, input_ids, attention_mask, labels, position_ids)

            # Apply InternVL's exact token replacement pattern
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            # Find IMG_CONTEXT token positions
            selected = (input_ids_flat == self.img_context_token_id)
            # Sanity check: ensure replacement counts match available visual tokens
            if selected.sum().item() != vit_embeds.reshape(-1, C).shape[0]:
                print(
                    f"warning: IMG_CONTEXT count ({selected.sum().item()}) != vit tokens ({vit_embeds.reshape(-1, C).shape[0]}), applying safe truncation"
                )

            if selected.any():
                try:
                    # CRITICAL: Use InternVL's exact pattern to preserve gradient flow
                    # Mathematical operations preserve gradients, direct assignment breaks them
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                except Exception as e:
                    # Handle shape mismatch with InternVL's fallback pattern
                    vit_embeds_flat = vit_embeds.reshape(-1, C)
                    n_token = selected.sum()
                    print(f'warning: {e}, selected.sum()={n_token}, vit_embeds.shape={vit_embeds.shape}')
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat[:n_token]

            # Reshape back to batch format
            input_embeds = input_embeds.reshape(B, N, C)

# Step 3: Generate `position_ids` (following InternVL3's actual implementation)
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
# Standard sequential `position_ids` (proven InternVL approach)
            # Note: Despite V2PE being mentioned in InternVL3 paper, the actual codebase
            # uses standard sequential position encoding, which works excellently in practice
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Step 4: Apply Coconut's continuous reasoning to the multimodal sequence
        return self._coconut_continuous_reasoning(
            input_embeds, input_ids, attention_mask, labels, position_ids
        )

    def _coconut_continuous_reasoning(self, inputs_embeds, input_ids, attention_mask, labels, position_ids):
        """Implements Coconut's multi-pass continuous reasoning with proper chained latent token handling

        This method follows the original Coconut approach, using iterative multi-pass processing
        to handle chained latent tokens correctly. Each latent token's replacement uses hidden states that
        include the effects of all previously processed latent tokens, preserving dependency chains and
        enabling emergent BFS-like reasoning patterns.

        AI IMPLEMENTATION NOTE: Critical Multi-Pass Requirements from Original Coconut
        1. **Iterative Processing**: Process one latent token per pass to preserve dependencies
        2. **KV Cache Reuse**: Efficiently reuse computation from previous passes
        3. **Sequential Replacement**: Each latent uses hidden states from updated embeddings
        4. **Range Management**: Track computation ranges to minimize redundant forward passes
        5. **Dependency Preservation**: Maintain the chain of continuous thoughts for BFS reasoning

        **Concrete Example of the Fix**:
        For sequence: `[Question] <|latent|> <|latent|> [Answer]`

        ❌ **Old Broken Approach** (Single Pass):
        - Pass 1: Replace BOTH latents with hidden_state[pos-1] from SAME forward pass
        - Result: Second latent uses hidden state that doesn't include first latent's effect

        ✅ **New Correct Approach** (Multi-Pass):
        - Pass 1: Replace first latent with hidden_state[pos-1], then forward pass
        - Pass 2: Replace second latent with hidden_state[pos-1] from NEW forward pass
        - Result: Second latent uses hidden state that INCLUDES first latent's effect

        This dependency chain is crucial for BFS-like reasoning and multimodal iterative refinement.
        """

        # Find latent token positions (following original Coconut)
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        if len(latent_indices) == 0:
            # No latent tokens, do standard forward pass
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )
            return {"loss": outputs.loss, "logits": outputs.logits, "inputs_embeds": inputs_embeds}

        # Organize latent positions by batch (exact Coconut logic)
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        # Initialize computation range (before earliest latent token)
        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        logits = []

        # Multi-pass iterative processing with corrected range management
        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # First forward pass - compute up to first latent token
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = 0
            else:
                # Subsequent passes - reuse KV cache for efficiency
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Replace only ONE latent token per pass (preserving dependencies)
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            if filling_indices:
                # Convert to list of lists to avoid in-place operations (exact Coconut approach)
                tensor_list = [
                    [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                # Replace latent tokens with continuous thoughts
                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair

                    # Use hidden state from preceding position with correct offset (keep grads in training)
                    if token_idx > 0:
                        tensor_list[batch_idx][token_idx] = hidden_states[
                            batch_idx, token_idx - 1 - hidden_states_offset, :
                        ]

                # Reassemble inputs_embeds (exact Coconut approach)
                inputs_embeds = torch.stack([
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ])

            # CORRECTED: Update computation range for next pass
            # Need to compute up to the position BEFORE the next latent token we want to replace
            if pass_idx + 1 < max_n_latents:
                # Find the next latent token position across all batch items
                next_latent_positions = []
                for batch_idx, mask_list in enumerate(latent_lists):
                    if len(mask_list) > pass_idx + 1:
                        next_latent_positions.append(mask_list[pass_idx + 1])

                if next_latent_positions:
                    # Compute up to the position before the earliest next latent
                    next_latent_pos = min(next_latent_positions)
                    next_compute_range = (next_compute_range[1], next_latent_pos)
                else:
                    # No more latents to process, go to end
                    next_compute_range = (next_compute_range[1], input_ids.shape[1])
            else:
                # Last pass, compute to end
                next_compute_range = (next_compute_range[1], input_ids.shape[1])

        # Final pass to compute logits with all latent tokens processed
        if kv_cache is not None:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
        else:
            past_key_values = None

        final_outputs = self.language_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=past_key_values,
            labels=labels[:, next_compute_range[0]:next_compute_range[1]] if labels is not None else None,
            output_hidden_states=True,
        )

        # Combine logits from all passes (if needed for loss computation)
        if len(logits) > 0:
            # For training, we typically only need the final logits
            combined_logits = final_outputs.logits
        else:
            combined_logits = final_outputs.logits

        # CRITICAL: Update FSDP synchronization counter (from original Coconut)
        # This tracks forward passes for distributed training synchronization
        self.gen_forward_cnt += max_n_latents + 1

        return {
            "loss": final_outputs.loss,
            "logits": combined_logits,
            "inputs_embeds": inputs_embeds
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        synced_gpus=False,  # CRITICAL: FSDP synchronization parameter
        **kwargs
    ):
        """Generate method with proper dynamic latent token handling

        This method implements true Coconut-style generation that can handle latent tokens
        generated during autoregressive decoding, not just pre-filled ones.

        AI IMPLEMENTATION NOTE: Critical Generation Requirements from Coconut Codebase
        1. **Batch Size Extension**: Original Coconut only supports batch_size=1 - MUST extend for multimodal batching
        2. **Dynamic Latent Handling**: MUST handle latent tokens generated during autoregressive process, not just pre-filled
        3. **KV Cache Management**: Essential for efficiency - maintain past_key_values throughout generation
        4. **EOS Token Handling**: Check for EOS token to stop generation per batch item
        5. **FSDP Synchronization**: Use gen_forward_cnt for distributed training synchronization
        6. **Embedding Concatenation**: New tokens must be converted to embeddings and concatenated properly

        Unlike original Coconut which has `assert input_ids.shape[0] == 1`, this implementation
        MUST support full batch processing for efficient A-OKVQA evaluation and deployment.

        The _coconut_autoregressive_generate method implements comprehensive generation capabilities:
        1. **Token Appending**: Properly appends generated tokens to sequence tracking
        2. **EOS Handling**: Implements early stopping when EOS tokens are generated
3. **State Updates**: Updates `attention_mask` and `position_ids` during generation
        4. **Latent Processing**: Maintains dynamic latent token replacement during autoregressive decoding
        5. **Output Reconstruction**: Combines original and generated tokens for final output

        The method implements batched generation with per-item state tracking,
        dynamic latent handling, and variable-length output support following the original
        Coconut approach but extended for multimodal batch processing.
        """

        # Step 1: Preprocess inputs (following InternVL3's approach)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Step 2: Process visual inputs if present (InternVL3 approach)
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)

            # Replace IMG_CONTEXT tokens with visual embeddings (exact InternVL3 logic)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            # Find IMG_CONTEXT token positions
            selected = (input_ids_flat == self.img_context_token_id)

            # Replace with visual embeddings
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                # Handle shape mismatch (InternVL3 fallback)
                vit_embeds = vit_embeds.reshape(-1, C)
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

            # Reshape back
            input_embeds = input_embeds.reshape(B, N, C)

# Step 3: Generate standard `position_ids`
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Step 4: Process any existing latent tokens in the input
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        if len(latent_indices) > 0:
            # Process existing latent tokens using the continuous reasoning method
            processed_outputs = self._coconut_generate_reasoning(
                input_embeds, input_ids, attention_mask, position_ids
            )
            input_embeds = processed_outputs["inputs_embeds"]

        # Step 5: Dynamic generation with latent token handling
        generated_outputs = self._coconut_autoregressive_generate(
            input_embeds=input_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

        # Step 6: CRITICAL FSDP Synchronization with Batch-Aware Logic
        # Properly handle batch_size > 1 with variable latent counts per batch item
        if synced_gpus:
            # Calculate actual maximum latent tokens needed for this batch
            max_latents_in_batch = 0
            if len(input_ids.shape) > 1:
                for batch_idx in range(input_ids.shape[0]):
                    # Count latent tokens in each batch item
                    latent_count = (input_ids[batch_idx] == self.latent_token_id).sum().item()
                    max_latents_in_batch = max(max_latents_in_batch, latent_count)

            # Synchronize maximum latent count across all GPUs using distributed communication
            # This ensures all GPUs know the global maximum and perform the same number of passes
            if torch.distributed.is_initialized():
                max_latents_tensor = torch.tensor(max_latents_in_batch, device=input_ids.device)
                torch.distributed.all_reduce(max_latents_tensor, op=torch.distributed.ReduceOp.MAX)
                max_latents_in_batch = max_latents_tensor.item()

            # Add safety margin for latent tokens (from original Coconut approach)
            MAX_N_LATENT_SAFETY = 8  # Safety margin as in original
            target_forward_count = max_new_tokens + max_latents_in_batch + MAX_N_LATENT_SAFETY

            # Perform dummy forward passes to synchronize with other GPUs
            # This ensures FSDP doesn't desync during distributed training with variable batch content
            while self.gen_forward_cnt < target_forward_count:
                self.gen_forward_cnt += 1
                # Dummy forward pass to maintain synchronization
                # Use the last token embedding to avoid shape issues
                dummy_embeds = input_embeds[:, -1:, :] if input_embeds.shape[1] > 0 else input_embeds
                _ = self.language_model(inputs_embeds=dummy_embeds)

        return generated_outputs

    def _coconut_generate_reasoning(self, inputs_embeds, input_ids, attention_mask, position_ids):
        """Process existing latent tokens in input sequence using proper multi-pass approach

        This is a generation-specific version of _coconut_continuous_reasoning that:
        1. Doesn't compute loss (no labels needed)
        2. Uses the same multi-pass logic as training for consistency
        3. Returns processed embeddings ready for autoregressive generation
        4. Maintains proper KV cache handling throughout
        """

        # Find latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        if len(latent_indices) == 0:
            # No latent tokens, just do initial forward pass and return
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True
            )
            return {"inputs_embeds": inputs_embeds, "past_key_values": outputs.past_key_values}

        # Use the same multi-pass logic as training for consistency
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None

        # Multi-pass processing (same as training)
        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = next_compute_range[0]

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Replace latent tokens (same logic as training)
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            if filling_indices:
                tensor_list = [
                    [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair
                    if token_idx > 0:
                        tensor_list[batch_idx][token_idx] = hidden_states[
                            batch_idx, token_idx - 1 - hidden_states_offset, :
                        ]

                inputs_embeds = torch.stack([
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ])

        # Final forward pass to get complete KV cache
        if kv_cache is not None:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
        else:
            past_key_values = None

        final_outputs = self.language_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )

        return {"inputs_embeds": inputs_embeds, "past_key_values": final_outputs.past_key_values}

    def _coconut_autoregressive_generate(
        self,
        input_embeds,
        input_ids,
        attention_mask,
        position_ids,
        max_new_tokens=100,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        **kwargs
    ):
        """Efficient autoregressive generation with proper KV cache usage

        This implements the corrected approach that maintains KV cache across generation steps,
        following both Coconut and InternVL best practices for efficiency and correctness.

        Key improvements:
        1. Proper KV cache initialization and reuse
        2. Only process new tokens after initial forward pass
        3. Maintain latent token dependencies through cached states
        4. O(1) computation per step after prefix processing
        5. Batch-aware processing with per-item completion tracking
        """

        # CRITICAL: Reset generation counter for FSDP synchronization (from original Coconut)
        self.gen_forward_cnt = 0

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track generated tokens and completion status for each batch item
        generated_tokens = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Step 1: Initial forward pass to establish KV cache (following Coconut approach)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=True  # Critical: establish KV cache
        )

        # Initialize KV cache for efficient subsequent passes
        past_key_values = outputs.past_key_values
        # Track last hidden state per batch item for dynamic latent handling
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # Increment generation counter for FSDP synchronization
        self.gen_forward_cnt += 1

        # Step 2: Autoregressive generation with KV cache reuse
        current_attention = attention_mask
        current_pos_ids = position_ids

        for step in range(max_new_tokens):
            # Stop if all sequences are finished
            if torch.all(finished):
                break

            # Sample next tokens from current logits
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            if do_sample:
                if temperature != 1.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # Update completion status and collect generated tokens
            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token = next_tokens[batch_idx].item()
                    if token == self.eos_token_id:
                        finished[batch_idx] = True
                    else:
                        generated_tokens[batch_idx].append(token)

            # Stop if all sequences finished
            if torch.all(finished):
                break

            # Step 3: Build next-step embeddings using last-hidden reuse for dynamic latents
            is_latent = (next_tokens == self.latent_token_id)
            token_embeds = self.language_model.get_input_embeddings()(next_tokens)
            # Combine per item: latent uses last_hidden, non-latent uses token embedding
            next_step_embeds = torch.where(
                is_latent.view(-1, 1),
                last_hidden,  # reuse last hidden state
                token_embeds,
            )

# Update `attention_mask` and `position_ids` for new token
            new_attention = torch.ones(batch_size, 1, device=device)
            current_attention = torch.cat([current_attention, new_attention], dim=1)
            new_pos_ids = current_pos_ids[:, -1:] + 1
            current_pos_ids = torch.cat([current_pos_ids, new_pos_ids], dim=1)

            # Single batched forward with mixed latent/non-latent embeddings
            outputs = self.language_model(
                inputs_embeds=next_step_embeds.unsqueeze(1),
                attention_mask=current_attention,
                position_ids=new_pos_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )

            # Update KV cache and last hidden for next iteration
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # CRITICAL: Increment generation counter for FSDP synchronization
            self.gen_forward_cnt += 1

    def _process_dynamic_latents(self, current_embeds, new_latent_positions, attention_mask, position_ids):
        """Process dynamically generated latent tokens with multi-pass logic

        This method handles latent tokens that are generated during autoregressive decoding,
        ensuring proper dependency chains are maintained for BFS-like reasoning.

        Args:
            current_embeds: Full sequence embeddings including new latent tokens
            new_latent_positions: Boolean mask indicating which positions are latent tokens
            attention_mask: Full attention mask for the sequence
position_ids: `position_ids` for the sequence

        Returns:
            Processed embeddings with latent tokens replaced by continuous thoughts
        """

        # Find positions of new latent tokens
        latent_positions = []
        for batch_idx in range(new_latent_positions.shape[0]):
            if new_latent_positions[batch_idx]:
                # This batch item has a new latent token at the last position
                latent_positions.append((batch_idx, current_embeds.shape[1] - 1))

        if not latent_positions:
            return current_embeds

        # Process each latent token with proper multi-pass logic
        # For dynamically generated latents, we need to ensure they see the effects
        # of all previous tokens in the sequence, including other latents

        processed_embeds = current_embeds.clone()

        # Perform a forward pass to get hidden states for latent replacement
        outputs = self.language_model(
            inputs_embeds=processed_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False  # Don't use cache for this processing step
        )

        hidden_states = outputs.hidden_states[-1]

        # Replace latent tokens with hidden states from preceding positions
        for batch_idx, token_idx in latent_positions:
            if token_idx > 0:
                # In generation, it's safe to detach to avoid graph growth
                processed_embeds[batch_idx, token_idx, :] = hidden_states[batch_idx, token_idx - 1, :].detach()

        return processed_embeds

    def _reconstruct_embeddings_from_cache(self, past_key_values, seq_len):
        """Reconstruct embeddings from KV cache context

        This is a simplified reconstruction - in practice, we maintain the embeddings
        separately to avoid this expensive operation.

        Args:
            past_key_values: KV cache from previous forward passes
            seq_len: Length of sequence to reconstruct

        Returns:
            Reconstructed embeddings (placeholder implementation)
        """

        # This is a placeholder implementation
        # In practice, we would maintain the embeddings separately
        # to avoid expensive reconstruction from KV cache

        batch_size = past_key_values[0][0].shape[0] if past_key_values else 1
        hidden_size = self.language_model.config.hidden_size
        device = past_key_values[0][0].device if past_key_values else torch.device('cpu')

        # Return zero embeddings as placeholder
        # In a full implementation, we would maintain embeddings separately
        return torch.zeros(batch_size, seq_len, hidden_size, device=device)

        # Step 5: Reconstruct output sequences
        batch_outputs = []
        for batch_idx in range(batch_size):
            # Combine original input with generated tokens
            original_tokens = input_ids[batch_idx].tolist()
            new_tokens = generated_tokens[batch_idx]
            full_sequence = original_tokens + new_tokens
            batch_outputs.append(full_sequence)

        # Convert to tensor format
        max_len = max(len(seq) for seq in batch_outputs)
        padded_sequences = []

        for seq in batch_outputs:
            # Pad sequences to same length
            padded = seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
            padded_sequences.append(padded)

        generated_ids = torch.tensor(padded_sequences, device=device, dtype=torch.long)

        # Return in the expected format (unified, no conflicting branches)
        return {
            'sequences': generated_ids,
            'scores': None,  # Optional: add generation scores if needed
            'past_key_values': past_key_values
        }
```


#### 2.2.2 Data Processing with InternVL3 Token Pattern

The key insight is that InternVL3 handles multimodal fusion through direct token replacement, not separate fusion modules. The data processor must create the correct token sequence with `<IMG_CONTEXT>` tokens that will be replaced with visual embeddings:

AI IMPLEMENTATION NOTE: Critical Token Pattern Requirements from InternVL3 Codebase
1. **Image Token Replacement**: `<image>` placeholder MUST be replaced with `<img>` + (`<IMG_CONTEXT>` × num_image_token × num_patches) + `</img>` (literal tokens; conceptually the start/context/end tokens)
2. **Dynamic Tile Calculation**: `num_patches` comes from dynamic preprocessing - varies per image based on aspect ratio
3. **Token Order**: IMG_START → (IMG_CONTEXT × 256 × num_tiles) → IMG_END for each image
4. **Batch Processing**: Each batch item can have different numbers of tiles - handle with num_patches_list
5. **Tokenization Timing**: Replace `<image>` with actual tokens BEFORE tokenization, not after
6. **Context Token ID**: MUST call `tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")` to get the correct token ID
7. **Template Integration**: Follow InternVL3's conversation template system for proper prompt formatting

```python
def create_multimodal_prompt(self, question, choices, num_patches, image_present=True):
    """Create prompt following InternVL3's exact token pattern

    AI IMPLEMENTATION NOTE: Critical Prompt Creation Requirements
    1. **Token Order**: Must be IMG_START + (IMG_CONTEXT × 256 × num_patches) + IMG_END
    2. **Dynamic Patches**: num_patches varies per image based on dynamic preprocessing
    3. **Image Placement**: Image tokens must come BEFORE question text
    4. **Token Calculation**: Total IMG_CONTEXT tokens = 256 * num_patches per image
    5. **Multiple Choice Format**: Use A, B, C, D format matching A-OKVQA structure
    """

    # Format multiple choice question
    choices_formatted = "\n".join([
        f"{chr(65+i)}. {choice}"
        for i, choice in enumerate(choices)
    ])

    # Create base prompt
    base_prompt = (
        f"Question: {question}\n\n"
        f"Choices:\n{choices_formatted}\n\n"
        f"MUST answer with just one of: A, B, C, or D.\n"
        f"Let me think step by step:\n"
    )

    if image_present:
        # Insert image tokens following InternVL3 pattern
        # <image> gets replaced with <img><IMG_CONTEXT>...<IMG_CONTEXT></img>
        # Use the correct pattern: IMG_START + (IMG_CONTEXT * num_image_token * num_patches) + IMG_END
        image_tokens = (
            "<img>" + "<IMG_CONTEXT>" * (self.config.data.num_image_token * num_patches) + "</img>"
        )
        full_prompt = image_tokens + "\n" + base_prompt
    else:
        full_prompt = base_prompt

    return full_prompt
```


### 2.3 VQA Data Processing Pipeline (Hugging Face Integration)

#### 2.3.1 Multimodal Data Processor

```python
import os
import torch

class MultimodalProcessor:
    """Processes VQA data for multimodal continuous reasoning training

    Based on Coconut's data processing approach extended for visual inputs

    AI IMPLEMENTATION NOTE: Critical Data Processing Requirements
    1. **Dynamic Tiling**: MUST use InternVL3's dynamic preprocessing to handle variable image sizes
    2. **Token Replacement**: Replace `<image>` with actual image tokens BEFORE tokenization
    3. **Stage-Based Processing**: Apply Coconut's curriculum learning with proper latent token insertion
    4. **Batch Compatibility**: Ensure all outputs are compatible with variable tile batching
    5. **Label Supervision**: Only supervise reasoning and answer portions, not question/image tokens
    6. **Image Preprocessing**: Follow InternVL3's exact transform pipeline: RGB conversion → Resize → ToTensor → Normalize
    7. **Aspect Ratio Handling**: Use dynamic preprocessing to determine optimal tile configuration per image
    """

    def __init__(self, tokenizer, image_processor, config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

        # Add Coconut's special tokens
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        num_added = self.tokenizer.add_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")

        # Get token IDs (exact Coconut approach)
        self.start_latent_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_latent_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio for dynamic tiling (InternVL3 implementation)"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    # Use InternVL3's built-in image processing pipeline for dynamic preprocessing

    def load_image_with_dynamic_tiling(self, image, input_size=448, max_num=12):
        """Use InternVL3's actual dynamic preprocessing functions

        This uses InternVL3's proven dynamic_preprocess and build_transform functions
        for optimal image processing and tiling.
        """
         # Import InternVL3's actual functions
         # Prefer internvl.train.dataset; fall back to internvl_chat path if needed.
         try:
             from internvl.train.dataset import dynamic_preprocess, build_transform
         except Exception:
             from internvl_chat.internvl.train.dataset import dynamic_preprocess, build_transform

        # Use InternVL3's dynamic preprocessing for proper tiling
        images = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num
        )

        # Use InternVL3's build_transform function
        transform = build_transform(input_size=input_size)

        # Apply transforms to all tiles
        pixel_values = [transform(tile) for tile in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values

    def process_sample(self, sample, stage=0):
        """Convert VQA sample from Hugging Face to multimodal Coconut training format

        Args:
            sample: VQA data sample from Hugging Face `datasets`.
            stage: Current training stage (0 to max_latent_stage)

        Returns:
            Processed sample ready for multimodal continuous reasoning
        """
        # Load image from the 'image' field provided by Hugging Face `datasets`
        # and save it to a local path for reference.
        image = sample['image'].convert('RGB')
        image_id = sample.get('image_id', sample.get('question_id', 'unknown'))
        image_path = os.path.join(self.config.image_data_path, f"{image_id}.jpg")
        image.save(image_path)
        sample['image_path'] = image_path

        # Use InternVL3's dynamic preprocessing for proper tiling
        pixel_values = self.load_image_with_dynamic_tiling(
            image,
            input_size=448,
            max_num=12  # Allow up to 12 tiles for high-resolution images
        )

        # Get the actual number of tiles for correct token calculation
        num_patches = pixel_values.shape[0]

        # Format question with multiple choice options
        question_text = sample['question']
        choices_formatted = "\n".join([
            f"{chr(65+i)}. {choice}"
            for i, choice in enumerate(sample['choices'])
        ])

        # Create reasoning prompt WITH <image> placeholder (CRITICAL for InternVL3)
        full_question = (
            f"<image>\n"  # Essential: InternVL3 requires <image> placeholder
            f"Question: {question_text}\n\n"
            f"Choices:\n{choices_formatted}\n\n"
            f"Let me think step by step:\n"
        )

        # Extract reasoning steps from rationales (dataset-specific handling)
        reasoning_steps = sample.get('rationales', sample.get('steps', []))
        if not reasoning_steps:
            # Fallback: generate basic reasoning structure
            reasoning_steps = [
                "I need to analyze the image carefully",
                "I should consider what I see and apply relevant knowledge",
                "I can now determine the most appropriate answer"
            ]

        # Generate answer text (handle different answer formats)
        if 'correct_choice_idx' in sample and 'choices' in sample:
            # Multiple choice format
            correct_idx = sample['correct_choice_idx']
            correct_answer = sample['choices'][correct_idx]
            answer_text = f"The answer is {chr(65+correct_idx)}. {correct_answer}"
        elif 'answer' in sample:
            # Direct answer format
            answer_text = f"The answer is {sample['answer']}"
        else:
            # Fallback
            answer_text = "The answer is unknown"

        # Apply Coconut's stage-based training with visual-aware positioning
        processed_reasoning = self._apply_coconut_stage_training(reasoning_steps, stage,
                                                                has_visual_context=True)

        # Combine all components
        full_text = full_question + processed_reasoning + "\n" + answer_text

        # CRITICAL: Replace <image> with InternVL3's image tokens BEFORE tokenization
        # This follows InternVL3's exact approach from chat() method
        # num_patches now correctly reflects the actual number of tiles from dynamic preprocessing
        image_tokens = (
            "<img>" + "<IMG_CONTEXT>" * (self.config.data.num_image_token * num_patches) + "</img>"
        )
        full_text = full_text.replace('<image>', image_tokens, 1)

        # Tokenize text (now with proper image tokens)
        tokenized = self.tokenizer(
            full_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.data.max_sequence_length
        )

        # Create labels (supervise only reasoning and answer)
        # Need to recalculate question tokens after image token replacement
        question_with_image_tokens = full_question.replace('<image>', image_tokens, 1)
        question_tokens = len(self.tokenizer(question_with_image_tokens)['input_ids'])
        labels = tokenized['input_ids'].clone()
        labels[:, :question_tokens] = -100  # Don't supervise question

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'pixel_values': pixel_values,
            'image_flags': torch.tensor([1] * num_patches, dtype=torch.long),  # InternVL's native mechanism
            # passing choices for answer extraction
            'choices': sample.get('choices', None)
        }

    def _apply_coconut_stage_training(self, reasoning_steps, stage, has_visual_context=False):
        """Apply Coconut's exact stage-based curriculum learning with visual-aware adaptations

        Based on the curriculum approach from coconut codebase analysis, enhanced for multimodal reasoning

        AI IMPLEMENTATION NOTE: Critical Stage Training Requirements from Coconut Codebase
        1. **Stage Progression**: Stage 0 = full CoT, Stage N = replace first N reasoning steps with latent tokens
        2. **Latent Token Count**: n_latent_tokens = stage * c_thought (typically stage * 2)
        3. **Token Sequence Format**: `<|start-latent|>` + `<|latent|>` * n_latent_tokens + `<|end-latent|>`
        4. **Visual Bridge**: Add explicit visual grounding cues before latent sequences to maintain visual-text coherence
        5. **Enhanced Visual Continuity**: Ensure remaining text steps bridge properly from latent to explicit reasoning
        6. **Curriculum Balance**: Maintain reasoning quality while progressively increasing latent usage
        """
        if stage == 0:
            # Stage 0: Full chain-of-thought reasoning
            if has_visual_context:
                # Add explicit visual grounding cues for multimodal reasoning
                enhanced_steps = []
                for i, step in enumerate(reasoning_steps):
                    if i == 0:
                        enhanced_steps.append(f"Looking at the image, {step.lower()}")
                    else:
                        enhanced_steps.append(step)
                return "\n".join(enhanced_steps)
            return "\n".join(reasoning_steps)

        # Calculate latent replacement based on Coconut's methodology
        max_latent_stage = min(self.config.max_latent_stage, len(reasoning_steps))
        current_stage = min(stage, max_latent_stage)

        # Number of steps to replace with latent thoughts
        n_latent_steps = current_stage
        n_latent_tokens = n_latent_steps * self.config.c_thought

        # Enhanced visual grounding preservation for multimodal reasoning
        # Create visual bridge to maintain explicit visual grounding throughout latent progression
        if has_visual_context:
            # Add visual bridge before latent sequence to maintain visual grounding
            visual_bridge = "Analyzing the visual information:"
            latent_sequence = (
                visual_bridge + "\n" +
                f"<|start-latent|>" +
                "<|latent|>" * n_latent_tokens +
                f"<|end-latent|>"
            )
        else:
            # Standard Coconut format for text-only reasoning
            latent_sequence = (
                f"<|start-latent|>" +
                "<|latent|>" * n_latent_tokens +
                f"<|end-latent|>"
            )

        # Keep remaining reasoning steps as text
        remaining_steps = reasoning_steps[n_latent_steps:]

        # Enhanced visual grounding for remaining text steps
        if has_visual_context and remaining_steps:
            # Ensure visual continuity in remaining steps
            enhanced_remaining = []
            for i, step in enumerate(remaining_steps):
                if i == 0:
                    # First remaining step should bridge from latent to explicit reasoning
                    if not any(visual_cue in step.lower()
                              for visual_cue in ['image', 'see', 'visual', 'shown', 'picture', 'observe']):
                        enhanced_remaining.append(f"From what I can see, {step.lower()}")
                    else:
                        enhanced_remaining.append(step)
                else:
                    enhanced_remaining.append(step)
            remaining_steps = enhanced_remaining

        text_reasoning = "\n".join(remaining_steps) if remaining_steps else ""

        # Combine latent and text reasoning with proper visual grounding
        if text_reasoning:
            return latent_sequence + "\n" + text_reasoning
        else:
            return latent_sequence
```


#### 2.3.2 Multimodal Data Collator

```python
import torch
from torch.nn.utils.rnn import pad_sequence

class MultimodalCoconutCollator:
    """Custom collator for multimodal continuous reasoning with variable tile support

    Extends Coconut's MyCollator to handle visual inputs with dynamic tiling,
    following InternVL3's exact approach for variable-sized `pixel_values` batching.

    AI IMPLEMENTATION NOTE: Critical Collator Requirements from Coconut + InternVL
    1. **Latent Token Alignment**: Find latest earliest latent position and pad all sequences to align
    2. **Padding Order**: Apply latent alignment BEFORE standard sequence padding
3. **Position IDs**: Do not manually pad `position_ids` - generate dynamically in forward()
    4. **Image Flags Concatenation**: Use torch.cat() for both `pixel_values` and `image_flags`
    5. **InternVL Native Mechanism**: Use `image_flags` instead of num_patches_list for elegant batch processing

**Position IDs Handling**: The original Coconut approach of padding `position_ids` with zeros
    creates duplicate position 0s: [0,0,0,0,1,2,3,...] which confuses attention mechanisms.
InternVL3 always generates `position_ids` dynamically as `torch.arange(0, seq_len)` per sample after padding.
    This approach follows InternVL3's method for proper positional encoding.
    """

    def __init__(self, tokenizer, latent_id, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.latent_id = latent_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        """Collate batch with multimodal inputs and variable tile support

        Implements proper handling of variable-sized `pixel_values` using InternVL3's
        concatenation approach instead of stacking.
        """
        # Extract components
        input_ids = [item['input_ids'].squeeze() for item in batch]
        attention_masks = [item['attention_mask'].squeeze() for item in batch]
        labels = [item['labels'].squeeze() for item in batch]
        pixel_values_list = [item['pixel_values'] for item in batch]
        image_flags_list = [item['image_flags'] for item in batch]

        # Apply Coconut's latent token padding logic (exact implementation)
        earliest_latents = [
            ids.tolist().index(self.latent_id)
            for ids in input_ids
            if (ids == self.latent_id).any()
        ]

        if earliest_latents:
            latest_earliest_latent = max(earliest_latents)

            # Pad to align latent tokens (critical for continuous reasoning)
            for i, ids in enumerate(input_ids):
                if (ids == self.latent_id).any():
                    n_pad = latest_earliest_latent - ids.tolist().index(self.latent_id)
                    if n_pad > 0:
                        # Apply padding
                        pad_tokens = torch.full((n_pad,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                        input_ids[i] = torch.cat([pad_tokens, ids])

                        attention_masks[i] = torch.cat([
                            torch.zeros(n_pad, dtype=attention_masks[i].dtype),
                            attention_masks[i]
                        ])

                        labels[i] = torch.cat([
                            torch.full((n_pad,), self.label_pad_token_id, dtype=labels[i].dtype),
                            labels[i]
                        ])

                        # Do not manually pad position_ids with zeros
                        # This would create duplicate position 0s: [0,0,0,0,1,2,3,...]
# Instead, `position_ids` will be generated dynamically in forward()
                        # position_ids[i] = torch.cat([
                        #     torch.zeros(n_pad, dtype=position_ids[i].dtype),
                        #     position_ids[i]
                        # ])
                        pass  # Skip position_ids padding - handled in forward()

        # Standard sequence padding (ensure consistent padding side across tokenization)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
# Do not pad `position_ids` manually - they will be generated dynamically in forward()
# position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)

        # Handle variable-sized pixel_values and image_flags using InternVL3's concatenation approach
        # This follows InternVL's native mechanism for batch processing
        try:
            pixel_values = torch.cat(pixel_values_list, dim=0)  # Concatenate all tiles
            image_flags = torch.cat(image_flags_list, dim=0)    # Concatenate all flags
        except Exception as e:
            # Fallback for edge cases (e.g., None values)
            pixel_values = None
            image_flags = None
            print(f"Warning: Failed to concatenate pixel_values/image_flags: {e}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_flags': image_flags,  # InternVL's native batch processing mechanism
            # position_ids removed - generated dynamically in forward()
            # 'position_ids': position_ids
            # pass-through for evaluation robustness
            'choices': [item.get('choices', None) for item in batch]
        }
```


## 3. Project Structure and Implementation

### 3.1 Directory Structure

Following the original Coconut project structure for consistency and maintainability:

```
multicoco/
├── README.md
├── requirements.txt
├── setup.py
├── run.py                          # Main orchestrator script (like Coconut)
├── multicoco/
│   ├── __init__.py
│   ├── model.py                    # MultimodalCoconut implementation
│   ├── dataset.py                  # VQA data processing
│   ├── collator.py                 # Multimodal data collation
│   ├── trainer.py                  # Training pipeline
│   ├── evaluator.py                # Evaluation utilities
│   └── utils.py                    # Helper functions
├── configs/
│   ├── base.yaml                   # Base configuration
│   └── aokvqa.yaml                 # A-OKVQA specific config
├── checkpoints/                    # Model checkpoints
├── logs/                          # Training logs
└── reference/                     # Reference implementations
    ├── coconut/                   # Original Coconut code
    └── InternVL/                  # InternVL reference
```


#### 2.3.3 A-OKVQA Preprocessing Specification (HuggingFaceM4/A-OKVQA)

This project uses the Multiple-Choice (MC) portion of A-OKVQA only, with rationales for training supervision. The expected dataset fields (HF Datasets) are:

- Required (MC mode):
  - `image`: PIL image (datasets Image feature)
  - `question`: string
  - `choices`: List[str] of length 4 (A, B, C, D)
  - `correct_choice_idx`: int in [0, 3]
  - `rationales`: List[str] (present on train; may be absent or empty on val/test)
- Optional/DA fields (ignored):
  - `direct_answers` or free-form `answers` (list of 10 strings) → discarded
  - any web contexts, captions, generated_answers → discarded
  - dataset bookkeeping (`question_id`, `image_id`) → retained for logging only

Preprocessing rules:

1) Sample filtering
- Keep only multiple-choice samples with exactly 4 options and a valid `correct_choice_idx`.
- Skip training samples without non-empty `rationales` (per project policy).
- Enforce single image per question; if a sample provides multiple, drop or select the first image (we drop by default to avoid leakage).

2) Choice formatting
- Render choices into block text as:
  - `A. {choices[0]}`\n`B. {choices[1]}`\n`C. {choices[2]}`\n`D. {choices[3]}`
- This string is inserted into the question template under the `Choices:` section.

3) Prompt construction
- Base format (before image expansion):
  - `<image>\nQuestion: {question}\n\nChoices:\n{choices_block}\n\nLet me think step by step:\n`
- Reasoning steps: derive from `rationales` during training. We use the first rationale and, for Coconut stages, replace the first N steps with latent tokens according to the current stage.
- Answer line (supervised):
  - `The answer is {chr(65+correct_choice_idx)}. {short_explanation}`
  - `short_explanation` is the first sentence of the rationale if present; otherwise empty string for val/test.

4) Image token replacement (before tokenization)
- Compute tiles via InternVL dynamic preprocessing (max_num=8, use_thumbnail=True), get `num_patches`.
- Replace `<image>` with `IMG_START + (IMG_CONTEXT × 256 × num_patches) + IMG_END`.
- Sanity check: assert the number of `<IMG_CONTEXT>` tokens equals `256 * num_patches`.

5) Tokenization
- Right-padding; do not supply `position_ids` from data/loader; forward() will generate sequential `position_ids` per sample after padding.
- Mask supervision (labels): mask (set to -100) all tokens up to the end of the `question + choices + image tokens + latent tokens` region; supervise the remaining explicit text reasoning and the final answer line.

6) Collation
- Use `MultimodalCoconutCollator` with latent alignment padding across batch.
- Concatenate `pixel_values` for all tiles across the batch and build `image_flags` (1 per valid tile).

7) Evaluation extraction (MC)
- Decode generated text and extract choice with regex `(answer is|Answer:?\s*)([A-D])` (case-insensitive). If missing, default to `A` and log a warning.

Example preprocessing pseudocode:

```python
def preprocess_aokvqa_mc(sample, stage, config):
    # 1) filter
    if len(sample['choices']) != 4:
        return None
    if config.policy.skip_without_rationales and not sample.get('rationales'):
        return None

    # 2) format choices
    choices_block = "\n".join(
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample['choices'])
    )

    # 3) prompt
    question_text = (
        f"<image>\nQuestion: {sample['question']}\n\n"
        f"Choices:\n{choices_block}\n\n"
        f"Let me think step by step:\n"
    )

    rationale = (sample['rationales'][0] if sample.get('rationales') else "")
    reasoning = apply_coconut_stage_training(rationale, stage, has_visual_context=True)
    correct_idx = sample['correct_choice_idx']
    answer_line = f"The answer is {chr(65+correct_idx)}. {rationale.split('.')[0]}".strip()

    full_text = question_text + reasoning + "\n" + answer_line

    # 4) dynamic tiles and image tokens
    pixel_values = dynamic_preprocess(sample['image'], image_size=448, use_thumbnail=True, max_num=config.data.max_num_tiles)
    num_patches = pixel_values.shape[0]
    image_tokens = "<img>" + "<IMG_CONTEXT>" * (config.data.num_image_token * num_patches) + "</img>"
    full_text = full_text.replace('<image>', image_tokens, 1)

    # 5) tokenize & label mask
    tok = tokenizer(full_text, return_tensors='pt', padding=True, truncation=True, max_length=config.data.max_sequence_length)
    question_with_image = question_text.replace('<image>', image_tokens, 1)
    q_len = len(tokenizer(question_with_image)['input_ids'])
    labels = tok['input_ids'].clone()
    labels[:, :q_len] = -100

    return dict(
        input_ids=tok['input_ids'],
        attention_mask=tok['attention_mask'],
        labels=labels,
        pixel_values=pixel_values,
        image_flags=torch.tensor([1]*num_patches, dtype=torch.long),
    )
```

#### 2.3.4 Preprocessing Storage & Caching Policy

- Text/prompt/tokenization: on-the-fly
  - Curriculum stages change latent insertion; prompts and label masks must reflect current stage and settings.
  - Build prompt, insert image tokens, tokenize, and create label masks inside the transform per stage.

- Vision tiles: cached to disk
  - Dynamic tiling dominates preprocessing cost; cache tiles per image to amortize across epochs.
  - Cache path: `./data/cache/tiles/{split}/{image_id}_{hash(image_size,max_num,use_thumbnail,force_image_size,impl_ver)}.pt`
  - Contents: `{ 'pixel_values': FloatTensor[num_tiles, 3, 448, 448], 'num_patches': int }`
  - Invalidation: include tiling params and an implementation version in the hash; write atomically (tmp then rename) to avoid partial writes.

Net result: correctness (stage-aware), speed (cached tiles), flexibility (prompt updates without re-materializing JSONL).

### 3.2 Main Orchestrator Script

Following Coconut's approach with a central `run.py` orchestrator:

```python
# run.py - Main orchestrator script following Coconut pattern
import argparse
import yaml
import torch
from pathlib import Path
from types import SimpleNamespace
from multicoco.trainer import MultimodalCoconutTrainer
from multicoco.evaluator import MultimodalCoconutEvaluator
from multicoco.utils import setup_logging, set_seed

def _dict_to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ _dict_to_ns(v) for v in obj ]
    return obj

def load_config(config_path):
    """Load configuration from YAML file into a nested namespace"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return _dict_to_ns(data)

def main():
    parser = argparse.ArgumentParser(description='MultiCoCo: Multimodal Coconut Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (e.g., configs/aokvqa.yaml)')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'generate'],
                       default='train', help='Running mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation or generation')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    config.debug = args.debug

    # Setup logging and reproducibility
    setup_logging(config)
    set_seed(config.seed)

    # Execute based on mode
    if args.mode == 'train':
        trainer = MultimodalCoconutTrainer(config)
        trainer.train_all_stages()
    elif args.mode == 'eval':
        evaluator = MultimodalCoconutEvaluator(config)
        results = evaluator.evaluate()
        print(f"Evaluation Results: {results}")
    elif args.mode == 'generate':
        evaluator = MultimodalCoconutEvaluator(config)
        evaluator.generate_samples()

    print(f"MultiCoCo {args.mode} completed successfully!")

if __name__ == '__main__':
    main()
```

### 3.3 Configuration System

Following Coconut's YAML-based configuration approach with dataset-specific configs:

#### 3.3.1 Base Configuration

```yaml
# configs/base.yaml - Base configuration template
project: multicoco
name: multicoco-aokvqa-internvl3-1b
output_dir: ./outputs
seed: 42

# Model configuration
model:
  name: OpenGVLab/InternVL3-1B-Pretrained
  torch_dtype: bfloat16
  low_cpu_mem_usage: true
  trust_remote_code: true
  use_flash_attn: true

# Coconut-specific parameters
coconut:
  c_thought: 2                    # Continuous thoughts per reasoning step
  epochs_per_stage: 3             # Training epochs per stage
  epochs_stage0: 8                # Optional override for Stage 0 (full CoT) epochs
  max_latent_stage: 4             # Maximum latent stages
  pad_latent_to_max: true         # Consistent padding strategy
  reset_optimizer: true           # Reset optimizer between stages

# Training configuration
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 20
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
  bf16: true
  use_gradient_checkpointing: true
  save_only_improve: false
  num_workers: 4
  optimizer: adamw
  scheduler: cosine
  log_wandb: true
  wandb_project: multicoco
  wandb_run_name: aokvqa_ivl3_1b
  save_every_epoch: true
  save_best_by: val_accuracy

# Data configuration
data:
  max_sequence_length: 2048
  image_size: 448 # InternVL uses this
  max_num_tiles: 8 # can edit
  num_image_token: 256
  padding_side: right

# Hardware configuration
hardware:
  device: cuda
  world_size: 1
  distributed: false
```

#### 3.3.2 Dataset-Specific Configurations

```yaml
# configs/aokvqa.yaml - A-OKVQA specific configuration (updated)
base_config: configs/base.yaml

# Dataset configuration
dataset:
  name: aokvqa
  hf_dataset_id: HuggingFaceM4/A-OKVQA
  train_split: train
  val_split: validation
  test_split: test

# Prompting/policy
policy:
  multiple_choice_only: true
  skip_without_rationales: true
  single_image_only: true

# Data processing
data:
  question_format: |
    Question: {question}

    Choices:
    {choices}

    Let me think step by step:
  answer_format: "The answer is {choice}. {explanation}"
  image_data_path: ./data/aokvqa/images

# Training specifics
training:
  batch_size: 8
  num_epochs: 15

# Evaluation
evaluation:
  metrics: [accuracy, choice_accuracy]
  generate_explanations: true
  max_new_tokens: 100
  decoding: greedy
```

### 3.4 Training Implementation

#### 3.4.1 Stage-Based Training Script

```python
class MultimodalCoconutTrainer:
    """Training pipeline for multimodal continuous reasoning

    Implements Coconut's exact multi-stage curriculum with multimodal extensions

    AI IMPLEMENTATION NOTE: Critical Training Pipeline Requirements
    1. **Stage-Based Progression**: MUST implement exact Coconut curriculum - Stage 0 (full CoT) → Stage N (N reasoning steps as latent)
    2. **Token Initialization**: New special tokens MUST be initialized with existing embeddings (use "the" token)
    3. **Optimizer Management**: MUST reset optimizer between stages if reset_optimizer=True
    4. **Model Resizing**: MUST call resize_token_embeddings() after adding special tokens
    5. **Distributed Training**: Handle multi-GPU training with proper synchronization
    6. **Checkpoint Management**: Save best models per stage with proper naming convention
    """

    def __init__(self, config):
        self.config = config
        self.setup_model_and_data()

    def setup_model_and_data(self):
        """Initialize model, tokenizer, and data components

        AI IMPLEMENTATION NOTE: Critical Model Setup Requirements
        1. **Special Token Addition**: MUST add Coconut tokens BEFORE model initialization
        2. **Token Embedding Initialization**: Initialize new tokens with existing embeddings (use "the" token)
        3. **Model Component Access**: Extract vision_model, language_model, mlp1 from base InternVL3 model
        4. **IMG_CONTEXT Token ID**: MUST get correct token ID for image context replacement
        5. **Embedding Resizing**: MUST call resize_token_embeddings() after adding special tokens
        """
        # Load InternVL3 components
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        from tqdm import tqdm
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.model.name)

        # Use correct InternVL3 model loading pattern with config variables
        base_model = AutoModel.from_pretrained(
            self.config.model.name,
            torch_dtype=(torch.bfloat16 if getattr(self.config.training, 'bf16', False) else torch.float32),
            low_cpu_mem_usage=getattr(self.config.model, 'low_cpu_mem_usage', True),
            trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
        ).eval()

        # Add special tokens (BOTH Coconut AND InternVL3 tokens)
        # CRITICAL FIX: Use add_special_tokens() to properly register them as special tokens
        # This prevents tokenization splitting and ensures proper special token behavior

        coconut_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        internvl_tokens = ["<img>", "</img>", "<IMG_CONTEXT>"]

        # Check which tokens already exist in the tokenizer
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        new_tokens = [token for token in coconut_tokens + internvl_tokens
                     if token not in existing_vocab]

        num_added = 0
        if new_tokens:
            # CORRECT: Use add_special_tokens() to properly register as special tokens
            # This ensures they won't be split during tokenization and have proper special token behavior
            special_tokens_dict = {
                "additional_special_tokens": new_tokens
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added} new special tokens: {new_tokens}")
        else:
            print("All required special tokens already exist in tokenizer")

        all_special_tokens = coconut_tokens + internvl_tokens

        # Get IMG_CONTEXT token ID with proper error handling
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_context_token_id == self.tokenizer.unk_token_id:
            raise ValueError("IMG_CONTEXT token not found in tokenizer vocabulary. "
                           "Please ensure InternVL3 tokens are properly added.")

        # Correct Model Component Access
        # InternVL3 components are accessed directly from the loaded model
        if not hasattr(base_model, 'vision_model'):
            raise AttributeError("Could not find 'vision_model' in base InternVL3 model")
        if not hasattr(base_model, 'language_model'):
            raise AttributeError("Could not find 'language_model' in base InternVL3 model")
        if not hasattr(base_model, 'mlp1'):
            raise AttributeError("Could not find 'mlp1' (visual projector) in base InternVL3 model")

        # Validate critical configuration parameters with config variables
        config_params = {
            'downsample_ratio': getattr(base_model.config, 'downsample_ratio', 0.5),
            'ps_version': getattr(base_model.config, 'ps_version', 'v1'),
            'select_layer': getattr(base_model.config, 'select_layer', -1),
            'template': getattr(base_model.config, 'template', 'internvl2_5'),
            'max_dynamic_patch': getattr(base_model.config, 'max_dynamic_patch', 12),
            'min_dynamic_patch': getattr(base_model.config, 'min_dynamic_patch', 1),
            'use_thumbnail': getattr(base_model.config, 'use_thumbnail', True),
            'pad2square': getattr(base_model.config, 'pad2square', False),
            'dynamic_image_size': getattr(base_model.config, 'dynamic_image_size', True),
            'force_image_size': getattr(base_model.config, 'force_image_size', 448)
        }

        # Expose InternVL3 config parameters on trainer config for convenience
        for param_name, param_value in config_params.items():
            setattr(self.config, param_name, param_value)

        # Calculate num_image_token from config
        if hasattr(base_model.config, 'vision_config'):
            image_size = config_params['force_image_size'] or base_model.config.vision_config.image_size
            patch_size = base_model.config.vision_config.patch_size
            downsample_ratio = config_params['downsample_ratio']
            self.config.data.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        else:
            self.config.data.num_image_token = 256  # Default fallback

        # Pass complete base model for delegation instead of extracting components
        self.model = MultimodalCoconut(
            base_internvl_model=base_model,  # Pass complete model for delegation
            tokenizer=self.tokenizer,
            latent_token_id=self.tokenizer.convert_tokens_to_ids("<|latent|>"),
            start_latent_id=self.tokenizer.convert_tokens_to_ids("<|start-latent|>"),
            end_latent_id=self.tokenizer.convert_tokens_to_ids("<|end-latent|>"),
            eos_token_id=self.tokenizer.eos_token_id,
            img_context_token_id=img_context_token_id,
            config=self.config
        )

        # Set img_context_token_id on model instance
        self.model.img_context_token_id = img_context_token_id

        # Resize embeddings for new tokens
        num_new_tokens = num_added
        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # Initialize new token embeddings (Improved approach combining Coconut + InternVL best practices)
        # Use averaging approach instead of single token to avoid semantic bias
        if num_new_tokens > 0:
            # Get input embeddings
            input_embeddings = self.model.language_model.get_input_embeddings()
            input_embeddings_data = input_embeddings.weight.data

            # Calculate average of existing embeddings (InternVL approach)
            input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

            # Initialize new tokens with averaged embeddings (more neutral than single token)
            input_embeddings_data[-num_new_tokens:] = input_embeddings_avg

            # Also initialize output embeddings if they exist and are not tied
            if hasattr(self.model.language_model, 'get_output_embeddings'):
                output_embeddings = self.model.language_model.get_output_embeddings()
                if output_embeddings is not None and not self.model.language_model.config.tie_word_embeddings:
                    output_embeddings_data = output_embeddings.weight.data
                    output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

            print(f"Initialized {num_new_tokens} new special tokens with averaged embeddings")

        # Setup data processor
        self.data_processor = MultimodalProcessor(
            self.tokenizer,
            self.image_processor,
            self.config
        )

    def train_all_stages(self):
        """Execute complete multi-stage training curriculum

        Follows Coconut's exact stage progression methodology
        """
        # Load VQA dataset based on config
        train_dataset = self.load_vqa_dataset(split=self.config.dataset.train_split)
        val_dataset = self.load_vqa_dataset(split=self.config.dataset.val_split)

        best_accuracy = 0.0

        # Multi-stage curriculum (exact Coconut approach)
        for stage in range(self.config.coconut.max_latent_stage + 1):
            print(f"\n=== Training Stage {stage} ===")
            print(f"Latent replacement: {stage} reasoning steps → {stage * self.config.coconut.c_thought} latent tokens")

            # Process data for current stage
            train_dataset.set_transform(lambda sample: self.data_processor.process_sample(sample, stage))
            val_dataset.set_transform(lambda sample: self.data_processor.process_sample(sample, stage))

            # Create data loaders
            collator = MultimodalCoconutCollator(
                self.tokenizer,
                self.tokenizer.convert_tokens_to_ids("<|latent|>")
            )

            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=self.config.training.num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=self.config.training.num_workers
            )

            # Setup optimizer for current stage
            if self.config.coconut.reset_optimizer or stage == 0:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )

            # Train current stage
            stage_accuracy = self.train_single_stage(
                train_loader, val_loader, optimizer, stage
            )

            # Save checkpoint if improved
            if stage_accuracy > best_accuracy:
                best_accuracy = stage_accuracy
                self.save_checkpoint(stage, stage_accuracy)

            print(f"Stage {stage} completed. Accuracy: {stage_accuracy:.3f}")

    def train_single_stage(self, train_loader, val_loader, optimizer, stage):
        """Train single stage following Coconut's methodology"""

        # Allow Stage 0 (full CoT) to run for a different number of epochs via config.coconut.epochs_stage0
        epochs_this_stage = (
            getattr(self.config.coconut, 'epochs_stage0', self.config.coconut.epochs_per_stage)
            if stage == 0 else self.config.coconut.epochs_per_stage
        )

        for epoch in range(epochs_this_stage):
            # Training loop
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.config.hardware.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']

                # Backward pass with gradient accumulation
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Validation
            val_accuracy = self.evaluate(val_loader)
            print(f"Stage {stage}, Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Accuracy: {val_accuracy:.3f}")

        return val_accuracy

    def evaluate(self, val_loader):
        """Evaluate model performance on validation set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.config.hardware.device) for k, v in batch.items()}

                # Generate predictions
                outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    max_new_tokens=self.config.evaluation.max_new_tokens,
                    do_sample=False
                )

                # Extract predicted answers
                sequences = outputs['sequences'] if isinstance(outputs, dict) else outputs
                for i in range(sequences.shape[0]):
                    generated_text = self.tokenizer.decode(sequences[i], skip_special_tokens=True)
                    # If choices are available in batch, pass them for robust extraction
                    batch_choices = batch.get('choices', None)
                    if batch_choices is not None:
                        # Expecting shape [B, 4] or list of lists
                        if isinstance(batch_choices, (list, tuple)):
                            choices_i = batch_choices[i]
                        else:
                            try:
                                choices_i = batch_choices[i]
                            except Exception:
                                choices_i = None
                    else:
                        choices_i = None

                    predicted_choice = self.extract_choice_from_text(generated_text, choices=choices_i)

                    # Compare with ground truth (implementation depends on label format)
                    # This would need to be adapted based on how labels are encoded
                    # For now, placeholder logic
                    correct += 1  # Placeholder
                    total += 1

        return correct / total if total > 0 else 0.0

    def extract_choice_from_text(self, text, choices=None):
        """Robust extraction of multiple-choice answers from free-form text.

        Supports indices (0/1/2/3), letters (A-D), or answer words.
        If `choices` is provided (list of 4 options), will fuzzy-match the option text.
        """
        import re
        from difflib import SequenceMatcher

        if not text:
            return 0

        t = text.strip()

        # 1) Prefer explicit letter mention near common answer phrases
        letter_pattern = r'(?:\b(?:answer\s*(?:is|:)\s*|^\s*)([A-D])\b)'
        m = re.search(letter_pattern, t, re.IGNORECASE)
        if m:
            return ord(m.group(1).upper()) - ord('A')

        # 2) Index-based answers like "2", "option 3", "the 1st one"
        index_patterns = [
            r'\boption\s*([1-4])\b',
            r'\bthe\s*([1-4])(?:st|nd|rd|th)\b',
            r'\bchoose\s*([1-4])\b',
            r'\banswer\s*([1-4])\b',
            r'^[\s]*([1-4])[\s]*$'
        ]
        for pat in index_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx <= 3:
                    return idx

        # 3) Single-letter tokens A-D anywhere (as a fallback if unambiguous)
        single_letter = re.findall(r'\b([A-D])\b', t, flags=re.IGNORECASE)
        if single_letter:
            # Use the last one (often the final conclusion)
            return ord(single_letter[-1].upper()) - ord('A')

        # 4) Word-based matching against provided choices
        if choices and len(choices) == 4:
            # Normalize
            norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
            nt = norm(t)
            nchoices = [norm(c) for c in choices]

            # Exact contains
            for i, c in enumerate(nchoices):
                if c and c in nt:
                    return i

            # Heuristic similarity (SequenceMatcher)
            best_i, best_score = 0, 0.0
            for i, c in enumerate(nchoices):
                if not c:
                    continue
                score = SequenceMatcher(None, nt, c).ratio()
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_score >= 0.6:
                return best_i

        # 5) Fallback to A
        return 0

    def save_checkpoint(self, stage, accuracy):
        """Save model checkpoint"""
        checkpoint_path = f"{self.config.output_dir}/stage_{stage}_acc_{accuracy:.3f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
```


    def load_vqa_dataset(self, split='train'):
        """Load VQA dataset based on configuration"""
        from datasets import load_dataset

        dataset = load_dataset(
            self.config.dataset.hf_dataset_id,
            split=split,
            trust_remote_code=True
        )

        return dataset

    def extract_choice_from_text(self, text, choices=None):
        """Robust extraction of multiple-choice answers from free-form text.

        Supports indices (0/1/2/3), letters (A-D), or answer words.
        If `choices` is provided (list of 4 options), will fuzzy-match the option text.
        """
        import re
        from difflib import SequenceMatcher

        if not text:
            return 0

        t = text.strip()

        # 1) Prefer explicit letter mention near common answer phrases
        letter_pattern = r'(?:\b(?:answer\s*(?:is|:)\s*|^\s*)([A-D])\b)'
        m = re.search(letter_pattern, t, re.IGNORECASE)
        if m:
            return ord(m.group(1).upper()) - ord('A')

        # 2) Index-based answers like "2", "option 3", "the 1st one"
        index_patterns = [
            r'\boption\s*([1-4])\b',
            r'\bthe\s*([1-4])(?:st|nd|rd|th)\b',
            r'\bchoose\s*([1-4])\b',
            r'\banswer\s*([1-4])\b',
            r'^[\s]*([1-4])[\s]*$'
        ]
        for pat in index_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx <= 3:
                    return idx

        # 3) Single-letter tokens A-D anywhere (as a fallback if unambiguous)
        single_letter = re.findall(r'\b([A-D])\b', t, flags=re.IGNORECASE)
        if single_letter:
            return ord(single_letter[-1].upper()) - ord('A')

        # 4) Word-based matching against provided choices
        if choices and len(choices) == 4:
            norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
            nt = norm(t)
            nchoices = [norm(c) for c in choices]
            for i, c in enumerate(nchoices):
                if c and c in nt:
                    return i
            best_i, best_score = 0, 0.0
            for i, c in enumerate(nchoices):
                if not c:
                    continue
                score = SequenceMatcher(None, nt, c).ratio()
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_score >= 0.6:
                return best_i

        return 0

    def save_checkpoint(self, stage, accuracy):
        """Save model checkpoint"""
        checkpoint_path = f"{self.config.output_dir}/stage_{stage}_acc_{accuracy:.3f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
```

### 3.5 Usage Examples

Following Coconut's command-line interface pattern:

#### 3.5.1 Training Commands

```bash
# Train on A-OKVQA dataset
python run.py --config configs/aokvqa.yaml --mode train

# Train with custom output directory
python run.py --config configs/aokvqa.yaml --mode train --output_dir ./custom_outputs

# Debug mode training
python run.py --config configs/aokvqa.yaml --mode train --debug
```

#### 3.5.2 Evaluation Commands

```bash
# Evaluate trained model
python run.py --config configs/aokvqa.yaml --mode eval --checkpoint ./checkpoints/stage_4_acc_0.723.pt

# Generate sample outputs
python run.py --config configs/aokvqa.yaml --mode generate --checkpoint ./checkpoints/best_model.pt
```

#### 3.5.3 Data Preparation

```bash
# Prepare A-OKVQA data
python scripts/prepare_data.py --dataset aokvqa --output_dir ./data/aokvqa
```


## 4. Implementation Plan and Validation

### 4.1 Phased Development Approach

#### Phase 1: Foundation Implementation (Weeks 1-2)

- **Core Architecture**: Implement `MultimodalCoconut` class with exact Coconut reasoning logic
- **Component Integration**: Integrate InternVL3 vision encoder
- **Data Pipeline**: Develop A-OKVQA processing with stage-based curriculum
- **Unit Testing**: Comprehensive testing of individual components


#### Phase 2: Training Pipeline (Weeks 3-4)

- **Multi-Stage Training**: Implement complete curriculum learning pipeline
- **Evaluation Framework**: Develop comprehensive evaluation metrics
- **Hyperparameter Optimization**: Fine-tune training parameters
- **Memory Optimization**: Implement gradient checkpointing, mixed precision training, and efficient KV-cache management


#### Phase 3: Experimental Validation (Weeks 5-6)

- **Baseline Comparison**: Compare against standard multimodal VQA models
- **Ablation Studies**: Analyze contribution of continuous reasoning components
- **Performance Analysis**: Comprehensive evaluation on A-OKVQA benchmark
- **Reasoning Quality Assessment**: Qualitative analysis of generated reasoning


#### Phase 4: Optimization and Deployment (Weeks 7-8)

- **Model Optimization**: Performance tuning and inference optimization
- **Documentation**: Complete technical documentation and usage guides
- **Reproducibility**: Ensure complete experimental reproducibility
- **Open Source Release**: Prepare for community contribution


### 4.2 Expected Performance Metrics

#### Quantitative Targets

- **VQA Accuracy**: Target >70% on validation sets across datasets
- **Training Efficiency**: 50% reduction in reasoning tokens compared to CoT
- **Convergence Speed**: Faster convergence through curriculum learning
- **Memory Efficiency**: Effective handling of multimodal inputs within 32GB VRAM

#### Qualitative Assessments

- **Reasoning Coherence**: Improved logical consistency in multi-step reasoning
- **Visual Understanding**: Enhanced integration of visual and textual information
- **Knowledge Integration**: Better application of commonsense and world knowledge
- **Generalization**: Strong performance across diverse question types and datasets

### 4.3 Risk Mitigation Strategies

#### Technical Challenges

- **Integration Complexity**: Modular development with comprehensive testing
- **Memory Constraints**: Gradient checkpointing, mixed precision training, and optimized batch sizes
- **Training Instability**: Careful hyperparameter tuning and monitoring
- **Performance Bottlenecks**: Profiling and optimization at each stage

#### Resource Management

- **Computational Requirements**: Efficient multi-GPU training strategies with model parallelism
- **Data Processing**: Optimized pipelines for multimodal data handling
- **Development Timeline**: Agile methodology with clear milestones
- **Quality Assurance**: Continuous integration and automated testing

## 5. Usage Examples

Following Coconut's command-line interface pattern:

### 5.1 Training Commands

```bash
# Train on A-OKVQA dataset
python run.py --config configs/aokvqa.yaml --mode train

# Train with custom output directory
python run.py --config configs/aokvqa.yaml --mode train --output_dir ./custom_outputs

# Debug mode training
python run.py --config configs/aokvqa.yaml --mode train --debug
```

### 5.2 Evaluation Commands

```bash
# Evaluate trained model
python run.py --config configs/aokvqa.yaml --mode eval --checkpoint ./checkpoints/stage_4_acc_0.723.pt

# Generate sample outputs
python run.py --config configs/aokvqa.yaml --mode generate --checkpoint ./checkpoints/best_model.pt
```

### 5.3 Data Preparation

```bash
# Prepare A-OKVQA data
python scripts/prepare_data.py --dataset aokvqa --output_dir ./data/aokvqa
```

## 6. Technical Innovation and Contributions

### 6.1 Novel Architectural Contributions

This design represents several significant technical innovations:

1. **First Multimodal Extension of Continuous Reasoning**: Pioneering application of latent-space reasoning to vision-language tasks
2. **Sophisticated Multimodal Fusion**: Advanced integration mechanisms specifically designed for continuous thought processing
3. **Curriculum Learning for Multimodal Reasoning**: Extension of Coconut's stage-based training to complex visual reasoning scenarios
4. **Efficient Visual Token Processing**: Integration of InternVL3's pixel unshuffle with continuous reasoning for optimal efficiency

### 6.2 Research Impact and Applications

The successful implementation of this system will:

- **Advance Multimodal AI**: Push boundaries of vision-language understanding through latent reasoning
- **Enable Efficient Reasoning**: Reduce computational overhead while improving reasoning quality
- **Facilitate Research**: Provide open-source framework for multimodal continuous reasoning research
- **Support Applications**: Enable deployment in resource-constrained environments requiring visual reasoning

### 6.3 Future Extensions and Scalability

This architecture provides a foundation for:

- **Video Understanding**: Extension to temporal visual reasoning
- **Document Analysis**: Application to complex document understanding tasks
- **Embodied AI**: Integration with robotics and interactive systems
- **Multi-Domain Reasoning**: Adaptation to scientific, medical, and technical domains
