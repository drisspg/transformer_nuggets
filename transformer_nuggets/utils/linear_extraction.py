import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from dataclasses import dataclass
import re

# Import transformers components
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


@dataclass
class LinearCapture:
    """Data class to store captured Linear layer information."""

    module_fqn: str  # Fully qualified name
    layer_idx: int | None
    weight: torch.Tensor
    bias: torch.Tensor | None
    inputs: list[torch.Tensor]  # Multiple captures from different prompts


# Global storage for captured data
linear_captures: dict[str, LinearCapture] = {}
target_layers: set[int] = set()
name_pattern: str | None = None


def _extract_text(item: dict) -> str | None:
    if "prompt" in item and item["prompt"]:
        return item["prompt"]

    if "text" in item and item["text"]:
        return item["text"]

    conversations = item.get("conversations")
    if isinstance(conversations, list):
        parts = [turn.get("value") for turn in conversations if isinstance(turn, dict)]
        parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        if parts:
            return "\n\n".join(parts)

    return None


def _get_layer_idx(fqn: str) -> int | None:
    """Extract layer_idx from fully qualified name."""
    # Try to extract layer index from patterns like:
    # - "model.layers.8.mlp.gate_proj"
    # - "transformer.h.8.mlp.c_fc"
    # - "model.decoder.layers.8.fc1"

    patterns = [
        r"\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, fqn)
        if match:
            return int(match.group(1))

    return None


def _create_hook(fqn: str):
    """Create a forward hook for a specific module."""

    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        # Extract layer index from FQN
        layer_idx = _get_layer_idx(fqn)

        # Apply layer filtering
        if target_layers and layer_idx not in target_layers:
            return

        # Apply name pattern filtering
        if name_pattern and not re.search(name_pattern, fqn):
            return

        # Initialize capture object if first time seeing this module
        if fqn not in linear_captures:
            linear_captures[fqn] = LinearCapture(
                module_fqn=fqn,
                layer_idx=layer_idx,
                weight=None,  # Will be populated later
                bias=None,  # Will be populated later
                inputs=[],
            )

        # Capture input
        # input is a tuple, we want the first element (the actual input tensor)
        # Keep on GPU for now, will move to CPU at the end
        input_tensor = input[0].detach()

        linear_captures[fqn].inputs.append(input_tensor)

        # Memory management
        gc.collect()
        torch.cuda.empty_cache()

    return hook


def _register_hooks(model: nn.Module):
    """Register forward hooks on all Linear layers in the model."""
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Apply name pattern filter at registration time for efficiency
            if name_pattern and not re.search(name_pattern, name):
                continue

            hook = module.register_forward_hook(_create_hook(name))
            hooks.append(hook)

    print(f"Registered hooks on {len(hooks)} Linear layers")
    return hooks


def _extract_weights_and_biases(model: nn.Module):
    """Extract weights and biases from Linear layers after forward passes and move everything to CPU."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in linear_captures:
            # Move weights and biases to CPU
            linear_captures[name].weight = module.weight.detach().cpu().clone()
            linear_captures[name].bias = (
                module.bias.detach().cpu().clone() if module.bias is not None else None
            )

            # Move all captured inputs to CPU
            linear_captures[name].inputs = [inp.cpu() for inp in linear_captures[name].inputs]


def extract_linear_data(
    model_id: str,
    dataset_name: str = "fka/awesome-chatgpt-prompts",
    num_samples: int = 2,
    max_length: int = 2048,
    min_prompt_length: int = 500,
    layers: list[int] | None = None,
    name_pattern_filter: str | None = None,
):
    """
    Extract Linear layer inputs, weights, and biases from a model.

    Args:
        model_id: HuggingFace model identifier
        dataset_name: HuggingFace dataset to use for prompts
        num_samples: Number of prompts to process
        max_length: Maximum sequence length in tokens
        min_prompt_length: Minimum prompt length in characters
        layers: List of layer indices to capture (None = all layers)
        name_pattern_filter: Regex pattern to filter module names (None = all modules)

    Returns:
        Dictionary mapping FQN to LinearCapture objects
    """
    global linear_captures, target_layers, name_pattern

    # Reset global state
    linear_captures = {}
    target_layers = set(layers) if layers is not None else set()
    name_pattern = name_pattern_filter

    if target_layers:
        print(f"Extracting from layers: {sorted(target_layers)}")
    else:
        print("Extracting from all layers")

    if name_pattern:
        print(f"Filtering modules with pattern: {name_pattern}")
    else:
        print("No name pattern filter applied")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    # Select prompts
    tolerance = max(64, int(max_length * 0.1))
    candidates: list[tuple[str, int]] = []

    for _, item in tqdm(enumerate(dataset), desc="Scanning prompts", total=len(dataset)):
        prompt = _extract_text(item)
        if prompt is None or len(prompt) <= min_prompt_length:
            continue

        encoded = tokenizer(
            prompt,
            padding=False,
            truncation=False,
        )
        token_len = len(encoded["input_ids"])

        candidates.append((prompt, token_len))

    if not candidates:
        raise ValueError(
            f"No prompts longer than {min_prompt_length} characters found in {dataset_name}"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    long_prompts = [(p, length) for p, length in candidates if length >= max_length]
    near_prompts = [
        (p, length) for p, length in candidates if max_length - tolerance <= length < max_length
    ]
    fallback_prompts = [(p, length) for p, length in candidates if length < max_length - tolerance]

    selected_prompts: list[str] = []
    selected_lengths: list[int] = []

    def take_from(bucket: list[tuple[str, int]]):
        for prompt, length in bucket:
            if len(selected_prompts) >= num_samples:
                break
            selected_prompts.append(prompt)
            selected_lengths.append(length)

    take_from(long_prompts)
    take_from(near_prompts)
    take_from(fallback_prompts)

    if len(selected_prompts) < num_samples:
        print(
            f"Warning: Requested {num_samples} prompts but only found {len(selected_prompts)} with sufficient length"
        )

    print(
        "Prompt buckets -> "
        f"long: {len(long_prompts)}, near: {len(near_prompts)}, fallback: {len(fallback_prompts)} "
        f"(tolerance={tolerance} tokens)"
    )
    print(f"Selected prompt lengths: {selected_lengths}")

    print(f"Processing {len(selected_prompts)} prompts")

    # Load model
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()

    # Register hooks
    hooks = _register_hooks(model)

    # Process prompts
    for _, prompt in tqdm(enumerate(selected_prompts), desc="Processing prompts"):
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        tokenized = {k: v.cuda() for k, v in tokenized.items()}

        with torch.inference_mode():
            model(**tokenized)

        del tokenized
        gc.collect()
        torch.cuda.empty_cache()

    # Extract weights and biases
    print("Extracting weights and biases...")
    _extract_weights_and_biases(model)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Collected data from {len(linear_captures)} Linear layers")
    for fqn, capture in linear_captures.items():
        print(f"  {fqn}: {len(capture.inputs)} inputs")

    return linear_captures


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/home/dev/meta/my_scripts")
    from quant_compare import compare_fp8_quantization

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    captures = extract_linear_data(
        model_id=model_id,
        dataset_name="fka/awesome-chatgpt-prompts",
        num_samples=2,
        max_length=2048,
        min_prompt_length=500,
        layers=[8],  # Layer 8
        name_pattern_filter="mlp",  # MLP modules only
    )

    print(f"\nExtracted {len(captures)} layers")
    for name, capture in captures.items():
        print(f"  {name}: weight {capture.weight.shape}, {len(capture.inputs)} inputs")

    # Run FP8 quantization comparison
    if len(captures) > 0:
        print("\n" + "=" * 80)
        print("Starting FP8 Quantization Comparison")
        print("=" * 80)
        results = compare_fp8_quantization(captures)
        print("\nComparison complete!")
    else:
        print("\nNo captures found - skipping quantization comparison")
        print("Note: Make sure the layer filtering matches the model architecture")
