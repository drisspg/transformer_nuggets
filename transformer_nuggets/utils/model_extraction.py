from transformers import AutoModelForCausalLM, AttentionInterface, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from datasets import load_dataset
import torch
import gc
from tqdm import tqdm


qkv = []
target_layers = set()


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


def my_new_sdpa(module, *args, **kwargs):
    layer_idx = module.layer_idx if hasattr(module, "layer_idx") else None

    if not target_layers or layer_idx in target_layers:
        qkv.append((args[0].cpu(), args[1].cpu(), args[2].cpu()))
        gc.collect()
        torch.cuda.empty_cache()

    return sdpa_attention_forward(module, *args, **kwargs)


def extract_attention_data(
    model_id: str,
    dataset_name: str = "fka/awesome-chatgpt-prompts",
    num_samples: int = 2,
    max_length: int = 2048,
    min_prompt_length: int = 500,
    layers: list[int] | None = None,
):
    global qkv, target_layers
    qkv = []
    target_layers = set(layers) if layers is not None else set()

    if target_layers:
        print(f"Extracting from layers: {sorted(target_layers)}")
    else:
        print("Extracting from all layers")

    AttentionInterface.register("my_new_sdpa", my_new_sdpa)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    tolerance = max(64, int(max_length * 0.1))
    candidates: list[tuple[str, int]] = []

    # pyrefly: ignore  # not-iterable, bad-argument-type
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

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="my_new_sdpa"
    ).cuda()

    # pyrefly: ignore  # not-iterable
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

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Collected {len(qkv)} QKV tuples from all layers")

    qq = torch.cat([e[0] for e in qkv], dim=0)
    kk = torch.cat([e[1] for e in qkv], dim=0)
    vv = torch.cat([e[2] for e in qkv], dim=0)

    print(f"Final shape: {qq.shape}")

    return qq, kk, vv


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    q, k, v = extract_attention_data(
        model_id=model_id,
        dataset_name="fka/awesome-chatgpt-prompts",
        num_samples=2,
        max_length=2048,
        min_prompt_length=500,
        # pyrefly: ignore  # unexpected-keyword
        num_random_samples=8,
    )

    torch.save({"q": q, "k": k, "v": v}, "attention_data.pt")
    print("Saved to attention_data.pt")
