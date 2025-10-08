from transformers import AutoModelForCausalLM, AttentionInterface, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from datasets import load_dataset
import torch
import gc
from tqdm import tqdm


qkv = []
target_layers = set()


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
    num_random_samples: int = 8,
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

    prompts = []
    for _, item in tqdm(enumerate(dataset), desc="Loading prompts"):
        if len(prompts) >= num_samples:
            break
        if "prompt" in item and len(item["prompt"]) > min_prompt_length:
            prompts.append(item["prompt"])

    if len(prompts) < num_samples:
        print(f"Warning: Only found {len(prompts)} prompts longer than {min_prompt_length} chars")

    print(f"Processing {len(prompts)} prompts")

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="my_new_sdpa"
    ).cuda()

    for _, prompt in tqdm(enumerate(prompts), desc="Processing prompts"):
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

    print(f"Total shape before sampling: {qq.shape}")

    idx = torch.randperm(qq.shape[0])[:num_random_samples]
    qq = qq[idx, ...]
    kk = kk[idx, ...]
    vv = vv[idx, ...]

    print(f"Final sampled shape: {qq.shape}")

    return qq, kk, vv


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    q, k, v = extract_attention_data(
        model_id=model_id,
        dataset_name="fka/awesome-chatgpt-prompts",
        num_samples=2,
        max_length=2048,
        min_prompt_length=500,
        num_random_samples=8,
    )

    torch.save({"q": q, "k": k, "v": v}, "attention_data.pt")
    print("Saved to attention_data.pt")
