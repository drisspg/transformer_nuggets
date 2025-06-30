# SPDX-License-Identifier: Apache-2.0

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from transformer_nuggets.utils.shape_trace import ShapeLog
from vllm import LLM, SamplingParams
from rich import print


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sharegpt_data(dataset_path: str, seed: int = 42):
    random.seed(seed)
    with open(dataset_path) as f:
        dataset = json.load(f)

    conversations = []
    for data in dataset:
        if len(data["conversations"]) >= 2:
            human_msg = data["conversations"][0]["value"]
            conversations.append(human_msg)

    random.shuffle(conversations)
    return conversations


def create_batches_by_length(
    dataset_path: str, batch_size: int, num_batches: int = 5, seed: int = 42
):
    conversations = load_sharegpt_data(dataset_path, seed)
    conversations.sort(key=len)

    batches = []
    total_convs = len(conversations)

    for i in range(num_batches):
        start_idx = (i * total_convs) // num_batches
        end_idx = ((i + 1) * total_convs) // num_batches

        batch_convs = conversations[start_idx:end_idx]
        random.shuffle(batch_convs)

        batch = batch_convs[:batch_size]

        while len(batch) < batch_size:
            batch.extend(batch_convs[: batch_size - len(batch)])

        batches.append(batch[:batch_size])

    return batches


def main(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    dataset_path: str = "/home/drisspg/meta/my_scripts/data/ShareGPT_V3_unfiltered_cleaned_split.json",
    batch_size: int = 32,
    num_batches: int = 5,
    max_tokens: int = 16,
    eager: bool = False,
    tp_size: int = 1,
    seed: int = 42,
):
    """
    Trace matmul operation shapes during vLLM inference.

    Args:
        model_name: HuggingFace model path to profile
        dataset_path: Path to ShareGPT dataset JSON file
        batch_size: Number of prompts per batch
        num_batches: Number of batches with different prompt lengths
        max_tokens: Maximum tokens to generate per prompt
        eager: Disable CUDA graphs and use eager execution
        tp_size: Tensor parallel size for distributed inference
        seed: Random seed for reproducibility
    """
    print(f"Using Model {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Batch size: {batch_size}, Num batches: {num_batches}")
    print(f"Seed: {seed}")

    set_seed(seed)

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=seed, max_tokens=max_tokens)

    llm = LLM(model=model_name, tensor_parallel_size=tp_size, enforce_eager=eager)

    print(f"Max model length: {llm.llm_engine.model_config.max_model_len}")

    batches = create_batches_by_length(dataset_path, batch_size, num_batches, seed)

    print(f"Created {len(batches)} batches")

    print("Warmup run...")
    _ = llm.generate(batches[0], sampling_params)

    shape_log_path = Path(
        f"data/{model_name.replace('/', '_')}_matmul_shapes_b{batch_size}_n{num_batches}.pkl"
    )

    print(f"Shape tracing enabled, saving to {shape_log_path}")

    tokenizer = llm.get_tokenizer()

    matmul_ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
    ]

    shape_tracer = ShapeLog(log_path=shape_log_path, with_type=True, specific_ops=matmul_ops)

    with shape_tracer:
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)} (batch size: {len(batch)})")

            token_counts = [len(tokenizer.encode(prompt)) for prompt in batch]
            total_prefill_tokens = sum(token_counts)
            avg_tokens = total_prefill_tokens // len(batch)

            print(f"  Total prefill tokens: {total_prefill_tokens}")
            print(f"  Average tokens per prompt: {avg_tokens}")
            print(f"  Token range: {min(token_counts)} - {max(token_counts)}")

            llm.generate(batch, sampling_params)

    print("Shape tracing complete!")
    print(f"Matmul shapes saved to {shape_log_path}")
    print("Use transformer_nuggets.utils.shape_trace.open_logs() to analyze the traces")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
