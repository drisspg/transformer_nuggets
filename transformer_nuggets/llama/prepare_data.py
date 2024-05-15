"""Use the tokenizer and data to prepare pretrain dataset - Heavily inspired by Nanogpt"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset  # huggingface datasets
from fire import Fire
from tqdm import tqdm
from transformer_nuggets.llama.tokenizer import Tokenizer


logging.basicConfig(level=logging.INFO)


@dataclass
class DataConfig:
    """Data configuration for pretraining"""

    tokenizer_path: Path
    output_dir: Path
    dataset_name: str = "openwebtext"
    test_size: float = 0.0005
    seed: int = 42
    num_proc: int = os.cpu_count() // 2
    num_proc_load_dataset: int = os.cpu_count() // 2


def main(tokenizer_path: str, output_dir: str):
    """Prepare pretraining dataset"""
    data_config = DataConfig(tokenizer_path=Path(tokenizer_path), output_dir=Path(output_dir))
    assert data_config.tokenizer_path.exists(), f"{data_config.tokenizer_path} does not exist"

    data_config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(str(data_config.tokenizer_path))

    logging.info("Loading dataset %s", data_config.dataset_name)
    dataset = load_dataset(data_config.dataset_name, num_proc=data_config.num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(
        test_size=data_config.test_size, seed=data_config.seed, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = tokenizer.encode(example["text"], bos=True, eos=True)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=data_config.num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = data_config.output_dir / f"{split}.bin"
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    logging.info("Wrote pretraining dataset to %s", data_config.output_dir)


if __name__ == "__main__":
    Fire(main)
