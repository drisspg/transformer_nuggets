"""
Used to train a model from scratch on big dense blocks of text data using causal attention.
"""
import csv
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fire import Fire
import transformer_nuggets.quant.qlora as qlora
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import transformer_nuggets.llama.train
from transformer_nuggets.llama.model import (
    ModelArgs,
    Transformer,
)
from transformer_nuggets.llama.train import (
    TrainingConfig,
    log_num_params,
    train,
    load_datasets,
)

logging.basicConfig(level=logging.INFO)

@dataclass
class Hyperparameters(transformer_nuggets.llama.train.Hyperparameters):
    # qlora config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

def main(
    hyper_params: Hyperparameters,
    training_config: TrainingConfig,
):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    os.makedirs(training_config.out_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)

    # Setup Model
    model_args = ModelArgs.from_name(training_config.model_name)
    logging.info(f"Initializing model: {training_config.model_name}")
    with training_config.device:
        model = Transformer(model_args).to(torch.bfloat16)
        model.init_parameters()

        qlora_config = qlora.QloraConfig(
            hyper_params.lora_r,
            hyper_params.lora_alpha,
            hyper_params.lora_dropout,
        )
        qlora.swap_for_qlora(model, qlora_config, torch.bfloat16)

    model.setup_caches(
        hyper_params.micro_batch_size, hyper_params.max_seq_length, training_config.device
    )

    logging.info("Setting up the dataloaders")
    train_data, val_data = load_datasets(hyper_params, training_config)
    train_dataloader = DataLoader(
        train_data, batch_size=hyper_params.micro_batch_size, num_workers=2
    )
    val_dataloader = DataLoader(val_data, batch_size=hyper_params.micro_batch_size, num_workers=2)

    log_num_params(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hyper_params.learning_rate,
        weight_decay=hyper_params.weight_decay,
        betas=(hyper_params.beta1, hyper_params.beta2),
        foreach=hyper_params.foreach_optimizer,
    )

    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        hyper_params,
        training_config,
    )

def entrypoint(
    profile: bool = False,
):
    assert isinstance(profile, bool), "profile must be bool"
    hyper_params = Hyperparameters()
    training_config = TrainingConfig(profile=profile)
    main(hyper_params, training_config)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    Fire(entrypoint)
