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

import numpy as np
import torch
from fire import Fire
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear import Float8Linear

# Float8 imports
from float8_experimental.float8_linear_utils import (
    linear_requires_sync,
    LinearType,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from transformer_nuggets.llama.model import ModelArgs, Transformer

LINEAR_TYPE_MAP = {
    LinearType.DELAYED: Float8Linear,
    LinearType.DYNAMIC: Float8DynamicLinear,
}

logging.basicConfig(level=logging.INFO)


@dataclass
class Hyperparameters:
    learning_rate: float = 6e-4
    batch_size: int = 128
    micro_batch_size: int = 1
    gradient_accumulation_iters: int = field(init=False)
    max_seq_length: int = 4096
    max_iters: int = 600000  # train dataset size
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = field(init=False)
    min_lr: float = 6e-5
    foreach_optimizer: bool = False

    # Float8 Specific Config
    # We want to skip the first embedding layer since scaled_mm needs to multiple of 16
    float8_skip_list: list[str] = field(default_factory=lambda: ["tok_embeddings"])
    fp8_linear_type: LinearType | None = None

    def __post_init__(self):
        self.gradient_accumulation_iters = self.batch_size // self.micro_batch_size
        self.lr_decay_iters = self.max_iters
        assert self.gradient_accumulation_iters > 0
        if self.fp8_linear_type is not None:
            self.fp8_linear_type = LinearType[self.fp8_linear_type.upper()]


@dataclass
class TrainingConfig:
    eval_interval: int = 500
    save_interval: int = 10000
    eval_iters: int = 100
    log_interval: int = 200
    val_step_count: int = 0
    deterministic_data_loading: bool = False

    # This overfit param is used to test numerical issues by overfitting
    # on a single batch. It should be set to False for normal training.
    overfit: bool = False

    compile: bool = False
    model_name: str = "7B"
    dataset_name: str = "openwebtext"
    base_path = Path("transformer_nuggets/llama/data")
    out_dir: Path = base_path / "out"
    data_dir: Path = base_path
    log_dir: Path = base_path / "logs"

    device: torch.device = torch.device("cuda:0")
    # If true we will profile iters 100-102 of the model training
    profile: bool = False


def write_loss_to_file(loss_file: Path, step: int, loss: float):
    """Writes the loss to a csv file for later plotting
    Args:
        loss_file: The file to write the loss to
        step: The current step
        loss: The loss to write
    """
    if not loss_file.exists():
        with open(loss_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
    with open(loss_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([step, loss])


def get_profile_context(hyper_params: Hyperparameters, train_config: TrainingConfig):
    """Returns a context manager that can be used to profile the model."""

    def trace_handler(prof):
        fp8_linear_type = hyper_params.fp8_linear_type

        dtype_str = fp8_linear_type if fp8_linear_type else "bf16"
        output_str = f"/tmp/trace_llama_7b_hf_{dtype_str}.json"
        prof.export_chrome_trace(output_str)
        logging.info(f"Wrote profile to: {output_str}")

    if train_config.profile:
        context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=100, warmup=1, active=2, repeat=1),
            record_shapes=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        )
        return context
    else:
        return nullcontext()


def log_num_params(model: Transformer):
    """Logs the number of parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"The number of trainable parameters: {num_params:,}")


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

    model.setup_caches(
        hyper_params.micro_batch_size,
        hyper_params.max_seq_length,
        training_config.device,
    )

    logging.info("Setting up the dataloaders")
    train_data, val_data = load_datasets(hyper_params, training_config)
    train_dataloader = DataLoader(
        train_data, batch_size=hyper_params.micro_batch_size, num_workers=2
    )
    val_dataloader = DataLoader(val_data, batch_size=hyper_params.micro_batch_size, num_workers=2)

    fp8_linear_type = hyper_params.fp8_linear_type
    if fp8_linear_type is not None:
        fp8_module = LINEAR_TYPE_MAP[fp8_linear_type]
        swap_linear_with_float8_linear(
            model, fp8_module, skip_fqn_list=hyper_params.float8_skip_list
        )

    log_num_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyper_params.learning_rate,
        weight_decay=hyper_params.weight_decay,
        betas=(hyper_params.beta1, hyper_params.beta2),
        foreach=hyper_params.foreach_optimizer,
    )
    if training_config.compile:
        model = torch.compile(model)

    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        hyper_params,
        training_config,
    )

    # Save the final LoRA checkpoint at the end of training
    # save_path = out_dir / "lit_model_full_finetuned.pth"
    # torch.save(save_path, {"model": model})


def train(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    hyper_params: Hyperparameters,
    training_config: TrainingConfig,
) -> None:
    """Lets go!"""
    step_count = 0
    progress_bar = tqdm(total=hyper_params.max_iters)

    model.train()
    profile_context = get_profile_context(hyper_params, training_config)
    train_iter = iter(train_data)

    # Sanity check
    fp8_linear_type = hyper_params.fp8_linear_type
    dtype_str = fp8_linear_type if fp8_linear_type else "bf16"

    if not hasattr(hyper_params, "lora_r"):
        loss_file_prefix = "pretrain"
    else:
        loss_file_prefix = "qlora"

    val_loss_file = (
        training_config.log_dir
        / f"{loss_file_prefix}_validation_loss_{dtype_str}_overfit_{training_config.overfit}_compile_{training_config.compile}.csv"
    )
    train_loss_file = (
        training_config.log_dir
        / f"{loss_file_prefix}_train_loss_{dtype_str}_overfit_{training_config.overfit}_compile_{training_config.compile}.csv"
    )
    logging.info(f"val_loss_file: {val_loss_file}")
    logging.info(f"train_loss_file: {train_loss_file}")

    sync_func = (
        torch.compile(sync_float8_amax_and_scale_history)
        if training_config.compile
        else sync_float8_amax_and_scale_history
    )
    with profile_context as p:
        for iter_num in range(hyper_params.max_iters):
            lr = get_lr(iter_num, hyper_params)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Sync the amax and scale history for the fp8 linear layers at the start of every iteration
            if linear_requires_sync(fp8_linear_type):
                sync_func(model)

            input_ids, targets = next(train_iter)
            input_ids = input_ids.pin_memory().to(training_config.device)
            targets = targets.pin_memory().to(training_config.device)
            is_accumulating = (iter_num + 1) % hyper_params.gradient_accumulation_iters != 0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

            # Calculate the loss
            loss = calculate_loss(logits, targets)

            # Scale the loss by grad_accumulation iters
            (loss / hyper_params.gradient_accumulation_iters).backward()

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            if not is_accumulating and step_count % training_config.eval_interval == 0:
                t0 = time.time()
                val_loss = validate(model, val_data, val_loss_file, training_config, step_count)
                t1 = time.time() - t0
                logging.info(
                    f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms"
                )

            if not is_accumulating and step_count % training_config.save_interval == 0:
                checkpoint_path = training_config.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                torch.save(checkpoint_path, {"model": model})

            if iter_num % training_config.log_interval == 0:
                # loss.item causes a sync so we update the progress bar sporadically
                write_loss_to_file(train_loss_file, step_count, loss.item())
                progress_bar.set_postfix_str(f"Loss {loss.item():.4f}")
            progress_bar.update(1)

            if training_config.profile and iter_num < 103:
                # We want to profile iters 100-102 of the model training
                p.step()


def calculate_loss(logits, targets):
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
    return loss


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def validate(
    model: Transformer,
    val_data: DataLoader,
    loss_file: Path,
    training_config: TrainingConfig,
    training_iter: int,
) -> torch.Tensor:
    logging.info("Validating ...")
    model.eval()
    val_iter = iter(val_data)
    losses = torch.zeros(training_config.eval_iters)
    for k in tqdm(range(training_config.eval_iters)):
        input_ids, targets = next(val_iter)
        input_ids = input_ids.pin_memory().to(training_config.device)
        targets = targets.pin_memory().to(training_config.device)
        logits = model(input_ids)
        loss = calculate_loss(logits, targets)
        losses[k] = loss

    val_loss = losses.mean()
    model.train()
    write_loss_to_file(loss_file, training_iter, loss.item())
    return val_loss.item()


def load_datasets(hyper_params: Hyperparameters, training_config: TrainingConfig):
    train_data = Dataset(
        str(training_config.data_dir / "train.bin"),
        max_seq_length=hyper_params.max_seq_length,
        training_config=training_config,
    )
    val_data = Dataset(
        str(training_config.data_dir / "val.bin"),
        max_seq_length=hyper_params.max_seq_length,
        training_config=training_config,
    )
    return train_data, val_data


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int, training_config: TrainingConfig):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length
        self.overfit = training_config.overfit
        self.deterministic_data_loading = training_config.deterministic_data_loading
        self.index = 0

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            if self.overfit:
                i = 0
            else:
                if self.deterministic_data_loading:
                    i = self.index
                    self.index += self.max_seq_length
                else:
                    i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, hyper_params: Hyperparameters):
    if not hyper_params.decay_lr:
        return hyper_params.learning_rate
    # 1) linear warmup for warmup_iters steps
    if it < hyper_params.warmup_iters:
        return hyper_params.learning_rate * it / hyper_params.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > hyper_params.lr_decay_iters:
        return hyper_params.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - hyper_params.warmup_iters) / (
        hyper_params.lr_decay_iters - hyper_params.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return hyper_params.min_lr + coeff * (hyper_params.learning_rate - hyper_params.min_lr)


def entrypoint(
    fp8_linear_type: LinearType | None = None,
    compile: bool = False,
    overfit: bool = False,
    profile: bool = False,
):
    assert isinstance(fp8_linear_type, str) or fp8_linear_type is None, (
        "fp8_linear_type must be str"
    )
    assert isinstance(compile, bool), "compile must be bool"
    assert isinstance(overfit, bool), "overfit must be bool"
    assert isinstance(profile, bool), "profile must be bool"

    if overfit:
        batch_size = 1
    else:
        batch_size = 128
    hyper_params = Hyperparameters(batch_size=batch_size, fp8_linear_type=fp8_linear_type)
    training_config = TrainingConfig(compile=compile, overfit=overfit, profile=profile)
    main(hyper_params, training_config)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    Fire(entrypoint)
