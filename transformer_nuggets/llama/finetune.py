"""
Used to train a model from scratch on big dense blocks of text data using causal attention.
"""
import argparse
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
import torch.distributed as dist
import torch.multiprocessing as mp
import transformer_nuggets.llama.train
import transformer_nuggets.quant.qlora as qlora
from fire import Fire
from float8_experimental.float8_linear_utils import (
    linear_requires_sync,
    LinearType,
    swap_linear_with_float8_linear,
    sync_float8_amax_and_scale_history,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformer_nuggets.llama.model import ModelArgs, Transformer, TransformerBlock
from transformer_nuggets.llama.train import (
    calculate_loss,
    get_lr,
    get_profile_context,
    log_num_params,
)

logging.basicConfig(level=logging.INFO)


TRAIN_DATASET_SIZE = 60000


@dataclass
class Hyperparameters:
    learning_rate: float = 6e-4
    batch_size: int = 128
    micro_batch_size: int = 1
    gradient_accumulation_iters: int = field(init=False)
    max_seq_length: int = 4096
    max_iters: int = TRAIN_DATASET_SIZE  # train dataset size
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = field(init=False)
    min_lr: float = 6e-5
    foreach_optimizer: bool = False
    use_te_linear: bool = False

    # Float8 Specific Config
    # We want to skip the first embedding layer since scaled_mm needs to multiple of 16
    float8_skip_list: List[str] = field(default_factory=lambda: ["tok_embeddings"])
    fp8_linear_type: Optional[LinearType] = None
    weight_caching: bool = False

    # qlora config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self):
        self.gradient_accumulation_iters = self.batch_size // self.micro_batch_size
        self.lr_decay_iters = self.max_iters
        assert self.gradient_accumulation_iters > 0
        if self.fp8_linear_type is not None:
            self.fp8_linear_type = LinearType[self.fp8_linear_type.upper()]
        if self.use_te_linear:
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe
            from transformer_engine.common.recipe import DelayedScaling, Format

            fp8_format = Format.HYBRID
            self.fp8_recipe = DelayedScaling(
                fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
            )


@dataclass
class TrainingConfig:
    eval_interval: int = 60
    save_interval: int = 100
    eval_iters: int = 40
    # it's convenient for log_interval to equal batch_size
    log_interval: int = 20
    val_step_count: int = 0
    deterministic_data_loading: bool = True

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
    track_max_memory: bool = False


class Dataset(IterableDataset):
    def __init__(
        self,
        data_file: Path,
        max_seq_length: int,
        training_config: TrainingConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length
        self.overfit = training_config.overfit
        self.deterministic_data_loading = training_config.deterministic_data_loading
        self.index = 0
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")

        # TODO(later): look into whether map dataset will work here, to clean
        # up the slicing logic

        # ensure multiple Datasets in distributed training all get different data
        # slices when deterministic data loading is enabled
        per_rank = int(TRAIN_DATASET_SIZE / float(self.world_size))
        rank_offset = self.rank * per_rank

        # ensure multiple workers all get different data slices when deterministic
        # data loading is enabled
        # source: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "single process data loading not implemented yet"
        per_worker = int(per_rank / float(worker_info.num_workers))
        worker_id = worker_info.id
        worker_offset = worker_id * per_worker

        while True:
            if self.overfit:
                i = 0
            else:
                if self.deterministic_data_loading:
                    i = self.index + rank_offset + worker_offset
                    self.index += self.max_seq_length
                else:
                    i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y


def load_datasets(
    hyper_params: Hyperparameters,
    training_config: TrainingConfig,
    rank: int,
    world_size: int,
):
    train_data = Dataset(
        str(training_config.data_dir / "train.bin"),
        max_seq_length=hyper_params.max_seq_length,
        training_config=training_config,
        rank=rank,
        world_size=world_size,
    )
    val_data = Dataset(
        str(training_config.data_dir / "val.bin"),
        max_seq_length=hyper_params.max_seq_length,
        training_config=training_config,
        rank=rank,
        world_size=world_size,
    )
    return train_data, val_data


def main(
    hyper_params: Hyperparameters,
    training_config: TrainingConfig,
    rank: int,
    world_size: int,
):
    torch.cuda.set_device(rank)

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    os.makedirs(training_config.out_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)

    # Setup Model
    model_args = ModelArgs.from_name(training_config.model_name)
    if rank == 0:
        logging.info("Initializing model")
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

    if rank == 0:
        logging.info("Setting up the dataloaders")
    train_data, val_data = load_datasets(hyper_params, training_config, rank, world_size)
    train_dataloader = DataLoader(
        train_data,
        batch_size=hyper_params.micro_batch_size,
        num_workers=2,
    )
    val_dataloader = DataLoader(val_data, batch_size=hyper_params.micro_batch_size, num_workers=2)

    fp8_linear_type = hyper_params.fp8_linear_type
    if fp8_linear_type is not None:
        fp8_module = LINEAR_TYPE_MAP[fp8_linear_type]
        if hyper_params.weight_caching:
            if rank == 0:
                logging.info("Using weight caching")
            fp8_config.allocate_float8_weight_cache_buffers = True
        swap_linear_with_float8_linear(
            model, fp8_module, skip_fqn_list=hyper_params.float8_skip_list
        )
    if hyper_params.use_te_linear:
        swap_linear_with_te_linear(model, hyper_params, training_config)

    log_num_params(model)

    if world_size > 1:
        # if we are in distributed, assume we want FSDP
        model = FSDP(
            model,
            # use_orig_params is required for torch.compile
            use_orig_params=True,
            # for a quick and dirty wrapping strategy, just wrap
            # each transformer block
            auto_wrap_policy=ModuleWrapPolicy([TransformerBlock]),
        )

    if training_config.compile:
        model = torch.compile(model)

    if rank == 0:
        print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        rank,
        world_size,
    )


def train(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    hyper_params: Hyperparameters,
    training_config: TrainingConfig,
    rank: int,
    world_size: int,
) -> None:
    """Lets go!"""
    step_count = 0

    model.train()
    profile_context = get_profile_context(hyper_params, training_config)
    train_iter = iter(train_data)

    # Sanity check
    fp8_linear_type = hyper_params.fp8_linear_type
    dtype_str = fp8_linear_type if fp8_linear_type else "bf16"

    val_loss_file = (
        training_config.log_dir
        / f"qlora_validation_loss_{dtype_str}_overfit_{training_config.overfit}_compile_{training_config.compile}_{rank}.csv"
    )
    train_loss_file = (
        training_config.log_dir
        / f"qlora_train_loss_{dtype_str}_overfit_{training_config.overfit}_compile_{training_config.compile}_{rank}.csv"
    )
    if rank == 0:
        logging.info(f"val_loss_file: {val_loss_file}")
        logging.info(f"train_loss_file: {train_loss_file}")

    sync_func = (
        torch.compile(sync_float8_amax_and_scale_history)
        if training_config.compile
        else sync_float8_amax_and_scale_history
    )

    this_batch_loss = torch.tensor(0.0, device=training_config.device)
    this_batch_n = 0
    fsdp_loss = torch.zeros(2, device=training_config.device)

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

            if iter_num % hyper_params.gradient_accumulation_iters == 0:
                with torch.no_grad():
                    this_batch_loss.fill_(0)
                this_batch_n = 0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if hyper_params.use_te_linear:
                    import transformer_engine.pytorch as te

                    if hyper_params.weight_caching:
                        global WEIGHT_CACHING_ON
                        if (
                            iter_num > 0
                            and (iter_num % hyper_params.gradient_accumulation_iters) != 0
                        ):
                            WEIGHT_CACHING_ON = True
                        else:
                            WEIGHT_CACHING_ON = False

                    with te.fp8_autocast(enabled=True, fp8_recipe=hyper_params.fp8_recipe):
                        logits = model(input_ids)
                else:
                    logits = model(input_ids)

            # Calculate the loss
            loss = calculate_loss(logits, targets)
            with torch.no_grad():
                this_batch_loss += loss
            this_batch_n += len(input_ids)

            # Scale the loss by grad_accumulation iters
            (loss / hyper_params.gradient_accumulation_iters).backward()

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                # We have just updated the weights so we disable weight_caching for the next iteration
                if hyper_params.weight_caching:
                    fp8_config.weight_cache_enabled = False

            if is_accumulating and hyper_params.weight_caching:
                # We want to enable weight caching for the next iteration
                fp8_config.weight_cache_enabled = True

            # TODO(future): fix this condition, eval currently only happens
            # if eval_interval and batch_size are multiples of each other
            if not is_accumulating and step_count % training_config.eval_interval == 0:
                t0 = time.time()
                val_loss = validate(
                    model, val_data, val_loss_file, training_config, step_count, rank, world_size
                )
                t1 = time.time() - t0
                if rank == 0:
                    logging.info(
                        f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms"
                    )

            if not is_accumulating and step_count % training_config.save_interval == 0:
                checkpoint_path = training_config.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                torch.save(checkpoint_path, {"model": model})

            if (iter_num + 1) % training_config.log_interval == 0:
                # loss.item causes a sync so we update the progress bar sporadically
                if world_size == 1:
                    with torch.no_grad():
                        avg_loss_this_batch = this_batch_loss / this_batch_n
                    loss_val = avg_loss_this_batch
                else:
                    fsdp_loss[0] = this_batch_loss
                    fsdp_loss[1] = this_batch_n
                    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
                    loss_val = fsdp_loss[0] / fsdp_loss[1]

                write_loss_to_file(train_loss_file, step_count, loss_val)

                if rank == 0:
                    logging.info(
                        f"iter={iter_num} max_iters={hyper_params.max_iters} loss={loss_val:.4f}"
                    )

            if training_config.profile and iter_num < 103:
                # We want to profile iters 100-102 of the model training
                p.step()

            if training_config.track_max_memory and rank == 0:
                print(
                    "iter_num",
                    iter_num,
                    "mem usage GB",
                    float(torch.cuda.max_memory_allocated()) / 1024 / 1024 / 1024,
                )
            torch.cuda.reset_peak_memory_stats()


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


def entrypoint(
    profile: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    batch_size = int(128 / world_size)
    assert isinstance(profile, bool), "profile must be bool"
    hyper_params = Hyperparameters(
        batch_size=batch_size,
        max_iters=int(TRAIN_DATASET_SIZE / world_size),
    )
    training_config = TrainingConfig(
        profile=profile,
        device=torch.device(f"cuda:{rank}"),
    )
    main(hyper_params, training_config, rank, world_size)


def fsdp_main(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    entrypoint(*args, rank=rank, world_size=world_size)
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Native PyTorch LLaMa trainer")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fsdp_num_gpus",
        type=int,
        default=1,
        help="if specified, runs FSDP with this many GPUs on a single host",
    )
    args = parser.parse_args()
    fsdp_num_gpus = args.fsdp_num_gpus
    inner_args = (args.profile,)

    if fsdp_num_gpus is None:
        # single host single GPU
        entrypoint(*inner_args)
    else:
        # single host multi GPU FSDP
        assert fsdp_num_gpus <= torch.cuda.device_count()
        mp.spawn(fsdp_main, args=(fsdp_num_gpus, inner_args), nprocs=fsdp_num_gpus, join=True)
