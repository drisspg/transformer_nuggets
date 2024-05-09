"""
Used to train a model from scratch on big dense blocks of text data using causal attention.
"""

import argparse
import functools
import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformer_nuggets.llama.train
import transformer_nuggets.quant.qlora as qlora
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformer_nuggets.llama.model import ModelArgs, Transformer, TransformerBlock
from transformer_nuggets.llama.train import (
    calculate_loss,
    get_lr,
    log_num_params,
    write_loss_to_file,
)


logging.basicConfig(level=logging.INFO)


@dataclass
class Hyperparameters(transformer_nuggets.llama.train.Hyperparameters):
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class TrainingConfig(transformer_nuggets.llama.train.TrainingConfig):
    log_interval: int = 10
    track_max_memory: bool = False
    use_fsdp2: bool = False


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
        hyper_params.micro_batch_size,
        hyper_params.max_seq_length,
        training_config.device,
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

    log_num_params(model)

    if world_size > 1:
        if training_config.use_fsdp2:
            # move import to top when fsdp2 is landed
            from torch.distributed._composable.fsdp import fully_shard

            fully_shard_fn = functools.partial(
                fully_shard,
                reshard_after_forward=True,
            )
            for layer in model.layers:
                fully_shard_fn(layer)
            fully_shard_fn(model)
        else:
            model = FSDP(
                model,
                use_orig_params=True,
                auto_wrap_policy=ModuleWrapPolicy([TransformerBlock]),
            )

    if training_config.compile:
        model = torch.compile(model)

    if rank == 0:
        logging.info(model)

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
        rank,
        world_size,
    )


def entrypoint(
    profile: bool = False,
    use_fsdp2: bool = False,
    model_name: str = "7B",
    rank: int = 0,
    world_size: int = 1,
):
    if profile:
        torch.cuda.memory._record_memory_history(max_entries=100000)
    batch_size = int(128 / world_size)
    assert isinstance(profile, bool), "profile must be bool"
    hyper_params = Hyperparameters(batch_size=batch_size)
    training_config = TrainingConfig(
        profile=profile,
        device=torch.device(f"cuda:{rank}"),
        use_fsdp2=use_fsdp2,
        model_name=model_name,
    )
    main(hyper_params, training_config, rank, world_size)


def fsdp_main(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    entrypoint(*args, rank=rank, world_size=world_size)
    dist.destroy_process_group()


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
    profile_context = get_profile_context(hyper_params, training_config, rank)
    train_iter = iter(train_data)

    dtype_str = "bf16"

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

    this_batch_loss = torch.tensor(0.0, device=training_config.device)
    this_batch_n = 0
    fsdp_loss = torch.zeros(2, device=training_config.device)

    with profile_context as p:
        for iter_num in range(hyper_params.max_iters):
            lr = get_lr(iter_num, hyper_params)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids, targets = next(train_iter)
            input_ids = input_ids.pin_memory().to(training_config.device)
            targets = targets.pin_memory().to(training_config.device)
            is_accumulating = (iter_num + 1) % hyper_params.gradient_accumulation_iters != 0

            if iter_num % hyper_params.gradient_accumulation_iters == 0:
                with torch.no_grad():
                    this_batch_loss.fill_(0)
                this_batch_n = 0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

            # TODO(future): fix this condition, eval currently only happens
            # if eval_interval and batch_size are multiples of each other
            if not is_accumulating and step_count % training_config.eval_interval == 0:
                pass
                # TODO: add validation loop
                # t0 = time.time()
                # val_loss = validate(
                #     model, val_data, val_loss_file, training_config, step_count, rank, world_size
                # )
                # t1 = time.time() - t0
                # if rank == 0:
                #     logging.info(
                #         f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms"
                #     )

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
                    mem_stats = torch.cuda.memory_stats()
                    peak_active = mem_stats["active_bytes.all.peak"] / (1024 * 1024)
                    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024 * 1024)
                    logging.info(
                        f"iter={iter_num} max_iters={hyper_params.max_iters} loss={loss_val:.4f} Peak Active Memory: {peak_active} MB, Peak Reserve Memory: {peak_reserved} MB"
                    )

            if training_config.profile:
                if iter_num < 103:
                    # dump profile traces in iters 100-102
                    p.step()
                if iter_num == 1:
                    # check memory during both init and train
                    memory_trace_path = str(
                        training_config.log_dir
                        / f"mem_trace_llama_{training_config.model_name}_{dtype_str}_rank_{rank}.pickle"
                    )
                    torch.cuda.memory._dump_snapshot(f"{memory_trace_path}")
                    torch.cuda.memory._record_memory_history(enabled=None)
                    logging.info(f"Wrote memory traces to: {memory_trace_path}")


class Dataset(IterableDataset):
    def __init__(
        self,
        data_file: Path,
        hyper_params: Hyperparameters,
        training_config: TrainingConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = hyper_params.max_seq_length
        self.max_iters = hyper_params.max_iters
        self.overfit = training_config.overfit
        self.deterministic_data_loading = training_config.deterministic_data_loading
        self.index = 0
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        per_rank = int(self.max_iters / float(self.world_size))
        rank_offset = self.rank * per_rank
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
        hyper_params=hyper_params,
        training_config=training_config,
        rank=rank,
        world_size=world_size,
    )
    val_data = Dataset(
        str(training_config.data_dir / "val.bin"),
        hyper_params=hyper_params,
        training_config=training_config,
        rank=rank,
        world_size=world_size,
    )
    return train_data, val_data


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
    parser.add_argument("--use_fsdp2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default="7B")
    args = parser.parse_args()
    fsdp_num_gpus = args.fsdp_num_gpus
    inner_args = (args.profile, args.use_fsdp2, args.model)

    if fsdp_num_gpus is None or fsdp_num_gpus == 1:
        entrypoint(*inner_args)
    else:
        assert fsdp_num_gpus <= torch.cuda.device_count()
        mp.spawn(fsdp_main, args=(fsdp_num_gpus, inner_args), nprocs=fsdp_num_gpus, join=True)


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def validate(
    model: Transformer,
    val_data: DataLoader,
    loss_file: Path,
    training_config: TrainingConfig,
    training_iter: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if rank == 0:
        logging.info("Validating ...")
    model.eval()
    val_iter = iter(val_data)
    losses = torch.zeros(training_config.eval_iters).to(rank)
    iterator = range(training_config.eval_iters)
    if rank == 0:
        iterator = tqdm(iterator)
    for k in iterator:
        input_ids, targets = next(val_iter)
        input_ids = input_ids.pin_memory().to(training_config.device)
        targets = targets.pin_memory().to(training_config.device)
        logits = model(input_ids)
        loss = calculate_loss(logits, targets)
        losses[k] = loss

    if world_size > 1:
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses = losses / world_size

    val_loss = losses.mean()
    model.train()
    write_loss_to_file(loss_file, training_iter, loss.item())
    return val_loss.item()


def get_profile_context(
    hyper_params: Hyperparameters, train_config: TrainingConfig, rank: int = 0
):
    """Returns a context manager that can be used to profile the model."""

    def trace_handler(prof):
        fp8_linear_type = hyper_params.fp8_linear_type

        dtype_str = fp8_linear_type if fp8_linear_type else "bf16"
        output_str = str(train_config.log_dir / f"trace_llama_7b_hf_{dtype_str}_rank_{rank}.json")
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
            with_flops=True,
            profile_memory=True,
            experimental_config=torch.profiler._ExperimentalConfig(
                enable_cuda_sync_events=True,
            ),
        )
        return context
    else:
        return nullcontext()
