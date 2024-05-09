import logging
from pathlib import Path

import torch
import torch.nn as nn
from jsonargparse import CLI

from transformer_nuggets.quant.qlora import (
    get_mlp_weights,
    get_sample_inputs,
    MLP,
    QloraMLP,
)
from transformer_nuggets.utils.benchmark import save_memory_snapshot

logging.basicConfig(level=logging.INFO)


def get_mlp_stack(use_qlora: bool = False) -> torch.nn.Module:
    layers = []
    for _ in range(10):
        if use_qlora:
            mlp = QloraMLP(*get_mlp_weights(8192))
        else:
            mlp = MLP(*get_mlp_weights(8192))
        layers.append(mlp)
    model = nn.ModuleList(layers)
    model.to(device="cuda", dtype=torch.bfloat16)
    return model


def main(output_folder: Path, use_qlora: bool = False, compile: bool = False):
    """Generate memory traces for main MLP and Qlora MLP

    Args:
        output_folder: Path to write out memory viz trace. Should be a directory since we ill save multiple files
        compile: Whether to use torch.compile
    """
    model = get_mlp_stack(use_qlora)

    x = get_sample_inputs(1, 8, 8192, "cuda", requires_grad=False)

    if compile:
        model = torch.compile(model)
        with torch.no_grad():
            model(x)  # warmup

    assert output_folder.is_dir(), f"output_folder {output_folder} should be a directory"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with save_memory_snapshot(output_folder / ("qlora_mlp" if use_qlora else "main_mlp")):
        for epoch in range(2):
            x = get_sample_inputs(1, 8, 8192, "cuda", requires_grad=False)
            for layer in model:
                x = layer(x)

            x.backward(torch.ones_like(x))
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python benchmarks/qlora_memory_trace.py benchmarks/data --use_qlora True
    """
    CLI(main)
