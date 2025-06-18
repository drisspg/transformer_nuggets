"""Iterate through torchbench models and collect info on shapes of inputs and outputs"""

import ast

import logging
import pickle as pkl
from collections import Counter, defaultdict

from pathlib import Path


import torch
import torch.overrides
from rich import print as rprint
from torch.fx.operator_schemas import normalize_function
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from transformer_nuggets.utils.benchmark import bcolors

from transformer_nuggets.utils.tracing import abbr_to_dtype, dtype_abbrs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ShapeLog(TorchDispatchMode):
    """Example usage:
    with LoggingMode():
        torch.nn.functional.linear(torch.randn(3, 4), torch.randn(5, 4))
    """

    def __init__(self, log_path, with_type: bool = True, specific_ops=None):
        self.with_type = with_type
        self.log_path = log_path
        self.logs = defaultdict(Counter)
        self.specific_ops = specific_ops

    def _fmt(self, a: object) -> str:
        if isinstance(a, torch.Tensor):
            maybe_type = ""
            shape_str = f"[{','.join(map(str, a.shape))}]"
            if self.with_type:
                maybe_type = dtype_abbrs[a.dtype]
            return f"{maybe_type}{shape_str}"
        else:
            return a

    def fmt_shape(self, kwargs: dict, with_type: bool = False) -> str:
        """This formats the tensor args and output into a string that is easy to parse
        Specifically:
         The input string will be broken up into two sections seperated by a ->
            - The first section will contain the input shapes of the tensors in the format
            <kwarg_name>=[shape1],<kwarg_name>=[shape2],...,[shapeN]
            - The second section will contain the output shape of the tensor in the format
            [shape]

        if with_type is True, then the shapes will be formatted with their respective types
            - Example: float32[1, 2, 3]
        """
        input_str = "|".join(f"{k}:{tree_map(self._fmt, v)}" for k, v in kwargs.items())
        return input_str

    def print_logs(self):
        rprint(self.logs)

    def save_to_disk(self, path: Path):
        with open(path, "wb") as f:
            pkl.dump(self.logs, f)
        logger.info(f"💾 Trace file 📄 saved to: {bcolors.OKGREEN}{path}{bcolors.ENDC}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Only log the specific ops if they are provided
        if self.specific_ops is not None and func not in self.specific_ops:
            return func(*args, **kwargs)
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)

        # Convert all to kwargs
        _, new_kwargs = normalize_function(
            func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        fmt_args = self.fmt_shape(new_kwargs)
        delimiter = "->"
        fmt_rets = tree_map(self._fmt, rs)
        log_msg = f"{fmt_args}{delimiter}{fmt_rets}"

        self.logs[str(func)][log_msg] += 1
        return rs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_to_disk(self.log_path)
        return super().__exit__(exc_type, exc_val, exc_tb)


def open_logs(path: Path):
    """Opens the logs file and returns the logs dict object"""
    with open(path, "rb") as f:
        return pkl.load(f)


def construct_input(
    log: dict,
    op,
    device: torch.device,
    default_dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    most_common: int | None = None,
):
    """Parses the log string and returns the input and output shapes
    Args:
        log: The log dictionary
        op: The name of the op
        device: The device to put the tensors on
        default_dtype: The default dtype of the tensors if not dype
            was saved during logging. Defaults to torch.float32.
        requires_grad: Whether the tensors require grad. Defaults to False.
        most_common: The number of most common shapes to return. Defaults to None.

    Returns:
        A list of dictionaries containing the inputs that can be used with the specified op
        Up to the N most common entries
    """
    input_to_self = lambda name: name if name != "input" else "self"
    op_entries = log[str(op)].most_common(most_common)
    op_inpts = []
    for entry in op_entries:
        input_str, output_str = entry[0].split("->")
        input_str = input_str.split("|")
        input_dict = {}
        for pair in input_str:
            k, v = pair.split(":")
            # If '[' is in the value, then it is a tensor
            if "[" in v:
                maybe_dtype_tuple = v.split("[")
                maybe_dtype = maybe_dtype_tuple[0]
                shape = "[" + maybe_dtype_tuple[1]
                shape = ast.literal_eval(shape)
                dtype = abbr_to_dtype[maybe_dtype] if maybe_dtype else default_dtype
                input_dict[input_to_self(k)] = torch.rand(
                    shape, dtype=dtype, device=device, requires_grad=requires_grad
                )
            else:
                input_dict[k] = ast.literal_eval(v)
        op_inpts.append(input_dict)
    return op_inpts
