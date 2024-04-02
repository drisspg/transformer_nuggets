import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Union
import math
import torch

logging.basicConfig(level=logging.INFO)

bnb_available = False

aten = torch.ops.aten
NF4_OPS_TABLE: Dict[Any, Any] = {}

def implements(aten_ops):
    """Register aten ops to the float8 op table"""

    def decorator(func):
        for op in aten_ops:
            NF4_OPS_TABLE[op] = func
        return func

    return decorator

@implements(
    [
        aten.detach.default,
    ]
)
def nf4_detach(aten_op, args, kwargs=None):
    # nn.Parameter need detach
    quantized_scalers = aten_op(args[0].quantized_scalers, *args[1:], **kwargs)
    quantization_factor = aten_op(args[0].quantization_factor, *args[1:], **kwargs)
    quantized_data = aten_op(args[0].quantized_data, *args[1:], **kwargs)
    scaler_mean = aten_op(args[0].scaler_mean, *args[1:], **kwargs)
    nf4 = aten_op(args[0].nf4, *args[1:], **kwargs)
    tensor_meta = SubclassTensorArgs(
        args[0].size(),
        args[0].stride(),
        args[0].storage_offset(),
        args[0].dtype,
        args[0].device,
        args[0].requires_grad,
    )
    return NF4Tensor(
        tensor_meta,
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        quantized_scalers,
        quantization_factor,
        scaler_mean,
        quantized_data,
        nf4,
    )

@implements(
    [
        aten.split.Tensor,
    ]
)
def nf4_split(aten_op, args, kwargs=None):
    # torch.chunk
    # TODO: find if there are other args/kwargs in aten.split
    assert len(args) == 2 and (kwargs is None or len(kwargs) == 0), "only support aten.split.Tensor with 2 args"
    # TODO: assert on dim-0 sharding. how to get dim from torch.chunk?
    num_chunks = args[0].size(0) // args[1]

    # TODO: assert numel % num_chunks == 0
    quantized_scalers_chunks = aten_op(args[0].quantized_scalers, args[0].quantized_scalers.numel() // num_chunks, **kwargs)
    quantization_factor_chunks = aten_op(args[0].quantization_factor, args[0].quantization_factor.numel() // num_chunks, **kwargs)
    quantized_data_chunks = aten_op(args[0].quantized_data, args[0].quantized_data.numel() // num_chunks, **kwargs)

    assert len(args) == 2, "only support 2d because of tensor meta"
    return [
        NF4Tensor(
            SubclassTensorArgs(
                (args[0].size(0) // num_chunks, args[0].size(1)),
                args[0].stride(),
                args[0].storage_offset(),
                args[0].dtype,
                args[0].device,
                args[0].requires_grad,
            ),
            args[0].block_size,
            args[0].n_blocks,
            args[0].scaler_block_size,
            quantized_scalers,
            quantization_factor,
            args[0].scaler_mean,
            quantized_data,
            args[0].nf4,
        ) for quantized_scalers, quantization_factor, quantized_data in zip(
            quantized_scalers_chunks, quantization_factor_chunks, quantized_data_chunks
        )
    ]

@implements(
    [
        aten.new_zeros.default,
    ]
)
def nf4_new_zeros(aten_op, args, kwargs=None):
    assert len(args[0].shape) == 2 and len(args[1]) == 2, "only support new zeros on 2D"
    assert args[0].numel() % math.prod(args[1]) == 0
    ratio = args[0].numel() // math.prod(args[1])

    assert args[0].quantized_scalers.size(0) % ratio == 0, f"quantized_scalers.numel() must be divisible by {ratio}"
    quantized_scalers_new_zeros = aten_op(args[0].quantized_scalers, [args[0].quantized_scalers.size(0) // ratio], **kwargs)

    assert args[0].quantization_factor.size(0) % ratio == 0, f"quantization_factor.size(0) must be divisible by {ratio}"
    quantization_factor_new_zeros = aten_op(args[0].quantization_factor, [args[0].quantization_factor.size(0) // ratio], **kwargs)

    assert args[0].quantized_data.size(0) % ratio == 0, f"quantized_data.size(0) must be divisible by {ratio}"
    quantized_data_new_zeros = aten_op(args[0].quantized_data, [args[0].quantized_data.size(0) // ratio], **kwargs)

    return NF4Tensor(
        SubclassTensorArgs(
            (args[1][0], args[1][1]),
            args[0].stride(),
            args[0].storage_offset(),
            args[0].dtype,
            args[0].device,
            args[0].requires_grad,
        ),
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        quantized_scalers_new_zeros,
        quantization_factor_new_zeros,
        args[0].scaler_mean,
        quantized_data_new_zeros,
        args[0].nf4,
    )

@implements(
    [
        aten.slice.Tensor,
    ]
)
def nf4_slice(aten_op, args, kwargs=None):
    assert len(args) == 4
    assert args[1] == 0, f"only support dim=0 but got dim={args[1]}"
    # TODO: maybe relax?
    assert args[2] == 0, f"only support start=0 but got start={args[2]}"
    assert args[3] == args[0].size(0), f"only support end == size(0) but got end={args[3]} and size(0)={args[0].size(0)}"
    return NF4Tensor(
        SubclassTensorArgs(
            args[0].size(),
            args[0].stride(),
            args[0].storage_offset(),
            args[0].dtype,
            args[0].device,
            args[0].requires_grad,
        ),
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        args[0].quantized_scalers,
        args[0].quantization_factor,
        args[0].scaler_mean,
        args[0].quantized_data,
        args[0].nf4,
    )

@implements(
    [
        aten.copy_.default,
    ]
)
def nf4_copy_(aten_op, args, kwargs=None):
    assert len(args) == 2 and (kwargs is None or len(kwargs) == 0), "only support aten.copy_.default with 2 args"
    quantized_scalers = aten_op(args[0].quantized_scalers, args[1].quantized_scalers, **kwargs)
    quantization_factor = aten_op(args[0].quantization_factor, args[1].quantization_factor, **kwargs)
    quantized_data = aten_op(args[0].quantized_data, args[1].quantized_data, **kwargs)
    scaler_mean = aten_op(args[0].scaler_mean, args[1].scaler_mean, **kwargs)
    nf4 = aten_op(args[0].nf4, args[1].nf4, **kwargs)
    tensor_meta = SubclassTensorArgs(
        args[1].size(),
        args[1].stride(),
        args[1].storage_offset(),
        args[1].dtype,
        args[1].device,
        args[1].requires_grad,
    )
    return NF4Tensor(
        tensor_meta,
        args[1].block_size,
        args[1].n_blocks,
        args[1].scaler_block_size,
        quantized_scalers,
        quantization_factor,
        scaler_mean,
        quantized_data,
        nf4,
    )

@implements(
    [
        aten.view.default,
    ]
)
def nf4_view(aten_op, args, kwargs=None):
    assert len(args) == 2, args[1] == -1
    quantized_scalers = aten_op(args[0].quantized_scalers, *(args[1:]), **kwargs)
    quantization_factor = aten_op(args[0].quantization_factor, *(args[1:]), **kwargs)
    quantized_data = aten_op(args[0].quantized_data, *(args[1:]), **kwargs)
    tensor_meta = SubclassTensorArgs(
        [args[0].numel()],
        (1, ),
        args[0].storage_offset(),
        args[0].dtype,
        args[0].device,
        args[0].requires_grad,
    )
    return NF4Tensor(
        tensor_meta,
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        quantized_scalers,
        quantization_factor,
        args[0].scaler_mean,
        quantized_data,
        args[0].nf4,
    )

@implements(
    [
        aten.as_strided.default,
    ]
)
def nf4_as_strided(aten_op, args, kwargs=None):
    assert len(args[1]) == 2 and math.prod(args[1]) == args[0].numel(), "only support same numel"
    assert args[2] == [args[1][1], 1], f"only support stride {[args[1][1], 1]}"
    assert args[0].storage_offset() == args[3], f"only support same storage offset"
    return NF4Tensor(
        SubclassTensorArgs(
            torch.Size(args[1]),
            tuple(args[2]),
            args[0].storage_offset(),
            args[0].dtype,
            args[0].device,
            args[0].requires_grad,
        ),
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        args[0].quantized_scalers,
        args[0].quantization_factor,
        args[0].scaler_mean,
        args[0].quantized_data,
        args[0].nf4,
    )

@implements(
    [
        aten._to_copy.default,
    ]
)
def nf4_to_copy(aten_op, args, kwargs=None):
    quantized_data_kwargs = kwargs
    quantized_data_kwargs['dtype'] = args[0].quantized_data.dtype
    quantized_data = aten_op(args[0].quantized_data, *(args[1:]), **quantized_data_kwargs)

    return NF4Tensor(
        SubclassTensorArgs(
            args[0].size(),
            args[0].stride(),
            args[0].storage_offset(),
            args[0].dtype,
            kwargs['device'],
            args[0].requires_grad,
        ),
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        args[0].quantized_scalers,
        args[0].quantization_factor,
        args[0].scaler_mean,
        quantized_data,
        args[0].nf4,
    )


@implements(
    [
        aten.is_pinned.default,
    ]
)
def nf4_is_pinned(aten_op, args, kwargs=None):
    return aten_op(args[0].quantized_data, *(args[1:]), **kwargs)


@implements(
    [
        aten._pin_memory.default,
    ]
)
def nf4_pin_memory(aten_op, args, kwargs=None):
    quantized_data = aten_op(args[0].quantized_data, *(args[1:]), **kwargs)

    return NF4Tensor(
        SubclassTensorArgs(
            args[0].size(),
            args[0].stride(),
            args[0].storage_offset(),
            args[0].dtype,
            args[0].device,
            args[0].requires_grad,
        ),
        args[0].block_size,
        args[0].n_blocks,
        args[0].scaler_block_size,
        args[0].quantized_scalers,
        args[0].quantization_factor,
        args[0].scaler_mean,
        quantized_data,
        args[0].nf4,
    )


@dataclass
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


def get_block_absmax(inpt_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Iterate through a flattened tensor getting the absmax scalers for each block

    Args:
        inpt_tensor: Input tensor to get scalers for
        block_size: Block size for the scanning window
    Returns:
        torch.Tensor: Tensor of scalers for each block
    """
    assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
    assert (
        inpt_tensor.numel() % block_size
    ) == 0, (
        f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {block_size}"
    )

    n_blocks = inpt_tensor.numel() // block_size
    blocks = inpt_tensor.view(n_blocks, block_size)
    block_scalers = blocks.abs().max(dim=1).values
    return block_scalers


class NF4Tensor(torch.Tensor):
    """NF4Tensor class for converting a weight to the QLoRA NF4 format"""

    def __new__(
        cls,
        # Args related for base tensor construction
        tensor_meta: SubclassTensorArgs,
        # Args stored on the instance
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Create a new NF4Tensor object
        Args:
            tensor_meta: Metadata for the tensor
            block_size: Size of the quantization block
            n_blocks: Number of blocks to cover the full tensor
            scaler_block_size: Block size for the scalar quantization
            quantized_scalers: Quantized scalers data' represented a uint8 tensor
            quantization_factor: Quantization factor, single scalar represented as torch.Tensor
            scaler_mean: Mean of the scalers
            quantized_data: Quantized data represented as uint8 tensor
            nf4: NF4 tensor LUT for the quantization and dequantization

        """

        nf4tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor_meta.original_shape,
            tensor_meta.original_strides,
            tensor_meta.storage_offset,
            dtype=tensor_meta.dtype,
            device=tensor_meta.device,
            requires_grad=tensor_meta.requires_grad,
        )
        return nf4tensor

    def __init__(
        self,
        tensor_meta: SubclassTensorArgs,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Initialize the NF4Tensor class"""
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.scaler_block_size = scaler_block_size
        self.quantized_scalers = quantized_scalers
        self.quantization_factor = quantization_factor
        self.scaler_mean = scaler_mean
        self.quantized_data = quantized_data
        self.nf4 = nf4

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        inpt_tensor: torch.Tensor,
        block_size: int = 64,
        scaler_block_size: int = 256,
    ):
        assert inpt_tensor.dtype == torch.bfloat16
        assert (
            inpt_tensor.numel() % block_size == 0
        ), "Input tensor must be divisible by block size"
        assert inpt_tensor.dtype == torch.bfloat16, "Input tensor must be bfloat16"
        assert inpt_tensor.is_contiguous, "Input tensor must be contiguous!"
        # I think I want do this
        # assert not inpt_tensor.requires_grad, "Input tensor must not require grad"
        device = inpt_tensor.device
        # Cache the tensor on the class def
        nf4 = torch.tensor(
            [
                -1.0000,
                -0.6962,
                -0.5251,
                -0.3949,
                -0.2844,
                -0.1848,
                -0.0911,
                0.0000,
                0.0796,
                0.1609,
                0.2461,
                0.3379,
                0.4407,
                0.5626,
                0.7230,
                1.0000,
            ],
            device=device,
            dtype=torch.bfloat16,
        )
        n_blocks = inpt_tensor.numel() // block_size
        # Double quantization
        (
            quantized_scalers,
            quantization_factor,
            scaler_mean,
        ) = cls.double_quantize_scalers(inpt_tensor.flatten(), block_size, scaler_block_size)
        quantized_data = cls.convert_to_norm_float_weight(inpt_tensor, n_blocks, block_size, nf4)
        tensor_meta = SubclassTensorArgs(
            inpt_tensor.size(),
            inpt_tensor.stride(),
            inpt_tensor.storage_offset(),
            inpt_tensor.dtype,
            inpt_tensor.device,
            inpt_tensor.requires_grad,
        )
        return cls(
            tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            nf4=nf4,
        )

    @staticmethod
    def double_quantize_scalers(
        inpt_tensor: torch.Tensor,
        block_size: int,
        scaler_block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used to achieve the double quantization of the scalers
        We take the input tensor first calculate the absmax quantization factors for each block.
        We then find the mean of our positive absmax scalers. We subtract this mean from the scalers
        And then we calculate the absmax quantization factors for each block again. We then quantize the scalers to int8.

        Args:
            inpt_tensor: Input tensor to convert to QLoRA format, typically a weight tensor

        Returns:
            torch.Tensor: Tensor of per_block quantization factors stored in int8 format
                size: (n_blocks)
            torch.Tensor: Tensor of per_scaler_block quantization factors stored in int16 format
                size: (n_scaler_blocks)
        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            inpt_tensor.numel() % scaler_block_size
        ) == 0, f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {scaler_block_size}"

        # First round of quantization
        # Produces: A tensor of size (n_blocks) of inpt_tensor.dtype
        scalers_1 = get_block_absmax(inpt_tensor, block_size)
        scalers_1_mean = scalers_1.mean()
        scalers_1 = scalers_1 - scalers_1_mean
        # Second round of quantization
        assert (
            scalers_1.numel() % scaler_block_size == 0
        ), "Number of scalers must be divisible by scaler block size"
        n_scaler_blocks = scalers_1.numel() // scaler_block_size
        scaler_blocks = scalers_1.view(n_scaler_blocks, scaler_block_size)

        scaler_absmax = get_block_absmax(scalers_1, scaler_block_size)
        scaler_absmax = scaler_absmax.unsqueeze(-1).expand(n_scaler_blocks, scaler_block_size)

        quantization_factor = 256 / (2 * scaler_absmax)
        # Length equal to weight numel // block_size
        quantized_scaler_blocks = scaler_blocks * quantization_factor
        quantized_scaler_blocks = quantized_scaler_blocks.round()
        quantized_scaler_blocks = quantized_scaler_blocks.clamp(-128, 127)

        # This is needed to make sure that quantization_factor remains a repeated view of n_scaler_blocks
        # For some reason the 127/scaler_absmax realizes n_scaler entries when only n_scaler_blocks are needed
        # The following will grab the first entry for the n_scaler_blocks which is the same across the scaler_block_size
        quantization_factor = quantization_factor[:, 0]

        return (
            quantized_scaler_blocks.flatten().to(torch.int8),
            quantization_factor.view(n_scaler_blocks),
            scalers_1_mean,
        )

    def dequantize_scalers(
        self, inpt_tensor: torch.Tensor, quantization_factor: torch.Tensor, scaler_block_size: int
    ) -> torch.Tensor:
        """Used to unpack the double quantized scalers

        Args;
            inpt_tensor: Input tensor to convert to QLoRA format this is the quantized scalers in int8 format
            quantization_factor: Tensor of per_scaler_block quantization factors stored in inpt_weight.dtype
                size: (n_scaler_blocks)
            scaler_block_size: Scaler block size to use for double quantization.

        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            inpt_tensor.numel() % scaler_block_size
        ) == 0, f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {scaler_block_size}"
        n_scaler_blocks = inpt_tensor.numel() // scaler_block_size
        inpt_tensor = inpt_tensor.view(n_scaler_blocks, scaler_block_size)
        dequantized = (inpt_tensor / quantization_factor.unsqueeze(-1)).flatten().to(
            torch.bfloat16
        ) + self.scaler_mean
        return dequantized

    @staticmethod
    def convert_to_norm_float_weight(
        inpt_tensor: torch.Tensor, n_blocks: int, block_size: int, nf4: torch.tensor
    ) -> torch.Tensor:
        """Convert a tensor to the normalized float weight format"""
        flattened_tensor = inpt_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = inpt_tensor.numel()
        assert (
            numel % 2 == 0
        ), "Number of elements must be even just to not have to think about the end"
        # Reshape the flattened tensor into blocks of size self.block_size
        blocks = flattened_tensor.view(n_blocks, block_size)

        # Scale the blocks
        scalers = get_block_absmax(inpt_tensor.flatten(), block_size)
        scales = scalers.unsqueeze(-1).expand(n_blocks, block_size)
        scaled_blocks = blocks / scales

        # Returns a flattened tensor with each element quantized to nf4 index
        # The weird behavior comes here with how qlora vs bnb break nf4 ties.
        # Since we ust torch.min(nf4 - inpt/scale) we will always pick the smallest index
        # While bnb appears to be pick the larger index when breaking ties
        # ACTUALLYYY I think that what ever op bnb is using to get the nearest NF4 value
        # Is not consistent with torch.round. Example: input 1.1016 with abs max
        # scale of 2.2821 will get mapped to 1.25 while mine will get mapped to 0.9570
        # The difference for mine is 0.1445 and for bnb 0.1484
        quantized_blocks = NF4Tensor.quantize_tensor_nearest(scaled_blocks.flatten(), nf4)

        # Combine the quantized elements into uint8 values
        # This lays out two consecutive elements in the same byte
        # [0, 1, 2, 3] -> [01, 23], the first element is the most significant
        # The size of combined blocks will be half the size of the original tensor
        combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

        return combined_blocks.to(torch.uint8)

    def get_original_weight(self) -> torch.Tensor:
        """Get the original weight from the normalized float weight format"""
        # since we are using uint8 we will decode 2 entries per byte
        # Shift elements down 4 and select out the bottom 4 bits
        first_elements = (self.quantized_data >> 4).to(torch.long)
        second_elements = (self.quantized_data & 0b1111).to(torch.long)

        # Dequantize every element
        dequantized_first = self.dequantize(first_elements, self.nf4)
        dequantized_second = self.dequantize(second_elements, self.nf4)

        # Build up matrix of scalers repeated for each element in the block
        # Since first and second elements make up a full block
        # we expand out to half the size of the full block
        scalers = self.dequantize_scalers(
            self.quantized_scalers, self.quantization_factor, self.scaler_block_size
        )
        repeated = scalers.unsqueeze(-1).expand(scalers.size(0), self.block_size // 2)

        scaled_first = dequantized_first * repeated.flatten()
        scaled_second = dequantized_second * repeated.flatten()

        # Flip them to be vertical and them stack them together horizontally
        # Upon flattening this will interleave the elements
        scaled_first = scaled_first.unsqueeze(-1).transpose(0, 1)
        scaled_second = scaled_second.unsqueeze(-1).transpose(0, 1)
        return torch.stack([scaled_first, scaled_second], dim=-1).reshape(self.shape)

    @staticmethod
    def quantize_tensor_nearest(value: torch.float16, nf4: torch.Tensor) -> torch.Tensor:
        """Quantize a float16 tensor to nf4 format to nearest and not rounded up"""
        value = value.unsqueeze(-1)  # (numel, 1)
        # Compare the value tensor with the nf4 tensor element-wise
        diff = (value - nf4).abs()
        # BnB appears to break ties by choosing the larger nf4 value
        closest_nf4 = diff.min(dim=-1).indices
        return closest_nf4

    @staticmethod
    def dequantize(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor:
        """Dequantize a nf4 value to float16 format"""
        # return nf4.index_select(0, value)
        return nf4[value]

    def unpack(
        self,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
        return (
            self.block_size,
            self.n_blocks,
            self.scaler_block_size,
            self.quantized_scalers,
            self.quantization_factor,
            self.scaler_mean,
            self.quantized_data,
        )

    def __repr__(self):
        return f"Quantized Data: {self.quantized_data}\nScalers: {self.quantized_scalers}\n"

    def __str__(self):
        return f"NF4Tensor({self.shape}, {self.block_size})"

    def __tensor_flatten__(self):
        tensor_meta = SubclassTensorArgs(
            self.shape,
            self.stride(),
            self.storage_offset(),
            self.dtype,
            self.device,
            self.requires_grad,
        )
        ctx = {
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "scaler_block_size": self.scaler_block_size,
            "tensor_meta": tensor_meta,
        }
        return [
            "quantized_data",
            "scaler_mean",
            "quantization_factor",
            "quantized_scalers",
            "nf4",
        ], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 5, "Expected 5 inner tensors"
        return NF4Tensor(
            metadata["tensor_meta"],
            metadata["block_size"],
            metadata["n_blocks"],
            metadata["scaler_block_size"],
            inner_tensors["quantized_scalers"],
            inner_tensors["quantization_factor"],
            inner_tensors["scaler_mean"],
            inner_tensors["quantized_data"],
            inner_tensors["nf4"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(torch._subclasses.functional_tensor.FunctionalTensor, type)
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in NF4_OPS_TABLE:
            return NF4_OPS_TABLE[func](func, args, kwargs)

        raise NotImplementedError(f"NF4Tensor does not support torch dispatch {func}")

    __torch_function__ = torch._C._disabled_torch_function_impl

    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     # Define a standard `__torch_function__` that propagates state
    #     kwargs = kwargs or {}

    #     def wrap(tensor_meta, block_size, n_blocks, scaler_block_size, quantized_scalers, quantization_factor, scaler_mean, quantized_data, nf4, o: Any):
    #         if isinstance(o, torch.Tensor) and not isinstance(o, cls):
    #             return cls(tensor_meta, block_size, n_blocks, scaler_block_size, quantized_scalers, quantization_factor, scaler_mean, quantized_data, nf4)
    #         return o

    #     with torch._C.DisableTorchFunctionSubclass():
    #         if isinstance(args[0], cls):
    #             out = func(*args, **kwargs)
    #             return tree_map(
    #                 functools.partial(wrap, args[0].tensor_meta, args[0].block_size, args[0].n_blocks, args[0].scaler_block_size, args[0].quantized_scalers, args[0].quantization_factor, args[0].scaler_mean, args[0].quantized_data, args[0].nf4), out
    #             )
    #         return func(*args, **kwargs)
