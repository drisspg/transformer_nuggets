# A loop based (slow) implementation of the QLoRA weight
import torch
from scipy.stats import norm
from tqdm import tqdm


class NF4TensorDebug:
    """QLoRA Weight written in a more Debug friendly manner"""

    @staticmethod
    def get_nf4(cached=True) -> torch.Tensor:
        if cached:
            return torch.tensor(
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
                ]
            )

        offset = 0.9677083
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        # v2 = [0]*(256-15)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
        # v = v1 + v3 + 0.0
        nkf = torch.tensor(v1 + v3 + [0.0])
        nkf = nkf.sort().values
        nkf /= nkf.max()
        return nkf

    @staticmethod
    def quantize(value: torch.float16, nkf: torch.Tensor) -> torch.Tensor:
        """Quantize a float16 value to nkf format"""
        for i in range(len(nkf)):
            if value <= nkf[i]:
                # print("value", value, "nkf", nkf[i])
                return 0 | i
        return 0 | (len(nkf) - 1)

    @staticmethod
    def quantize_nearest(value: torch.float16, nkf: torch.Tensor) -> torch.Tensor:
        closest_index = 0
        closest_diff = abs(nkf[0] - value)
        for i in range(1, len(nkf)):
            diff = abs(nkf[i] - value)
            if diff < closest_diff:
                closest_diff = diff
                closest_index = i
        return 0 | closest_index

    @staticmethod
    def dequantize(value: torch.Tensor, nkf: torch.Tensor) -> torch.Tensor:
        """Dequantize a nkf value to float16 format"""
        # return nkf.index_select(0, value)
        return nkf[value]

    def get_scalers(self, inpt_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
        """Iterate through a flattened tensor getting the scalers for each block"""
        flattened_tensor = inpt_tensor.flatten()
        block_scalers = []
        for block_start in range(0, inpt_tensor.numel(), block_size):
            block_end = min(block_start + block_size, inpt_tensor.numel())
            block = flattened_tensor[block_start:block_end]
            block_max = block.abs().max()
            block_scalers.append(block_max)
        return torch.tensor(block_scalers)

    def __init__(self, inpt_tensor: torch.Tensor, block_size=64):
        assert inpt_tensor.numel() % block_size == 0, (
            "Input tensor must be divisible by block size"
        )
        self.block_size = block_size
        self.n_blocks = inpt_tensor.numel() // block_size
        self.scalers = self.get_scalers(inpt_tensor, self.block_size)
        self.norm_float_weight = self.get_norm_float_weight(inpt_tensor.clone())
        self.original_shape = inpt_tensor.shape
        self.dtype = inpt_tensor.dtype

    def get_norm_float_weight(self, inpt_tensor: torch.Tensor) -> torch.Tensor:
        nkf = self.get_nf4()
        flattened_tensor = inpt_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = inpt_tensor.numel()
        assert numel % 2 == 0, (
            "Number of elements must be even just to not have to think about the end"
        )
        quantized_length = numel // 2
        quantized_tensor = torch.zeros(quantized_length, dtype=torch.uint8)
        for i in tqdm(range(len(self.scalers))):
            block_start = i * self.block_size
            block_end = min(block_start + self.block_size, flattened_tensor.numel())
            block = flattened_tensor[block_start:block_end]
            # Scale the block
            block /= self.scalers[i]
            # We will iterate over each element in the block and quantize it
            # In groups of 2
            for j in range(0, self.block_size, 2):
                # Combine two bfloat16s via quantization to 4 bit types into a single uint8
                element_1 = self.quantize_nearest(block[j], nkf)
                element_2 = self.quantize_nearest(block[j + 1], nkf)
                combined = element_1 << 4 | element_2
                quantized_tensor[(i * self.block_size // 2) + j // 2] = combined
        return quantized_tensor

    def get_original_weight(self):
        # since we are using uint8 we will decode 2 entries per byte
        nkf = self.get_nf4()
        original_weight = torch.empty(2 * (self.norm_float_weight.numel()), dtype=self.dtype)
        # Scalers is a proxy for num_blocks
        for i in range(len(self.scalers)):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block = original_weight[block_start:block_end]
            for j in range(0, self.block_size, 2):
                combined = self.norm_float_weight[(i * self.block_size // 2) + j // 2]
                # Shift element down 4
                element_1 = combined >> 4
                # Select out the bottom 4 bits
                element_2 = combined & 0b1111
                block[j] = self.dequantize(element_1.item(), nkf) * self.scalers[i]
                block[j + 1] = self.dequantize(element_2.item(), nkf) * self.scalers[i]
        return original_weight.reshape(self.original_shape)
