import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, cast
from collections.abc import Callable
from contextlib import contextmanager
from torch.profiler import profile, ProfilerActivity

from torch.nn.attention.flex_attention import BlockMask, flex_attention, _DEFAULT_SPARSE_BLOCK_SIZE
from torch.nn.attention import sdpa_kernel, SDPBackend, activate_flash_attention_impl

from transformer_nuggets.flex import FlexAttentionKernelArgs
from transformer_nuggets import init_logging

init_logging()
Tensor = torch.Tensor
AttentionBackend = Literal["sdpa", "flex"]
RopeCache = tuple[Tensor, Tensor]

__all__ = ["RopeAttention"]


@contextmanager
def cuda_kernel_profiler(kernel_pattern: str = "flash_attncute"):
    """
    Context manager that profiles CUDA kernels and checks for a pattern.

    Usage:
        with cuda_kernel_profiler("flash_attncute") as result:
            flex_attention(...)
        print(result["found"])  # True if flash kernel was called
        print(result["kernel_names"])  # List of all CUDA kernel names
    """
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()  # type: ignore
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


def get_flash_block_size(device: str = "cuda") -> tuple[int, int]:
    """
    Get block size for Flash backend based on GPU compute capability.

    On SM100+ (Blackwell): Q block must be 256, KV block is 128.
    On SM80/SM90: Both use default 128.
    """
    q_block = _DEFAULT_SPARSE_BLOCK_SIZE
    kv_block = _DEFAULT_SPARSE_BLOCK_SIZE

    dev = torch.device(device)
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        if major >= 10:
            q_block *= 2

    return (q_block, kv_block)


def calculate_tflops(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    time_ms: float,
    sparsity: float = 0.0,
) -> float:
    """Calculate TFLOPs for attention forward pass."""
    qk_flops = 2 * seq_q * seq_kv * head_dim
    sv_flops = 2 * seq_q * head_dim * seq_kv
    total_flops = batch * heads * (qk_flops + sv_flops) * (1 - sparsity)
    return total_flops / (time_ms * 1e9)  # ms to TFLOPs


class RopeAttention(nn.Module):
    """
    Minimal self-attention block with rotary embeddings.

    Pipeline:
        1) Up-project input to Q/K/V
        2) Apply RoPE to Q and K
        3) Call either torch sdpa or flex_attention
        4) Down-project back to the model dimension
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        backend: AttentionBackend = "sdpa",
        rope_base: float = 10000.0,
        causal: bool = True,
        flex_kernel_args: FlexAttentionKernelArgs | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even to apply RoPE.")

        self.backend = backend
        self.rope_base = rope_base
        self.causal = causal
        self.flex_kernel_args = flex_kernel_args

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self._rope_cache: dict[tuple[torch.device, torch.dtype], RopeCache] = {}

    @sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        block_mask: BlockMask | None = None,
        rope_cache: RopeCache | None = None,
        backend: AttentionBackend | None = None,
        score_mod: Callable | None = None,
        scale: float | None = None,
        flex_kernel_args: FlexAttentionKernelArgs | None = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, dim).
            attn_mask: Optional attention mask (sdpa path only).
            block_mask: Optional BlockMask for flex_attention.
            rope_cache: Precomputed (cos, sin) cache. If not provided it is built internally.
            backend: Override the init-time backend selection.
            score_mod: Score modifier callable for flex_attention.
            scale: Optional attention scale override.
            flex_kernel_args: Optional FlexAttentionKernelArgs override.
        """

        bsz, seqlen, _ = x.shape
        current_backend = backend or self.backend
        kernel_args = flex_kernel_args or self.flex_kernel_args

        qkv = self.qkv(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        rope_cache = self._get_rope_cache(seqlen, x.device, x.dtype, rope_cache)
        q = self._apply_rope(q, rope_cache)
        k = self._apply_rope(k, rope_cache)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        match current_backend:
            case "flex":
                if attn_mask is not None:
                    raise ValueError("attn_mask is only supported when backend='sdpa'.")

                kernel_options = kernel_args.asdict() if kernel_args is not None else None
                print(q.stride())
                attn_out = cast(
                    Tensor,
                    flex_attention(
                        q,
                        k,
                        v,
                        score_mod=score_mod,
                        block_mask=block_mask,
                        scale=scale,
                        kernel_options=kernel_options,
                    ),
                )
            case "sdpa":
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=self.causal,
                    scale=scale,
                )
            case other:
                raise ValueError(f"Unsupported attention backend: {other}")

        attn_out = attn_out.transpose(1, 2).view(bsz, seqlen, self.dim)
        return self.out_proj(attn_out)

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)

    def get_input(
        self,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = True,
    ) -> Tensor:
        return torch.randn(
            (batch_size, seq_len, self.dim),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def get_attn_params(self) -> tuple[int, int, int, int]:
        """Returns Hq, Hkv, Dq, Dkv"""
        return (self.num_heads, self.num_heads, self.head_dim, self.head_dim)

    def _get_rope_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        override: RopeCache | None,
    ) -> RopeCache:
        if override is not None:
            return override

        cache_key = (device, dtype)
        cache = self._rope_cache.get(cache_key)
        if cache is None or cache[0].size(0) < seq_len:
            self._rope_cache[cache_key] = self._build_rope_cache(seq_len, device, dtype)
        cos, sin = self._rope_cache[cache_key]
        return cos[:seq_len], sin[:seq_len]

    def _build_rope_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> RopeCache:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin

    def _apply_rope(self, x: Tensor, rope_cache: RopeCache) -> Tensor:
        cos, sin = rope_cache
        cos = cos.view(1, x.size(1), 1, -1)
        sin = sin.view(1, x.size(1), 1, -1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def main(compile: bool = False) -> None:
    from rich import print

    activate_flash_attention_impl("FA4")
    model = RopeAttention(dim=2048, num_heads=16).to("cuda", torch.bfloat16)
    if compile:
        # pyrefly: ignore [no-matching-overload]
        model = torch.compile(model)
    x = model.get_input(batch_size=4, seq_len=512)
    hq, hkv, dq, dkv = model.get_attn_params()
    print(f"Hq: {hq}, Hkv: {hkv}, Dq: {dq}, Dkv: {dkv}")
    print(f"Input shape: {x.shape}")

    from transformer_nuggets.utils.benchmark import profiler

    def forw_back():
        out = model(x)
        grads = torch.autograd.grad(out, (x,), torch.randn_like(out))
        return out, grads

    with cuda_kernel_profiler("flash_attncute") as result:
        out, grads = forw_back()
    for kernel in result["kernel_names"]:  # type: ignore
        if "flash" in kernel:
            print(f"Found flash kernel: {kernel}")

    with profiler("data/flash_attncute.json") as result:
        out, grads = forw_back()

    print("jobs done!")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
