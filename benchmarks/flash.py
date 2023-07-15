import torch
import pytest
import triton
from transformer_nuggets.flash import BiasMode, build_alibi_mask, attention

device = torch.device("cuda")

BATCH, N_HEADS, D_HEAD = 4, 16, 64
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(10, 14)],
    line_arg='provider',
    line_vals=['triton'] + (['pytorch'] ),
    line_names=['Triton'] + (['Pytorch']),
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal-{causal}-bias-{bias_choice}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, 'mode': mode, 'causal': causal, 'bias_choice': bias_choice}
) for mode in ['fwd', 'bwd'] for causal in [False, True] for bias_choice in [BiasMode.none, BiasMode.rel_pos, BiasMode.alibi]]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, causal, bias_choice, provider, dtype=torch.float16, device=device):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    sm_scale = 1
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: attention(q, k, v, causal, sm_scale, bias_choice)
        if mode == 'bwd':
            o, mask = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == 'pytorch':
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        # reference implementation
        if bias_choice == BiasMode.rel_pos:
            attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=1, causal=causal)
            attn_bias = attn_bias.expand(BATCH, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
        elif bias_choice == BiasMode.alibi:
            attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=None, causal=causal)
            attn_bias = attn_bias.expand(BATCH, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
        elif bias_choice == BiasMode.none:
            attn_bias = None
        is_causal = causal if (bias_choice == BiasMode.none) else False
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_bias, scale=sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == '__main__':
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=None, print_data=True)