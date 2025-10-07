import torch
import numpy as np
import pandas as pd
import argparse
import math


M_LOG2E = math.log2(math.e)


def non_iterative_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_exp2: bool = False
) -> torch.Tensor:
    qk_scale = q.shape[-1] ** -0.5
    q = q * qk_scale**0.5
    k = k * qk_scale**0.5
    scores = q @ k.transpose(-2, -1)

    if use_exp2:
        scores = scores * M_LOG2E
        max_score = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp2(scores - max_score)
    else:
        max_score = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_score)

    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    attn_weights = exp_scores / sum_exp
    return attn_weights @ v


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def iterative_attention(q, k, v, chunk_size, use_exp2=False):
    d = q.shape[-1]
    qk_scale = d**-0.5
    accum_dtype = torch.float32 if q.dtype in [torch.float16, torch.bfloat16] else q.dtype

    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    device = q.device

    q_ = q.to(accum_dtype)
    k_ = k.to(accum_dtype)
    v_ = v.to(accum_dtype)

    m = torch.full((B, H, Nq, 1), float("-inf"), dtype=accum_dtype, device=device)
    l = torch.zeros((B, H, Nq, 1), dtype=accum_dtype, device=device)  # noqa: E741
    acc = torch.zeros((B, H, Nq, D), dtype=accum_dtype, device=device)

    exp_fn = torch.exp2 if use_exp2 else torch.exp
    log2e = M_LOG2E if use_exp2 else 1.0

    num_chunks = ceil_div(Nk, chunk_size)
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, Nk)

        scores_chunk = (q_ @ k_[:, :, s:e, :].transpose(-2, -1)) * qk_scale
        scores_chunk = scores_chunk * log2e if use_exp2 else scores_chunk

        m_chunk = scores_chunk.max(dim=-1, keepdim=True).values  # (B, H, Nq, 1)
        m_new = torch.maximum(m, m_chunk)

        alpha = exp_fn(m - m_new)  # (B, H, Nq, 1)
        acc = acc * alpha

        p = exp_fn(scores_chunk - m_new)  # (B, H, Nq, chunk)
        l = l * alpha  # noqa: E741  # (B, H, Nq, 1)
        l = l + p.sum(dim=-1, keepdim=True)  # noqa: E741  # (B, H, Nq, 1)

        acc = acc + p @ v_[:, :, s:e, :]  # (B, H, Nq, D)

        m = m_new

    out = acc / l  # (B, H, Nq, D)
    return out.to(q.dtype)


def compute_errors(baseline: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    baseline_np = baseline.cpu().numpy()
    test_np = test.cpu().numpy()

    mse = np.mean((baseline_np - test_np) ** 2)
    mae = np.mean(np.abs(baseline_np - test_np))
    max_abs_error = np.max(np.abs(baseline_np - test_np))

    relative_error = np.mean(np.abs((baseline_np - test_np) / (baseline_np + 1e-10)))

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
    }


def run_experiment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_sizes: list[int],
    precisions: list[torch.dtype],
    use_exp2: bool = False,
) -> pd.DataFrame:
    q_fp64 = q.to(torch.float64)
    k_fp64 = k.to(torch.float64)
    v_fp64 = v.to(torch.float64)

    baseline = non_iterative_attention(q_fp64, k_fp64, v_fp64, use_exp2=use_exp2)

    results = []

    for precision in precisions:
        precision_name = str(precision).split(".")[-1]

        for chunk_size in chunk_sizes:
            q_test = q_fp64.to(precision)
            k_test = k_fp64.to(precision)
            v_test = v_fp64.to(precision)

            output = iterative_attention(q_test, k_test, v_test, chunk_size, use_exp2=use_exp2)

            output_fp64 = output.to(torch.float64)

            errors = compute_errors(baseline, output_fp64)

            results.append(
                {
                    "precision": precision_name,
                    "chunk_size": chunk_size,
                    "exp_fn": "exp2" if use_exp2 else "exp",
                    **errors,
                }
            )

    return pd.DataFrame(results)


def qkv_factory(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    factory: str = "rand",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, seq_len, head_dim)

    if factory == "rand":
        q = torch.randn(shape, device=device, dtype=torch.float64)
        k = torch.randn(shape, device=device, dtype=torch.float64)
        v = torch.randn(shape, device=device, dtype=torch.float64)
    elif factory == "make_tensor":
        q = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)
        k = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)
        v = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown factory: {factory}. Must be 'rand' or 'make_tensor'")

    scale = head_dim**-0.5
    q = q * scale

    return q, k, v


def wrapped_print(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Test iterative softmax precision with different chunk sizes"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="Chunk sizes to test",
    )
    parser.add_argument(
        "--factory",
        type=str,
        default="make_tensor",
        choices=["rand", "make_tensor"],
        help="Factory method for creating Q, K, V tensors",
    )
    parser.add_argument(
        "--use-exp2",
        action="store_true",
        help="Use exp2 (base-2) instead of exp (natural base) for exponentials",
    )

    args = parser.parse_args()

    WIDTH = 80

    wrapped_print("SETUP", WIDTH)
    print(f"Factory: {args.factory}")
    print(f"Shape: [{args.batch_size}, {args.num_heads}, {args.seq_len}, {args.head_dim}]")
    print(f"Exponential: {'exp2 (base-2)' if args.use_exp2 else 'exp (natural)'}")

    q, k, v = qkv_factory(
        args.batch_size, args.num_heads, args.seq_len, args.head_dim, args.factory, args.device
    )

    chunk_sizes = [cs for cs in args.chunk_sizes if cs <= args.seq_len]
    chunk_sizes.append(args.seq_len)
    chunk_sizes = sorted(set(chunk_sizes))

    precisions = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

    print(f"\nChunk sizes: {chunk_sizes}")
    print(f"Precisions: {[str(p).split('.')[-1] for p in precisions]}")

    wrapped_print("RUNNING EXPERIMENTS", WIDTH)

    results_df = run_experiment(q, k, v, chunk_sizes, precisions, use_exp2=args.use_exp2)

    wrapped_print("RESULTS", WIDTH)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2e}".format)

    print("\n" + results_df.to_string(index=False))

    wrapped_print("ANALYSIS BY PRECISION", WIDTH)

    for precision in results_df["precision"].unique():
        precision_data = results_df[results_df["precision"] == precision]
        print(f"\n{precision}:")
        print(
            precision_data[
                ["chunk_size", "mse", "mae", "max_abs_error", "relative_error"]
            ].to_string(index=False)
        )

    wrapped_print("KEY INSIGHTS", WIDTH)

    for precision in results_df["precision"].unique():
        precision_data = results_df[results_df["precision"] == precision]

        min_error_row = precision_data.loc[precision_data["mse"].idxmin()]
        max_error_row = precision_data.loc[precision_data["mse"].idxmax()]

        print(f"\n{precision}:")
        print(
            f"  Best chunk size: {int(min_error_row['chunk_size'])} (MSE: {min_error_row['mse']:.2e})"
        )
        print(
            f"  Worst chunk size: {int(max_error_row['chunk_size'])} (MSE: {max_error_row['mse']:.2e})"
        )

    print("\n" + "=" * WIDTH)


if __name__ == "__main__":
    main()
