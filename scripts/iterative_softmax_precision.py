import torch
import pandas as pd
import math

from transformer_nuggets.utils.model_extraction import extract_attention_data


M_LOG2E = math.log2(math.e)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


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

    #  reuse L inited to 0 for first delta
    deltas = [l.to(torch.float64)]

    for chunk_index, start in enumerate(range(0, Nk, chunk_size)):
        end = min(start + chunk_size, Nk)

        scores_chunk = (q_ @ k_[:, :, start:end, :].transpose(-2, -1)) * qk_scale
        scores_chunk = scores_chunk * log2e if use_exp2 else scores_chunk

        m_chunk = scores_chunk.max(dim=-1, keepdim=True).values  # (B, H, Nq, 1)
        m_new = torch.maximum(m, m_chunk)

        if chunk_index != 0:
            # skip first chunk w/ is -inf
            delta = m.to(torch.float64) - m_new.to(torch.float64)
            deltas.append(delta)

        alpha = exp_fn(m - m_new)  # (B, H, Nq, 1)
        acc = acc * alpha

        p = exp_fn(scores_chunk - m_new)  # (B, H, Nq, chunk)
        l = l * alpha  # noqa: E741  # (B, H, Nq, 1)
        l = l + p.sum(dim=-1, keepdim=True)  # noqa: E741  # (B, H, Nq, 1)

        # Mimic Tensor core usage + accum in hp
        acc = acc + p.to(q.dtype).to(accum_dtype) @ v_[:, :, start:end, :]  # (B, H, Nq, D)

        m = m_new

    out = acc / l  # (B, H, Nq, D)
    deltas_tensor = torch.cat(deltas, dim=0)
    return out.to(q.dtype), deltas_tensor


def compute_errors(baseline: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    diff = baseline - test
    abs_baseline = baseline.abs()

    mse = diff.square().mean().item()
    mae = diff.abs().mean().item()
    max_abs_error = diff.abs().max().item()
    relative_error = (diff.abs() / (abs_baseline + 1e-10)).mean().item()

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
        precision_name = dtype_name(precision)
        q_test, k_test, v_test = (tensor.to(precision) for tensor in (q_fp64, k_fp64, v_fp64))

        for chunk_size in chunk_sizes:
            output, deltas = iterative_attention(
                q_test, k_test, v_test, chunk_size, use_exp2=use_exp2
            )

            errors = compute_errors(baseline, output.to(torch.float64))
            deltas_fp64 = deltas.to(torch.float64)
            delta_mean = deltas_fp64.mean().item()
            delta_std = deltas_fp64.std(unbiased=False).item()

            alpha_fn = torch.exp2 if use_exp2 else torch.exp
            alphas = alpha_fn(deltas_fp64)
            alpha_mean = alphas.mean().item()
            alpha_std = alphas.std(unbiased=False).item()

            results.append(
                {
                    "precision": precision_name,
                    "chunk_size": chunk_size,
                    "exp_fn": "exp2" if use_exp2 else "exp",
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
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
    qwen_model: str | None = None,
    qwen_layer: int | None = None,
    qwen_prompt: str | None = None,
    qwen_sample_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, seq_len, head_dim)
    dtype = torch.float64

    match factory:
        case "rand":
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)
        case "make_tensor":
            q = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            k = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            v = torch.testing.make_tensor(shape, device=device, dtype=dtype)
        case "sorted_scores":
            q = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            v = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            k = torch.arange(seq_len, device=device, dtype=dtype)
            k = k.view(1, 1, seq_len, 1).expand(shape)
            return q, k, v
        case "descending_scores":
            q = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            v = torch.testing.make_tensor(shape, device=device, dtype=dtype)
            k = torch.arange(seq_len, 0, -1, device=device, dtype=dtype)
            k = k.view(1, 1, seq_len, 1).expand(shape)
            return q, k, v
        case "qwen":
            if qwen_model is None:
                raise ValueError("For 'qwen' factory, must provide --qwen-model")

            q, k, v = extract_attention_data(
                model_id=qwen_model,
                dataset_name="fka/awesome-chatgpt-prompts",
                num_samples=batch_size,
                max_length=seq_len,
                min_prompt_length=100,
                layers=[0],
            )

            q = q[qwen_sample_idx].unsqueeze(0).to(device).to(dtype)
            k = k[qwen_sample_idx].unsqueeze(0).to(device).to(dtype)
            v = v[qwen_sample_idx].unsqueeze(0).to(device).to(dtype)

            num_q_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            if num_q_heads != num_kv_heads:
                n_rep = num_q_heads // num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

            return q, k, v
        case _:
            raise ValueError(
                f"Unknown factory: {factory}. Must be 'rand', 'make_tensor', "
                "'sorted_scores', 'descending_scores', or 'qwen'"
            )

    scale = head_dim**-0.5
    q = q * scale

    return q, k, v


def wrapped_print(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_precision_sections(results: pd.DataFrame, columns: list[str], title: str):
    wrapped_print(title)
    for precision, data in results.groupby("precision"):
        print(f"\n{precision}:")
        print(data[columns].to_string(index=False))


def main(
    batch_size: int = 32,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 64,
    device: str = "cuda",
    chunk_sizes: list[int] = [16, 32, 64, 128, 256, 512, 1024],
    factory: str = "sorted_scores",
    qwen_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    qwen_layer: int = 0,
    qwen_prompt: str | None = None,
    use_exp2: bool = False,
):
    """Test iterative softmax precision with different chunk sizes.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device to run on
        chunk_sizes: Chunk sizes to test
        factory: Factory method for creating Q, K, V tensors (rand, make_tensor, sorted_scores, descending_scores, or qwen)
        qwen_model: Qwen model name (only used with factory=qwen)
        qwen_layer: Qwen layer to extract (only used with factory=qwen)
        qwen_prompt: Prompt for Qwen inference (only used with factory=qwen)
        use_exp2: Use exp2 (base-2) instead of exp (natural base) for exponentials
    """
    wrapped_print("SETUP")
    print(f"Factory: {factory}")
    if factory == "qwen":
        print(f"Qwen Model: {qwen_model}")
        print(f"Qwen Layer: {qwen_layer}")
        if qwen_prompt is None:
            prompt_display = "<long prompt>"
        else:
            prompt_display = qwen_prompt if len(qwen_prompt) <= 60 else qwen_prompt[:57] + "..."
        print(f"Qwen Prompt: {prompt_display}")
    print(f"Shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"Exponential: {'exp2 (base-2)' if use_exp2 else 'exp (natural)'}")

    q, k, v = qkv_factory(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        factory,
        device,
        qwen_model=qwen_model,
        qwen_layer=qwen_layer,
        qwen_prompt=qwen_prompt,
    )

    chunk_sizes = sorted({cs for cs in chunk_sizes if cs <= seq_len} | {seq_len})

    precisions = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

    print(f"\nChunk sizes: {chunk_sizes}")
    print(f"Precisions: {[dtype_name(p) for p in precisions]}")

    wrapped_print("RUNNING EXPERIMENTS")

    results_df = run_experiment(q, k, v, chunk_sizes, precisions, use_exp2=use_exp2)

    print_precision_sections(
        results_df,
        ["chunk_size", "mse", "mae", "max_abs_error", "relative_error"],
        "ANALYSIS BY PRECISION",
    )

    print_precision_sections(
        results_df,
        ["chunk_size", "delta_mean", "delta_std", "alpha_mean", "alpha_std"],
        "DELTA / ALPHA STATISTICS BY PRECISION",
    )

    wrapped_print("KEY INSIGHTS")

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

    print("\n" + "=" * 80)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
