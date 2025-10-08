import torch
import numpy as np
import pandas as pd
import math

from transformer_nuggets.utils import AttentionExtractor


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

        # Mimic Tensor core usage + accum in hp
        acc = acc + p.to(q.dtype).to(accum_dtype) @ v_[:, :, s:e, :]  # (B, H, Nq, D)

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
    qwen_model: str | None = None,
    qwen_layer: int | None = None,
    qwen_prompt: str | None = None,
    qwen_sample_idx: int = 0,
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
    elif factory == "sorted_scores":
        q = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)
        v = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)

        k = torch.arange(seq_len, device=device, dtype=torch.float64)
        k = k.view(1, 1, seq_len, 1).expand(batch_size, num_heads, seq_len, head_dim)

        return q, k, v
    elif factory == "descending_scores":
        q = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)
        v = torch.testing.make_tensor(shape, device=device, dtype=torch.float64)

        k = torch.arange(seq_len, 0, -1, device=device, dtype=torch.float64)
        k = k.view(1, 1, seq_len, 1).expand(batch_size, num_heads, seq_len, head_dim)

        return q, k, v
    elif factory == "qwen":
        if qwen_model is None or qwen_layer is None:
            raise ValueError("For 'qwen' factory, must provide --qwen-model and --qwen-layer")

        extractor = AttentionExtractor(qwen_model, device=device)
        extractor.load_model()
        extractor.register_hooks([qwen_layer])
        extractor.run_inference(prompts=[qwen_prompt] if qwen_prompt else None, seq_len=seq_len)

        layer_name = f"layer_{qwen_layer}"
        q = extractor.attention_data[layer_name]["q"][qwen_sample_idx].to(torch.float64)
        k = extractor.attention_data[layer_name]["k"][qwen_sample_idx].to(torch.float64)
        v = extractor.attention_data[layer_name]["v"][qwen_sample_idx].to(torch.float64)

        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            n_rep = num_q_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        extractor.cleanup()

        return q, k, v
    else:
        raise ValueError(
            f"Unknown factory: {factory}. Must be 'rand', 'make_tensor', 'sorted_scores', 'descending_scores', or 'qwen'"
        )

    scale = head_dim**-0.5
    q = q * scale

    return q, k, v


def wrapped_print(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def main(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 8192,
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
    WIDTH = 80

    wrapped_print("SETUP", WIDTH)
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

    chunk_sizes = [cs for cs in chunk_sizes if cs <= seq_len]
    chunk_sizes.append(seq_len)
    chunk_sizes = sorted(set(chunk_sizes))

    precisions = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

    print(f"\nChunk sizes: {chunk_sizes}")
    print(f"Precisions: {[str(p).split('.')[-1] for p in precisions]}")

    wrapped_print("RUNNING EXPERIMENTS", WIDTH)

    results_df = run_experiment(q, k, v, chunk_sizes, precisions, use_exp2=use_exp2)

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
    from jsonargparse import CLI

    CLI(main)
