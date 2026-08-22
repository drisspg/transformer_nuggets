"""Compare fused and handwritten eager RMSNorm paths in one Perfetto trace."""

from pathlib import Path

import torch

from transformer_nuggets.fx_analysis import RooflineSpec, profile_fx_fusion
from transformer_nuggets.utils.tracing import cuda_kernel_profiler


EPS = 1e-6
HIDDEN_SIZE = 4096


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run PyTorch's fused eager RMSNorm operator."""
    output, _ = torch.ops.aten._fused_rms_norm.default(
        x,
        [x.shape[-1]],
        weight,
        EPS,
    )
    return output


def handwritten_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Implement RMSNorm from ordinary eager PyTorch operations."""
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(variance + EPS)
    return (normalized * weight.float()).to(x.dtype)


def compare_rms_norms(
    fused_input: torch.Tensor,
    handwritten_input: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute fused and handwritten RMSNorm as independent graph branches."""
    return (
        fused_rms_norm(fused_input, weight),
        handwritten_rms_norm(handwritten_input, weight),
    )


def kernel_names(function, *args) -> list[str]:
    """Return the CUDA kernels launched by one eager function call."""
    with cuda_kernel_profiler(record_name=function.__name__) as result:
        function(*args)
        torch.cuda.synchronize()
    return result["kernel_names"]


def main() -> None:
    """Check correctness and emit FX plus Perfetto artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")

    x = torch.randn(2, 2048, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    fused = fused_rms_norm(x, weight)
    handwritten = handwritten_rms_norm(x, weight)
    torch.testing.assert_close(fused, handwritten, atol=2e-2, rtol=2e-2)
    fused_kernels = kernel_names(fused_rms_norm, x, weight)
    handwritten_kernels = kernel_names(handwritten_rms_norm, x, weight)

    output_dir = Path("artifacts/rms_norm_fusion")
    result = profile_fx_fusion(
        compare_rms_norms,
        (x, x.clone(), weight),
        output_dir / "rms_norm_comparison.pftrace",
        roofline_spec=RooflineSpec(
            name="example ceilings: 100 TFLOP/s, 6.4 TB/s",
            peak_compute_tflops=100.0,
            peak_memory_gbps=6400.0,
            launch_latency_us=3.0,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rms_norm_fx_graph.txt").write_text(str(result.analysis.graph_module.graph))
    (output_dir / "rms_norm_fx_graph.md").write_text(
        f"""# Eager RMSNorm comparison

- Fused path: {len(fused_kernels)} CUDA kernel
- Handwritten path: {len(handwritten_kernels)} CUDA kernels

```mermaid
flowchart LR
    XF[\"BF16 fused input\"] --> F[\"aten._fused_rms_norm\\none opaque fused operation\"]
    W[\"BF16 weight\"] --> F
    F --> FO[\"fused output\"]

    XH[\"BF16 handwritten input\"] --> C[\"cast to FP32\"]
    C --> SQ[\"square\"]
    SQ --> MEAN[\"mean over hidden dimension\"]
    MEAN --> ADD[\"add epsilon\"]
    ADD --> RSQRT[\"rsqrt\"]
    C --> SCALE[\"multiply normalization scale\"]
    RSQRT --> SCALE
    W --> WC[\"cast weight to FP32\"]
    WC --> AFFINE[\"multiply weight\"]
    SCALE --> AFFINE
    AFFINE --> CAST[\"cast to BF16\"]
    CAST --> HO[\"handwritten output\"]
```
"""
    )

    print(f"Fused RMSNorm kernels: {len(fused_kernels)}")
    print(f"Handwritten RMSNorm kernels: {len(handwritten_kernels)}")
    for region in result.analysis.regions:
        print(
            f"{region.region_id}: {len(region.op_names)} ops, "
            f"{region.minimum_avoidable_bytes / 2**20:.1f} MiB avoidable, "
            f"{region.ideal_bytes / 2**20:.1f} MiB ideal I/O"
        )
    print(f"Wrote {result.trace_path.resolve()}")
    print(f"Wrote {(output_dir / 'rms_norm_fx_graph.md').resolve()}")


if __name__ == "__main__":
    main()
