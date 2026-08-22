"""Generate an eager FX fusion-opportunity trace for Perfetto."""

from pathlib import Path

import torch

from transformer_nuggets.fx_analysis import profile_fx_fusion


def pointwise_gate(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply a deliberately unfused pointwise chain."""
    product = x * scale
    gate = torch.sigmoid(product)
    return product * gate


def main() -> None:
    """Profile the example and print the detected fusion regions."""
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")

    x = torch.randn(8192, 4096, device="cuda")
    scale = torch.randn((), device="cuda")
    result = profile_fx_fusion(
        pointwise_gate,
        (x, scale),
        Path("artifacts/fx_fusion_pointwise.pftrace"),
    )

    for region in result.analysis.regions:
        print(
            f"{region.region_id}: {len(region.op_names)} ops, "
            f"{region.minimum_avoidable_bytes / 2**20:.1f} MiB avoidable, "
            f"{region.ideal_bytes / 2**20:.1f} MiB ideal I/O"
        )
    print(f"Wrote {result.trace_path.resolve()}")


if __name__ == "__main__":
    main()
