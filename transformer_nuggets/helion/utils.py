import torch

from transformer_nuggets.utils import benchmark_cuda_function_in_microseconds
from rich import print


def test_kernel(kernel_fn, spec_fn, *args):
    """Test a Helion kernel against a reference implementation."""
    # Run our implementation
    result = kernel_fn(*args)
    # Run reference implementation
    expected = spec_fn(*args)

    # Check if results match
    torch.testing.assert_close(result, expected)
    print("✅ Results Match ✅")


def benchmark_kernel(kernel_fn, *args, **kwargs):
    """Benchmark a Helion kernel."""
    no_args = lambda: kernel_fn(*args, **kwargs)
    time_in_ms = benchmark_cuda_function_in_microseconds(no_args)
    print(f"⏱ Time: {time_in_ms} ms")


def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = benchmark_cuda_function_in_microseconds(kernel_no_args)
    spec_time = benchmark_cuda_function_in_microseconds(spec_no_args)
    print(
        f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time / kernel_time:.3f}x"
    )
