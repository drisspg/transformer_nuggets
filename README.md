## transformer_nuggets

A grab-bag of experimental transformer kernels and utilities (mostly PyTorch + Triton).

![transformer_nuggies](https://github.com/drisspg/transformer_nuggets/assets/32754868/8329986a-aa9f-41a6-a332-49a0d71438aa)

### What’s in here

- **`transformer_nuggets/flash`**: Triton FlashAttention experiments + masking/bias utilities.
- **`transformer_nuggets/quant`**: NF4 tensor subclass + QLoRA building blocks (pure PyTorch).
- **`transformer_nuggets/fp8`**: FP8 casting / scaled-quantization kernels (Triton).
- **`transformer_nuggets/cute`**: CUTE DSL experiments and tooling (includes an intra-kernel profiler).
- **`transformer_nuggets/misc`**: Odds and ends (e.g. attention wrappers, utilities).
- **`transformer_nuggets/llama`**: LLaMA-ish model + training/finetune scripts (research-grade).

This repository is research code: APIs are not stable and may change.

### Install

You’ll need a working PyTorch install first (CPU or CUDA). Follow the official
[PyTorch install instructions](https://pytorch.org/get-started/locally/).

To install from PyPI:

```shell
pip install transformer_nuggets
```

To hack on the code locally:

```shell
git clone https://github.com/drisspg/transformer_nuggets.git
cd transformer_nuggets
pip install -e .
```

Optional extras:

```shell
pip install "transformer_nuggets[flash]"  # triton
pip install "transformer_nuggets[qlora]"  # bitsandbytes (optional comparisons)
pip install "transformer_nuggets[llama]"  # llama training utilities
```

### Quick examples

NF4 tensor round-trip:

```python
import torch

from transformer_nuggets.quant.nf4_tensor import NF4Tensor

w = torch.randn(4096, 4096, dtype=torch.float16).contiguous()
w.requires_grad_(False)

nf4_w = NF4Tensor.from_tensor(w)
w_dequant = nf4_w.get_original_weight()
```

FlashAttention (requires CUDA + Triton; API is experimental):

```python
import math
import torch

from transformer_nuggets.flash import BiasMode, attention

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)

o, _ = attention(q, k, v, causal=True, sm_scale=1 / math.sqrt(q.size(-1)), bias_choice=BiasMode.none)
```

CUTE intra-kernel profiling (writes a Perfetto trace):

```shell
python -m transformer_nuggets.cute.profiler.example
```

### Repo layout

- **`transformer_nuggets/`**: Python package.
- **`benchmarks/`**: Microbenchmarks and profiling scripts.
- **`examples/`**: Small runnable examples.
- **`scripts/`**: One-off utilities.
- **`test/`**: PyTest suite.

### Development

```shell
pip install -e ".[dev]"
pre-commit install
pytest
```
