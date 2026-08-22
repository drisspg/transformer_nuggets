## transformer_nuggets

A grab-bag of experimental transformer kernels and utilities (mostly PyTorch + Triton).

![transformer_nuggies](https://github.com/drisspg/transformer_nuggets/assets/32754868/8329986a-aa9f-41a6-a332-49a0d71438aa)

### What’s in here

- **FlashAttention experiments**: removed; the useful pieces have been upstreamed to PyTorch as FlexAttention in a commit.
- **NF4 / QLoRA quantization experiments**: removed; that work now lives in torchao.
- **`transformer_nuggets/fp8`**: FP8 casting / scaled-quantization kernels (Triton).
- **`transformer_nuggets/cute`**: CUTE DSL experiments and tooling (includes an intra-kernel profiler).
- **`transformer_nuggets/misc`**: Odds and ends (e.g. attention wrappers, utilities).
- **`transformer_nuggets/llama`**: LLaMA-ish model + training/finetune scripts (research-grade).
- **Logical roofline analysis**: eager FX/AOT training attribution, raw Kineto trace
  decoration, per-kernel Perfetto tracks, ranked fusion findings, and replay/NCU follow-ups.

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
pip install "transformer_nuggets[llama]"       # llama training utilities
pip install "transformer_nuggets[huggingface]" # Hugging Face profiling example
```

### Quick examples

Use torchao for quantization experiments.

Use PyTorch FlexAttention instead of the old local FlashAttention experiments.

Annotate an existing PyTorch profiler trace with logical roofline metadata:

```shell
annotate-roofline trace.json.gz -o trace.roofline.pftrace \
  --formula-module my_project.roofline_formulas \
  --peak-compute-tflops 1000 --peak-memory-gbps 4000
```

Profile an eager callable or forward/backward training step from Python with
`transformer_nuggets.fx_analysis.profile_fx_fusion` or `profile_aot_training`.
Logical formulas and physical NCU counters are intentionally kept separate. See
`.agents/skills/analyzing-pytorch-rooflines/SKILL.md` for the workflow contract.

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
