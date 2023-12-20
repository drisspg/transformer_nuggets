# transformer_nuggets
A place to store reusable transformer components of my own creation or found on the interwebs

![transformer_nuggies](https://github.com/drisspg/transformer_nuggets/assets/32754868/8329986a-aa9f-41a6-a332-49a0d71438aa)

## Getting Started
Clone the repository:
```Shell
git clone https://github.com/drisspg/transformer_nuggets.git
```

### Install Package
```Shell
pip install -e .
```

#### Dev Tool Chain
``` Shell
pip install -e ".[dev]"
```
pre-commit is used to make sure that I don't forget to format stuff, I am going to see if I like this or not. This
should be installed when installing the dev tools.

## Project Structure

- **benchmarks**: Contains scripts and data related to benchmarking the transformer components.
  - **data**: Benchmark data files.
  - `flash.py`: Benchmarking script for Flash.
  - `llama.py`: Benchmarking script for Llama.
  - `qlora.py`: Benchmarking script for Qlora.
  - `fp8_sat_cast.py`: Benchmarks for comparing FP8 saturated casting kernel to eager and compile code.

- **transformer_nuggets**: The main directory containing all transformer components/modules.
  - **flash**: Components related to the FlashAttention.
  - **quant**: Implementation of NF4 Tensor and QLora in pure Pytorch
  - **sdpa**: Prototype for updated SDPA interface in Pytorch.
  - **fp8**: Components related interacting with PyTorch FP8 tensors.
  - **llama**: Contains a model def for llama2 models as well as a pretraining script.
  - **utils**: General utility functions and scripts.
    - `benchmark.py`: Benchmark-related utility functions.
    - `tracing.py`: Tracing utilities for transformers.

- **test**: Contains test scripts for various transformer components.
  - `test_flash.py`: Tests for Flash.
  - `test_qlora.py`: Tests for Qlora.
  - `test_sdpa.py`: Tests for SDPA.
  - `test_fp8.py`: Tests for FP8.
