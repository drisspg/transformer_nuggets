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

## Project Structure

- **benchmarks**: Contains scripts and data related to benchmarking the transformer components.
  - **data**: Benchmark data files.
  - `flash.py`: Benchmarking script for Flash.
  - `llama.py`: Benchmarking script for Llama.
  - `qlora.py`: Benchmarking script for Qlora.

- **transformer_nuggets**: The main directory containing all transformer components/modules.
  - **flash**: Components related to the FlashAttention.
  - **quant**: Implementation of NF4 Tensor and QLora in pure Pytorch
  - **sdpa**: Prototype for updated SDPA interface in Pytorch.
  - **utils**: General utility functions and scripts.
    - `benchmark.py`: Benchmark-related utility functions.
    - `tracing.py`: Tracing utilities for transformers.

- **test**: Contains test scripts for various transformer components.
  - `test_flash.py`: Tests for Flash.
  - `test_qlora.py`: Tests for Qlora.
  - `test_sdpa.py`: Tests for SDPA.
