# Llama Pretraining

This directory contains the code for pretraining Llama. The model definition is from [gpt-fast](https://github.com/pytorch-labs/gpt-fast). It is slightly modified to remove the kvcache since this is not needed during pre-training.

The Tokenizer is from the original [LLama repo](https://github.com/facebookresearch/llama) and uses sentencepiece under the hood. Instead of training the tokenizer from scratch the tokenizer.bin file from llama2 release is used.

The training loop can be found in `train.py`. It expects that the `prepare_data.py` script has been run to generate the training data. The training data is expected to be in the `data/` directory.

### Usage
Get the Llama2 tokenizer, file and place inside the `llama/data` directory.

The following paths are assumed you are in the top level `transformer_nuggets/` directory.

Then run the following command:
``` Shell
python transformer_nuggets/llama/prepare_data.py \
    --tokenizer_path=transformer_nuggets/llama/data/tokenizer.model \
    --output_dir=transformer_nuggets/llama/data/
```






 ### Notes
To get the Llama2 tokenizer go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access. This will get you pretrained weights as well as the tokenizer.
