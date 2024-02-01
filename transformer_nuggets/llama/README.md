# Llama Pretraining

This directory contains the code for pretraining Llama. The model definition is from [gpt-fast](https://github.com/pytorch-labs/gpt-fast). It is slightly modified to remove the kvcache since this is not needed during pre-training.

The Tokenizer is from the original [LLama repo](https://github.com/facebookresearch/llama) and uses sentencepiece under the hood. Instead of training the tokenizer from scratch the `tokenizer.bin` file from llama2 release is used.

The training loop can be found in [`train.py`](./train.py). It expects that the [`prepare_data.py`](./prepare_data.py) script has been run to generate the training data. The training data is expected to be in the `data/` directory.

### Usage

#### Install dependencies
``` Shell
pip install -e .
pip install -e ".[llama]"
```
Get the Llama2 tokenizer, file and place inside the `llama/data` directory.

The following paths are assumed you are in the top level `transformer_nuggets/` directory.

#### Prepare Data
Following the [nanogpt](https://github.com/karpathy/nanoGPT) repo we using huggingface's dataset library to grab openweb data and convert into long strings of tokens that can be used for pretraining.

Then run the following command:

``` Shell
mkdir -p transformer_nuggets/llama/data

python transformer_nuggets/llama/prepare_data.py \
    --tokenizer_path=transformer_nuggets/llama/data/tokenizer.model \
    --output_dir=transformer_nuggets/llama/data/
```
This should take around 3 minutes to run and prepare the training data.

#### Train Model
To edit the training configs take a look at [`train.py`](./train.py). The `entrypoint` function constructs the hyper_param configs as well as the
training configs. By default this will train a 7b model and and save the checkpoints to `transformer_nuggets/llama/data/out/`. It will also save the loss
logs to `transformer_nuggets/llama/data/logs`.


To tain the model using delayed scaling with torch compile run the command
``` Shell
python transformer_nuggets/llama/train.py \
    --fp8_linear_type "delayed" --compile True
```


To finetune model with qlora on single GPU
``` Shell
python transformer_nuggets/llama/finetune.py
```

To finetune model with qlora + FSDP on 2 GPUs
``` Shell
python transformer_nuggets/llama/finetune.py --fsdp_num_gpus 2
```

 ### Notes
To get the Llama2 tokenizer go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access. This will get you pretrained weights as well as the tokenizer.
