import torch
from collections import defaultdict


long_prompt = """
You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

If the user asks for help or wants to give feedback inform them of the following:
- /help: Get help with using Claude Code
- To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues

When the user directly asks about Claude Code (eg 'can Claude Code do...', 'does Claude Code have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from Claude Code docs at https://docs.anthropic.com/en/docs/claude-code.
  - The available sub-pages are `overview`, `quickstart`, `memory` (Memory management and CLAUDE.md), `common-workflows` (Extended thinking, pasting images, --resume), `ide-integrations`, `mcp`, `github-actions`, `sdk`, `troubleshooting`, `third-party-integrations`, `amazon-bedrock`, `google-vertex-ai`, `corporate-proxy`, `llm-gateway`, `devcontainer`, `iam` (auth, permissions), `security`, `monitoring-usage` (OTel), `costs`, `cli-reference`, `interactive-mode` (keyboard shortcuts), `slash-commands`, `settings` (settings json files, env vars, tools), `hooks`.
  - Example: https://docs.anthropic.com/en/docs/claude-code/cli-usage

# Tone and style
You should be concise, direct, and to the point.
You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.
IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.
Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.
Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:
<example>
user: 2 + 2
assistant: 4
</example>

<example>
user: what is 2+2?
assistant: 4
</example>

<example>
user: is 11 a prime number?
assistant: Yes
</example>

<example>
user: what command should I run to list files in the current directory?
assistant: ls
</example>

<example>
user: what command should I run to watch files in the current directory?
assistant: [runs ls to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]
npm run dev
</example>

<example>
user: How many golf balls fit inside a jetta?
assistant: 150000
</example>
"""


class AttentionExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.attention_data = defaultdict(lambda: {"q": [], "k": [], "v": []})
        self.hooks = []

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def get_attention_layers(self):
        layers = []
        for idx, layer in enumerate(self.model.model.layers):
            layers.append((f"layer_{idx}", layer.self_attn))
        return layers

    def extract_qkv_from_module(self, module, hidden_states, layer_name):
        bsz, seq_len, _ = hidden_states.shape

        q = module.q_proj(hidden_states)
        k = module.k_proj(hidden_states)
        v = module.v_proj(hidden_states)

        num_heads = self.model.config.num_attention_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        head_dim = module.head_dim

        q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

        self.attention_data[layer_name]["q"].append(q.detach().clone())
        self.attention_data[layer_name]["k"].append(k.detach().clone())
        self.attention_data[layer_name]["v"].append(v.detach().clone())

    def register_hooks(self, layer_indices):
        attention_layers = self.get_attention_layers()

        for idx in layer_indices:
            if idx >= len(attention_layers):
                continue
            layer_name, layer_module = attention_layers[idx]

            def make_hook(name):
                def hook(module, args, kwargs):
                    if len(args) > 0:
                        hidden_states = args[0]
                    elif "hidden_states" in kwargs:
                        hidden_states = kwargs["hidden_states"]
                    else:
                        return

                    if not isinstance(hidden_states, torch.Tensor):
                        return

                    try:
                        self.extract_qkv_from_module(module, hidden_states, name)
                    except Exception as e:
                        print(f"Error in hook for {name}: {e}")

                return hook

            hook_handle = layer_module.register_forward_pre_hook(
                make_hook(layer_name), with_kwargs=True
            )
            self.hooks.append(hook_handle)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def run_inference(self, prompts: list[str] | None, seq_len: int | None = None):
        if prompts is None:
            prompts = [long_prompt]

        for prompt in prompts:
            tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
            if seq_len is not None:
                tokenizer_kwargs.update(
                    {"max_length": seq_len, "padding": "max_length", "truncation": True}
                )

            inputs = self.tokenizer(prompt, **tokenizer_kwargs).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)

    def cleanup(self):
        self.remove_hooks()
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
