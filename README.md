# GGUF → MLX Converter

Convert **GGUF** models to **MLX** (Apple Silicon) safetensors format with zero manual steps.

```bash
uv run python -m gguf2mlx -i model.gguf -o ./mlx-model
```

## What It Does

- **Architecture Detection** — 44+ architectures (Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, DBRX, Grok…)
- **Full Dequantization** — Q4_0, Q8_0, Q4_K, Q6_K, and all GGUF quant types → float16/float32
- **Weight Transposition** — GGUF → HuggingFace layout (correctly handles 2D tensor dimensions)
- **Tokenizer Extraction** — LLama/BPE tokenizers → standard vocab.json, merges.txt, tokenizer.json
- **Intelligent Sharding** — Automatic multi-file safetensors output for large models (>4.5 GB per shard)
- **BOS/EOS Fixes** — Auto-detects and corrects special tokens for Qwen, DeepSeek, and others

## Quick Start

```bash
# Install
pip install gguf2mlx
# or with uv:
uv add gguf2mlx

# Convert any GGUF
gguf2mlx --input model-Q4_K_M.gguf --output ./mlx-model

# Run inference (requires mlx-lm)
uv run demo.py --input model.gguf --prompt "Hello, world!"
```

## Requirements

- Python 3.10+
- `gguf>=0.18.0` — GGUF file reader + dequantization
- `safetensors` — MLX-compatible tensor format
- `numpy`, `tqdm`
- Optional: `mlx`, `mlx-lm` — for inference verification

## Architecture Support

Supports all major LLM architectures in GGUF format:

| Category | Architectures |
|----------|--------------|
| **Llama family** | llama, mistral, falcon, stablelm |
| **Qwen family** | qwen2, qwen2moe, qwen3moe |
| **DeepSeek** | deepseek2, deepseek3 |
| **Gemma** | gemma, gemma2, gemma3 |
| **Phi** | phi, phi3 |
| **GPT family** | gpt2, gptneox, gpt_bigcode, refact |
| **MoE models** | dbrx, grok-1 |
| **Others** | bert, bloom, cohere, olmo, granite, nemotron, exaone, xverse, orion … |

Missing an architecture? PRs welcome.

## Advanced

### Output Data Type

```bash
gguf2mlx -i model.gguf -o ./mlx --dtype float32  # F32 (larger, more precise)
gguf2mlx -i model.gguf -o ./mlx --dtype float16  # F16 (default, good for inference)
```

### Inspect without Converting

```bash
gguf2mlx -i model.gguf --skip-weights
# Prints architecture, tensor count, and all metadata fields
```

## License

MIT
