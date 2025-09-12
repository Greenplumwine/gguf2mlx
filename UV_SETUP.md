# UV Installation Instructions for GGUF2MLX

## Quick Setup with UV (Recommended)

### Install UV
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or via pip
pip install uv
```

### Project Setup
```bash
# Clone or navigate to project directory
cd gguf2mlx

# Create virtual environment and install dependencies
uv sync

# Activate the environment (if needed for manual operations)
source .venv/bin/activate
```

### Development Setup
```bash
# Install with development dependencies
uv sync --extra dev

# Install all optional dependencies
uv sync --all-extras
```

### Running Commands
```bash
# Convert GGUF to MLX
uv run python gguf2mlx.py --input ./models/phi3-mini.gguf --output ./models/phi3-mini-mlx

# Run demos
uv run python demo.py --input ./models/phi3-mini-mlx --prompt "Hello world" --output result.txt
uv run python demo1.py --input ./models/phi3-mini-mlx --prompt "Explain AI" --output ai_result.txt
```

### Package Management
```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade

# Show installed packages
uv pip list
```

## Why UV?

✅ **Faster**: 10-100x faster than pip
✅ **Deterministic**: Exact reproducible installs via lock file
✅ **Reliable**: Resolves dependency conflicts automatically
✅ **Simple**: Single command setup and management
✅ **Compatible**: Works with existing pip/conda workflows