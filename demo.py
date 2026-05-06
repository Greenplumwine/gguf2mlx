#!/usr/bin/env python3
"""
Demo: GGUF → MLX Inference Pipeline
Converts a GGUF model and runs inference in one step.
Usage: uv run python demo.py --input model.gguf --prompt "Hello"
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Add src dir for gguf2mlx import
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gguf2mlx import convert


def main():
    parser = argparse.ArgumentParser(description="GGUF → MLX Inference Demo")
    parser.add_argument("--input", "-i", required=True, help="Input GGUF file")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--keep", action="store_true", help="Keep converted files")
    parser.add_argument("--output", "-o", help="Output directory (default: temp)")

    args = parser.parse_args()

    if args.keep and args.output:
        output_dir = Path(args.output)
    elif args.keep:
        output_dir = Path(args.input).with_suffix(".mlx")
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="gguf2mlx_"))

    print(f"Converting {args.input} → {output_dir}")
    success = convert(args.input, str(output_dir))
    if not success:
        sys.exit(1)

    print(f"\n--- Running inference ---")

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("⚠ mlx_lm not installed — skipping inference.")
        print(f"  Converted files saved to: {output_dir}")
        return

    model, tokenizer = load(str(output_dir))
    response = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
    print(f"\n{'=' * 60}")
    print(f"Prompt:   {args.prompt}")
    print(f"Response: {response}")
    print(f"{'=' * 60}")

    if not args.keep:
        import shutil

        shutil.rmtree(output_dir)
        print("(Cleaned up temp files)")


if __name__ == "__main__":
    main()
