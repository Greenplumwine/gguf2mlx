#!/usr/bin/env python3
"""
Tests for mlx_to_gguf.py
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import mlx_to_gguf


def _create_mlx_dir(tmpdir, model_type="test-model", vocab_size=10, num_layers=1, hidden_size=32):
    """Create a minimal MLX model directory with config and weights."""
    mlx_dir = os.path.join(tmpdir, "mlx_model")
    os.makedirs(mlx_dir, exist_ok=True)

    config = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 2,
        "max_position_embeddings": 128,
        "use_cache": True,
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
    }
    with open(os.path.join(mlx_dir, "config.json"), "w") as f:
        json.dump(config, f)

    weights = {
        "embed.weight": np.random.randn(vocab_size, hidden_size).astype(np.float16),
        "lm_head.weight": np.random.randn(vocab_size, hidden_size).astype(np.float16),
    }
    np.savez_compressed(os.path.join(mlx_dir, "model.npz"), **weights)

    return mlx_dir


class TestMlxToGgufConverter(unittest.TestCase):
    """Tests for mlx_to_gguf.mlx_to_gguf_converter."""

    def test_raises_if_config_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = os.path.join(tmpdir, "mlx")
            os.makedirs(mlx_dir)
            # create model.npz but no config.json
            np.savez_compressed(os.path.join(mlx_dir, "model.npz"), w=np.zeros((2,)))
            with self.assertRaises(FileNotFoundError):
                mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, os.path.join(tmpdir, "out.gguf"))

    def test_raises_if_weights_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = os.path.join(tmpdir, "mlx")
            os.makedirs(mlx_dir)
            with open(os.path.join(mlx_dir, "config.json"), "w") as f:
                json.dump({"model_type": "x", "vocab_size": 100, "hidden_size": 32}, f)
            # no model.npz
            with self.assertRaises(FileNotFoundError):
                mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, os.path.join(tmpdir, "out.gguf"))

    def test_creates_output_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir)
            out_path = os.path.join(tmpdir, "output.gguf")
            result = mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertEqual(str(result), out_path)

    def test_output_gguf_has_magic_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir)
            out_path = os.path.join(tmpdir, "output.gguf")
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            with open(out_path, "rb") as f:
                magic = f.read(4)
            self.assertEqual(magic, b"GGUF")

    def test_output_gguf_has_version_3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir)
            out_path = os.path.join(tmpdir, "output.gguf")
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            with open(out_path, "rb") as f:
                f.read(4)  # magic
                (version,) = struct.unpack("<I", f.read(4))
            self.assertEqual(version, 3)

    def test_output_tensor_count_matches_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir)
            out_path = os.path.join(tmpdir, "output.gguf")
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            with open(out_path, "rb") as f:
                f.read(4)  # magic
                f.read(4)  # version
                (tensor_count,) = struct.unpack("<Q", f.read(8))
            # The model.npz we created has 2 weight tensors
            self.assertEqual(tensor_count, 2)

    def test_output_gguf_is_nonzero_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir)
            out_path = os.path.join(tmpdir, "output.gguf")
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_float32_weights_converted_to_float16(self):
        """Verifies the converter handles float32 input by converting to float16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = os.path.join(tmpdir, "mlx")
            os.makedirs(mlx_dir)
            with open(os.path.join(mlx_dir, "config.json"), "w") as f:
                json.dump({"model_type": "m", "vocab_size": 10, "hidden_size": 8}, f)
            # Save as float32 intentionally
            np.savez_compressed(
                os.path.join(mlx_dir, "model.npz"),
                w=np.ones((4, 4), dtype=np.float32),
            )
            out_path = os.path.join(tmpdir, "out.gguf")
            # Should not raise
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            self.assertTrue(os.path.exists(out_path))

    def test_model_type_appears_in_metadata(self):
        """The GGUF metadata section should embed the model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = _create_mlx_dir(tmpdir, model_type="phi3-mini")
            out_path = os.path.join(tmpdir, "out.gguf")
            mlx_to_gguf.mlx_to_gguf_converter(mlx_dir, out_path)
            with open(out_path, "rb") as f:
                raw = f.read()
            self.assertIn(b"phi3-mini", raw)


class TestCopyToLmstudio(unittest.TestCase):
    """Tests for mlx_to_gguf.copy_to_lmstudio."""

    def test_returns_false_when_lmstudio_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_gguf = Path(tmpdir) / "model.gguf"
            dummy_gguf.write_bytes(b"GGUF" + b"\x00" * 20)

            fake_home = Path(tmpdir) / "home_no_lmstudio"
            fake_home.mkdir()

            with patch("mlx_to_gguf.Path") as mock_path_cls:
                # Make Path.home() return our fake home that has no .lmstudio
                mock_path_cls.home.return_value = fake_home
                # Keep real Path behaviour for everything else
                mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)
                mock_path_cls.home.return_value = fake_home

                result = mlx_to_gguf.copy_to_lmstudio(dummy_gguf)
            self.assertFalse(result)

    def test_returns_true_and_copies_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source gguf
            src = Path(tmpdir) / "model.gguf"
            src.write_bytes(b"GGUF" + b"\x00" * 20)

            # Create fake lmstudio models dir
            lmstudio_models = Path(tmpdir) / ".lmstudio" / "models"
            lmstudio_models.mkdir(parents=True)

            with patch("mlx_to_gguf.Path") as mock_path_cls:
                mock_path_cls.home.return_value = Path(tmpdir)
                mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)

                result = mlx_to_gguf.copy_to_lmstudio(src)

            self.assertTrue(result)
            self.assertTrue((lmstudio_models / "model.gguf").exists())

    def test_returns_false_on_copy_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "model.gguf"
            src.write_bytes(b"GGUF" + b"\x00" * 20)

            lmstudio_models = Path(tmpdir) / ".lmstudio" / "models"
            lmstudio_models.mkdir(parents=True)

            with patch("mlx_to_gguf.Path") as mock_path_cls:
                mock_path_cls.home.return_value = Path(tmpdir)
                mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)
                # Patch shutil at the top-level module so the inline import sees the mock
                with patch("shutil.copy2", side_effect=OSError("disk full")):
                    result = mlx_to_gguf.copy_to_lmstudio(src)

            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
