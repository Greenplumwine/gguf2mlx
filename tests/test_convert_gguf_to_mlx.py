#!/usr/bin/env python3
"""
Tests for convert_gguf_to_mlx.py
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide stubs for optional packages before importing the module under test
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", MagicMock())
sys.modules.setdefault("mlx.nn", MagicMock())
sys.modules.setdefault("llama_cpp", MagicMock())

import convert_gguf_to_mlx


def _make_gguf_bytes(version=3, tensor_count=5, metadata_count=10):
    data = b"GGUF"
    data += struct.pack("<I", version)
    data += struct.pack("<Q", tensor_count)
    data += struct.pack("<Q", metadata_count)
    return data


class TestReadGgufHeader(unittest.TestCase):
    """Tests for convert_gguf_to_mlx.read_gguf_header (mirrors gguf2mlx)."""

    def test_valid_header(self):
        data = _make_gguf_bytes(version=2, tensor_count=3, metadata_count=4)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(data)
            tmp = f.name
        try:
            result = convert_gguf_to_mlx.read_gguf_header(tmp)
            self.assertEqual(result["version"], 2)
            self.assertEqual(result["tensor_count"], 3)
            self.assertEqual(result["metadata_count"], 4)
        finally:
            os.unlink(tmp)

    def test_invalid_magic_returns_none(self):
        data = b"XXXX" + struct.pack("<I", 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            self.assertIsNone(convert_gguf_to_mlx.read_gguf_header(tmp))
        finally:
            os.unlink(tmp)

    def test_nonexistent_file_returns_none(self):
        self.assertIsNone(convert_gguf_to_mlx.read_gguf_header("/does/not/exist.gguf"))


class TestCreateMlxConfig(unittest.TestCase):
    """Tests for convert_gguf_to_mlx.create_mlx_config."""

    def test_intermediate_size_is_four_x_hidden(self):
        cfg = convert_gguf_to_mlx.create_mlx_config({"hidden_size": 256})
        self.assertEqual(cfg["intermediate_size"], 256 * 4)

    def test_model_type_priority(self):
        # model_type from weights_info takes priority over model_name
        cfg = convert_gguf_to_mlx.create_mlx_config(
            {"model_type": "from-weights"}, model_name="from-arg"
        )
        self.assertEqual(cfg["model_type"], "from-weights")

    def test_model_name_fallback(self):
        cfg = convert_gguf_to_mlx.create_mlx_config({}, model_name="arg-model")
        self.assertEqual(cfg["model_type"], "arg-model")

    def test_empty_weights_info_uses_defaults(self):
        cfg = convert_gguf_to_mlx.create_mlx_config({})
        self.assertEqual(cfg["vocab_size"], 32000)
        self.assertEqual(cfg["hidden_size"], 1536)
        self.assertEqual(cfg["num_hidden_layers"], 24)
        self.assertEqual(cfg["num_attention_heads"], 24)

    def test_custom_values_override_defaults(self):
        cfg = convert_gguf_to_mlx.create_mlx_config(
            {"vocab_size": 50000, "hidden_size": 1024, "num_layers": 12, "num_heads": 16}
        )
        self.assertEqual(cfg["vocab_size"], 50000)
        self.assertEqual(cfg["hidden_size"], 1024)
        self.assertEqual(cfg["num_hidden_layers"], 12)
        self.assertEqual(cfg["num_attention_heads"], 16)


class TestExtractGgufWeights(unittest.TestCase):
    """Tests for convert_gguf_to_mlx.extract_gguf_weights."""

    def test_returns_none_without_llama_cpp(self):
        original = convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE
        try:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = False
            self.assertIsNone(convert_gguf_to_mlx.extract_gguf_weights("/any.gguf"))
        finally:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = original

    def test_returns_none_on_llama_exception(self):
        original = convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE
        try:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = True
            with patch("convert_gguf_to_mlx.Llama", side_effect=RuntimeError("fail")):
                self.assertIsNone(
                    convert_gguf_to_mlx.extract_gguf_weights("/any.gguf")
                )
        finally:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = original

    def test_model_type_derived_from_stem(self):
        original = convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE
        try:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = True
            with patch("convert_gguf_to_mlx.Llama", return_value=MagicMock()):
                result = convert_gguf_to_mlx.extract_gguf_weights("/models/llama-7b.gguf")
            self.assertEqual(result["model_type"], "llama-7b")
        finally:
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = original


class TestSaveMlxModel(unittest.TestCase):
    """Tests for convert_gguf_to_mlx.save_mlx_model."""

    def _base_config(self, model_type="generic", vocab_size=50):
        return {
            "model_type": model_type,
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_position_embeddings": 128,
            "use_cache": True,
            "torch_dtype": "float16",
            "transformers_version": "4.36.0",
        }

    def test_creates_expected_files(self):
        import numpy as np

        cfg = self._base_config(vocab_size=10)
        weights = {"w": MagicMock()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with patch("convert_gguf_to_mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                result = convert_gguf_to_mlx.save_mlx_model(cfg, weights, out)
            self.assertIsNotNone(result)
            for fname in ("config.json", "model.npz", "tokenizer_config.json", "vocab.json"):
                self.assertTrue(os.path.exists(os.path.join(out, fname)), f"Missing: {fname}")

    def test_phi3_tokenizer_config(self):
        import numpy as np

        cfg = self._base_config(model_type="phi3-mini", vocab_size=5)
        weights = {"w": MagicMock()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with patch("convert_gguf_to_mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                convert_gguf_to_mlx.save_mlx_model(cfg, weights, out)
            with open(os.path.join(out, "tokenizer_config.json")) as f:
                tok = json.load(f)
            self.assertIn("special_tokens", tok)

    def test_non_phi3_tokenizer_config(self):
        import numpy as np

        cfg = self._base_config(model_type="mistral", vocab_size=5)
        weights = {"w": MagicMock()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with patch("convert_gguf_to_mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                convert_gguf_to_mlx.save_mlx_model(cfg, weights, out)
            with open(os.path.join(out, "tokenizer_config.json")) as f:
                tok = json.load(f)
            self.assertIn("speaker_tokens", tok)

    def test_returns_none_on_os_error(self):
        import numpy as np

        cfg = self._base_config(vocab_size=5)
        weights = {"w": MagicMock()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            # Trigger error inside the try block
            with patch("convert_gguf_to_mlx.json.dump", side_effect=OSError("disk full")):
                with patch("convert_gguf_to_mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                    result = convert_gguf_to_mlx.save_mlx_model(cfg, weights, out)
        self.assertIsNone(result)


class TestConvertGgufToMlxIntegration(unittest.TestCase):
    """Integration tests for convert_gguf_to_mlx.convert_gguf_to_mlx."""

    def test_returns_false_when_mlx_unavailable(self):
        orig = convert_gguf_to_mlx.MLX_AVAILABLE
        try:
            convert_gguf_to_mlx.MLX_AVAILABLE = False
            self.assertFalse(convert_gguf_to_mlx.convert_gguf_to_mlx("/a.gguf", "/out"))
        finally:
            convert_gguf_to_mlx.MLX_AVAILABLE = orig

    def test_returns_false_for_missing_file(self):
        orig = convert_gguf_to_mlx.MLX_AVAILABLE
        try:
            convert_gguf_to_mlx.MLX_AVAILABLE = True
            self.assertFalse(convert_gguf_to_mlx.convert_gguf_to_mlx("/no.gguf", "/out"))
        finally:
            convert_gguf_to_mlx.MLX_AVAILABLE = orig

    def test_successful_end_to_end(self):
        import numpy as np

        orig_mlx = convert_gguf_to_mlx.MLX_AVAILABLE
        orig_llama = convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE
        try:
            convert_gguf_to_mlx.MLX_AVAILABLE = True
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = False

            with tempfile.TemporaryDirectory() as tmpdir:
                gguf_path = os.path.join(tmpdir, "mymodel.gguf")
                with open(gguf_path, "wb") as f:
                    f.write(_make_gguf_bytes())
                out_dir = os.path.join(tmpdir, "out")

                dummy = np.zeros((2, 2), dtype=np.float16)
                mock_mx = MagicMock()
                mock_mx.random.normal.return_value = dummy
                mock_mx.ones.return_value = dummy
                mock_mx.float16 = np.float16

                with patch("convert_gguf_to_mlx.mx", mock_mx):
                    with patch("convert_gguf_to_mlx.np.array", return_value=dummy):
                        ok = convert_gguf_to_mlx.convert_gguf_to_mlx(gguf_path, out_dir)

                self.assertTrue(ok)
                self.assertTrue(os.path.exists(os.path.join(out_dir, "config.json")))
        finally:
            convert_gguf_to_mlx.MLX_AVAILABLE = orig_mlx
            convert_gguf_to_mlx.LLAMA_CPP_AVAILABLE = orig_llama


if __name__ == "__main__":
    unittest.main()
