#!/usr/bin/env python3
"""
Tests for gguf2mlx.py
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Provide a stub for mlx so the module can be imported without the real package
mlx_core_stub = MagicMock()
mlx_nn_stub = MagicMock()
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", mlx_core_stub)
sys.modules.setdefault("mlx.nn", mlx_nn_stub)
sys.modules.setdefault("llama_cpp", MagicMock())

import importlib

# Re-import with stubs in place
import gguf2mlx


def _make_gguf_bytes(version=3, tensor_count=5, metadata_count=10):
    """Helper: build a minimal valid GGUF binary header."""
    data = b"GGUF"
    data += struct.pack("<I", version)
    data += struct.pack("<Q", tensor_count)
    data += struct.pack("<Q", metadata_count)
    return data


class TestReadGgufHeader(unittest.TestCase):
    """Tests for gguf2mlx.read_gguf_header."""

    def test_valid_header_returns_correct_values(self):
        data = _make_gguf_bytes(version=3, tensor_count=7, metadata_count=12)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(data)
            tmp_path = f.name
        try:
            result = gguf2mlx.read_gguf_header(tmp_path)
            self.assertIsNotNone(result)
            self.assertEqual(result["version"], 3)
            self.assertEqual(result["tensor_count"], 7)
            self.assertEqual(result["metadata_count"], 12)
        finally:
            os.unlink(tmp_path)

    def test_valid_header_version_1(self):
        data = _make_gguf_bytes(version=1, tensor_count=0, metadata_count=0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(data)
            tmp_path = f.name
        try:
            result = gguf2mlx.read_gguf_header(tmp_path)
            self.assertIsNotNone(result)
            self.assertEqual(result["version"], 1)
        finally:
            os.unlink(tmp_path)

    def test_invalid_magic_returns_none(self):
        data = b"NOTG" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            tmp_path = f.name
        try:
            result = gguf2mlx.read_gguf_header(tmp_path)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_returns_none(self):
        result = gguf2mlx.read_gguf_header("/nonexistent/path/model.gguf")
        self.assertIsNone(result)

    def test_truncated_file_returns_none(self):
        # Only write magic + 2 bytes (not enough for version field)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF\x01\x00")
            tmp_path = f.name
        try:
            result = gguf2mlx.read_gguf_header(tmp_path)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            tmp_path = f.name
        try:
            result = gguf2mlx.read_gguf_header(tmp_path)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)


class TestCreateMlxConfig(unittest.TestCase):
    """Tests for gguf2mlx.create_mlx_config."""

    def _make_weights_info(self, **kwargs):
        base = {
            "model_type": "test-model",
            "vocab_size": 32000,
            "hidden_size": 1536,
            "num_layers": 24,
            "num_heads": 24,
        }
        base.update(kwargs)
        return base

    def test_basic_config_keys(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info())
        expected_keys = {
            "model_type",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
            "use_cache",
            "torch_dtype",
            "transformers_version",
        }
        self.assertEqual(set(cfg.keys()), expected_keys)

    def test_intermediate_size_is_four_times_hidden(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info(hidden_size=512))
        self.assertEqual(cfg["intermediate_size"], 512 * 4)

    def test_model_type_from_weights_info(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info(model_type="phi3-mini"))
        self.assertEqual(cfg["model_type"], "phi3-mini")

    def test_model_name_fallback_when_no_model_type_in_weights(self):
        weights_info = {
            "vocab_size": 32000,
            "hidden_size": 1536,
            "num_layers": 24,
            "num_heads": 24,
        }
        cfg = gguf2mlx.create_mlx_config(weights_info, model_name="fallback-model")
        self.assertEqual(cfg["model_type"], "fallback-model")

    def test_unknown_model_when_both_absent(self):
        weights_info = {
            "vocab_size": 32000,
            "hidden_size": 1536,
            "num_layers": 24,
            "num_heads": 24,
        }
        cfg = gguf2mlx.create_mlx_config(weights_info)
        self.assertEqual(cfg["model_type"], "unknown-model")

    def test_defaults_when_keys_missing(self):
        cfg = gguf2mlx.create_mlx_config({})
        self.assertEqual(cfg["vocab_size"], 32000)
        self.assertEqual(cfg["hidden_size"], 1536)
        self.assertEqual(cfg["num_hidden_layers"], 24)
        self.assertEqual(cfg["num_attention_heads"], 24)

    def test_use_cache_is_true(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info())
        self.assertTrue(cfg["use_cache"])

    def test_torch_dtype_float16(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info())
        self.assertEqual(cfg["torch_dtype"], "float16")

    def test_max_position_embeddings(self):
        cfg = gguf2mlx.create_mlx_config(self._make_weights_info())
        self.assertEqual(cfg["max_position_embeddings"], 4096)


class TestExtractGgufWeights(unittest.TestCase):
    """Tests for gguf2mlx.extract_gguf_weights."""

    def test_returns_none_when_llama_cpp_unavailable(self):
        original = gguf2mlx.LLAMA_CPP_AVAILABLE
        try:
            gguf2mlx.LLAMA_CPP_AVAILABLE = False
            result = gguf2mlx.extract_gguf_weights("/any/path.gguf")
            self.assertIsNone(result)
        finally:
            gguf2mlx.LLAMA_CPP_AVAILABLE = original

    def test_returns_none_on_exception(self):
        original = gguf2mlx.LLAMA_CPP_AVAILABLE
        try:
            gguf2mlx.LLAMA_CPP_AVAILABLE = True
            with patch("gguf2mlx.Llama", side_effect=RuntimeError("load failed")):
                result = gguf2mlx.extract_gguf_weights("/any/path.gguf")
            self.assertIsNone(result)
        finally:
            gguf2mlx.LLAMA_CPP_AVAILABLE = original

    def test_returns_dict_with_expected_keys_on_success(self):
        original = gguf2mlx.LLAMA_CPP_AVAILABLE
        try:
            gguf2mlx.LLAMA_CPP_AVAILABLE = True
            mock_llama = MagicMock()
            with patch("gguf2mlx.Llama", return_value=mock_llama):
                result = gguf2mlx.extract_gguf_weights("/models/phi3-mini.gguf")
            self.assertIsNotNone(result)
            for key in ("model_type", "vocab_size", "hidden_size", "num_layers", "num_heads"):
                self.assertIn(key, result)
        finally:
            gguf2mlx.LLAMA_CPP_AVAILABLE = original

    def test_model_type_derived_from_filename(self):
        original = gguf2mlx.LLAMA_CPP_AVAILABLE
        try:
            gguf2mlx.LLAMA_CPP_AVAILABLE = True
            with patch("gguf2mlx.Llama", return_value=MagicMock()):
                result = gguf2mlx.extract_gguf_weights("/models/my-custom-model.gguf")
            self.assertEqual(result["model_type"], "my-custom-model")
        finally:
            gguf2mlx.LLAMA_CPP_AVAILABLE = original


class TestSaveMlxModel(unittest.TestCase):
    """Tests for gguf2mlx.save_mlx_model."""

    def _make_config(self, model_type="test-model", vocab_size=100):
        return {
            "model_type": model_type,
            "vocab_size": vocab_size,
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 512,
            "use_cache": True,
            "torch_dtype": "float16",
            "transformers_version": "4.36.0",
        }

    def _make_numpy_weights(self):
        import numpy as np
        return {
            "embed.weight": MagicMock(**{"__class__": MagicMock()}),
        }

    def test_creates_output_directory(self):
        import numpy as np

        config = self._make_config(vocab_size=10)
        # Use numpy arrays directly (no mlx needed)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "model_out")
            # Patch np.array to return a real numpy array so savez_compressed works
            with patch("gguf2mlx.np.array", return_value=np.zeros((2, 2), dtype=np.float16)):
                result = gguf2mlx.save_mlx_model(config, weights, output_dir)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isdir(output_dir))

    def test_config_json_written_correctly(self):
        import numpy as np

        config = self._make_config(model_type="phi3-mini", vocab_size=10)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            with patch("gguf2mlx.np.array", return_value=np.zeros((2, 2), dtype=np.float16)):
                gguf2mlx.save_mlx_model(config, weights, output_dir)
            with open(os.path.join(output_dir, "config.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["model_type"], "phi3-mini")
            self.assertEqual(saved["vocab_size"], 10)

    def test_tokenizer_config_phi3_has_special_tokens(self):
        import numpy as np

        config = self._make_config(model_type="phi3-mini", vocab_size=10)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            with patch("gguf2mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                gguf2mlx.save_mlx_model(config, weights, output_dir)
            with open(os.path.join(output_dir, "tokenizer_config.json")) as f:
                tok = json.load(f)
            self.assertIn("special_tokens", tok)
            self.assertIn("<|system|>", tok["special_tokens"])

    def test_tokenizer_config_non_phi3_has_speaker_tokens(self):
        import numpy as np

        config = self._make_config(model_type="vibevoice", vocab_size=10)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            with patch("gguf2mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                gguf2mlx.save_mlx_model(config, weights, output_dir)
            with open(os.path.join(output_dir, "tokenizer_config.json")) as f:
                tok = json.load(f)
            self.assertIn("speaker_tokens", tok)
            self.assertIn("speak_token", tok)

    def test_vocab_json_has_correct_count(self):
        import numpy as np

        vocab_size = 150
        config = self._make_config(vocab_size=vocab_size)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            with patch("gguf2mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                gguf2mlx.save_mlx_model(config, weights, output_dir)
            with open(os.path.join(output_dir, "vocab.json")) as f:
                vocab = json.load(f)
            self.assertEqual(len(vocab), vocab_size)

    def test_vocab_json_first_100_tokens_have_angle_brackets(self):
        import numpy as np

        config = self._make_config(vocab_size=110)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            with patch("gguf2mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                gguf2mlx.save_mlx_model(config, weights, output_dir)
            with open(os.path.join(output_dir, "vocab.json")) as f:
                vocab = json.load(f)
            self.assertIn("<token_0>", vocab)
            self.assertIn("<token_99>", vocab)
            self.assertIn("token_100", vocab)

    def test_returns_none_on_write_error(self):
        import numpy as np

        config = self._make_config(vocab_size=10)
        weights = {"w": MagicMock()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            # Patch json.dump to raise once the directory exists and mkdir succeeds
            with patch("gguf2mlx.json.dump", side_effect=OSError("disk full")):
                with patch("gguf2mlx.np.array", return_value=np.zeros((2,), dtype=np.float16)):
                    result = gguf2mlx.save_mlx_model(config, weights, output_dir)
        self.assertIsNone(result)


class TestConvertGgufToMlx(unittest.TestCase):
    """Integration tests for gguf2mlx.convert_gguf_to_mlx."""

    def _write_gguf(self, path):
        data = _make_gguf_bytes(version=3, tensor_count=2, metadata_count=3)
        with open(path, "wb") as f:
            f.write(data)

    def test_returns_false_when_mlx_not_available(self):
        original = gguf2mlx.MLX_AVAILABLE
        try:
            gguf2mlx.MLX_AVAILABLE = False
            result = gguf2mlx.convert_gguf_to_mlx("/any.gguf", "/any_out")
            self.assertFalse(result)
        finally:
            gguf2mlx.MLX_AVAILABLE = original

    def test_returns_false_when_gguf_file_not_found(self):
        original = gguf2mlx.MLX_AVAILABLE
        try:
            gguf2mlx.MLX_AVAILABLE = True
            result = gguf2mlx.convert_gguf_to_mlx("/nonexistent.gguf", "/any_out")
            self.assertFalse(result)
        finally:
            gguf2mlx.MLX_AVAILABLE = original

    def test_returns_false_when_header_invalid(self):
        original = gguf2mlx.MLX_AVAILABLE
        try:
            gguf2mlx.MLX_AVAILABLE = True
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
                f.write(b"NOTGGUF")
                tmp = f.name
            try:
                result = gguf2mlx.convert_gguf_to_mlx(tmp, "/any_out")
                self.assertFalse(result)
            finally:
                os.unlink(tmp)
        finally:
            gguf2mlx.MLX_AVAILABLE = original

    def test_successful_conversion(self):
        import numpy as np

        original_mlx = gguf2mlx.MLX_AVAILABLE
        original_llama = gguf2mlx.LLAMA_CPP_AVAILABLE
        try:
            gguf2mlx.MLX_AVAILABLE = True
            gguf2mlx.LLAMA_CPP_AVAILABLE = False  # triggers default weights_info path

            with tempfile.TemporaryDirectory() as tmpdir:
                gguf_path = os.path.join(tmpdir, "testmodel.gguf")
                self._write_gguf(gguf_path)
                output_dir = os.path.join(tmpdir, "out")

                dummy_arr = np.zeros((2, 2), dtype=np.float16)
                mock_mx = MagicMock()
                mock_mx.random.normal.return_value = dummy_arr
                mock_mx.ones.return_value = dummy_arr
                mock_mx.float16 = np.float16

                with patch.dict(sys.modules, {"mlx.core": mock_mx}):
                    with patch("gguf2mlx.mx", mock_mx):
                        with patch("gguf2mlx.np.array", return_value=dummy_arr):
                            result = gguf2mlx.convert_gguf_to_mlx(gguf_path, output_dir)

                self.assertTrue(result)
                self.assertTrue(os.path.exists(os.path.join(output_dir, "config.json")))
        finally:
            gguf2mlx.MLX_AVAILABLE = original_mlx
            gguf2mlx.LLAMA_CPP_AVAILABLE = original_llama


class TestCreateDummyMlxWeights(unittest.TestCase):
    """Tests for gguf2mlx.create_dummy_mlx_weights."""

    def _make_config(self, vocab_size=100, hidden_size=16, num_layers=2, num_heads=2):
        return {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 4,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
        }

    def test_weight_count_matches_expected(self):
        import numpy as np

        cfg = self._make_config(num_layers=2)
        # per layer: q/k/v/o proj (4) + gate/up/down mlp (3) + ln_1/ln_2 (2) = 9
        # global: embeddings (1) + ln_f (1) + lm_head (1) + audio_encoder (1) + audio_decoder (1)
        expected = 2 * 9 + 5

        mock_mx = MagicMock()
        mock_mx.random.normal.return_value = np.zeros((2, 2), dtype=np.float16)
        mock_mx.ones.return_value = np.zeros((2,), dtype=np.float16)
        mock_mx.float16 = np.float16

        with patch("gguf2mlx.mx", mock_mx):
            weights = gguf2mlx.create_dummy_mlx_weights(cfg)

        self.assertEqual(len(weights), expected)

    def test_expected_key_names_present(self):
        import numpy as np

        cfg = self._make_config(num_layers=1)
        mock_mx = MagicMock()
        mock_mx.random.normal.return_value = np.zeros((2, 2), dtype=np.float16)
        mock_mx.ones.return_value = np.zeros((2,), dtype=np.float16)
        mock_mx.float16 = np.float16

        with patch("gguf2mlx.mx", mock_mx):
            weights = gguf2mlx.create_dummy_mlx_weights(cfg)

        self.assertIn("embeddings.word_embeddings.weight", weights)
        self.assertIn("transformer.h.0.attn.q_proj.weight", weights)
        self.assertIn("transformer.h.0.mlp.gate_proj.weight", weights)
        self.assertIn("transformer.ln_f.weight", weights)
        self.assertIn("lm_head.weight", weights)
        self.assertIn("audio_encoder.weight", weights)
        self.assertIn("audio_decoder.weight", weights)


if __name__ == "__main__":
    unittest.main()
