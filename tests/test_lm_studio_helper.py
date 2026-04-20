#!/usr/bin/env python3
"""
Tests for lm_studio_helper.py
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import lm_studio_helper


class TestCheckLmsCliAvailable(unittest.TestCase):
    """Tests for lm_studio_helper.check_lms_cli_available."""

    def _run(self, returncode=0, stdout="LM Studio 1.0", side_effect=None):
        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout
        if side_effect:
            with patch("lm_studio_helper.subprocess.run", side_effect=side_effect):
                return lm_studio_helper.check_lms_cli_available()
        with patch("lm_studio_helper.subprocess.run", return_value=mock_result):
            return lm_studio_helper.check_lms_cli_available()

    def test_returns_true_when_lms_available(self):
        ok, info = self._run(returncode=0, stdout="LM Studio 0.3.6")
        self.assertTrue(ok)
        self.assertIsInstance(info, str)

    def test_returns_false_when_nonzero_returncode(self):
        ok, info = self._run(returncode=1)
        self.assertFalse(ok)

    def test_returns_false_when_not_found(self):
        ok, info = self._run(side_effect=FileNotFoundError())
        self.assertFalse(ok)
        self.assertIn("not found", info.lower())

    def test_returns_false_on_timeout(self):
        ok, info = self._run(side_effect=subprocess.TimeoutExpired(cmd="lms", timeout=10))
        self.assertFalse(ok)
        self.assertIn("timeout", info.lower())


class TestCheckGgufFileStatus(unittest.TestCase):
    """Tests for lm_studio_helper.check_gguf_file_status."""

    def test_returns_false_for_nonexistent_path(self):
        result = lm_studio_helper.check_gguf_file_status("/no/such/file.gguf")
        self.assertFalse(result)

    def test_returns_false_for_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            tmp = f.name
        try:
            result = lm_studio_helper.check_gguf_file_status(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)

    def test_returns_false_for_invalid_magic(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            # Write more than 1 MB of random-looking bytes with wrong magic
            f.write(b"NOTG" + b"\x00" * (2 * 1024 * 1024))
            tmp = f.name
        try:
            result = lm_studio_helper.check_gguf_file_status(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)

    def test_returns_true_for_valid_gguf_with_sufficient_size(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            # Write valid magic + padding to exceed 100 MB threshold check
            # Actually, we just need > 1 MB and valid GGUF magic
            content = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
            content += b"\x00" * (200 * 1024 * 1024)  # 200 MB padding
            f.write(content)
            tmp = f.name
        try:
            result = lm_studio_helper.check_gguf_file_status(tmp)
            self.assertTrue(result)
        finally:
            os.unlink(tmp)

    def test_returns_true_for_small_but_valid_gguf(self):
        """File > 1 MB with valid GGUF magic should return True."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            content = b"GGUF" + b"\x00" * (2 * 1024 * 1024)  # 2 MB
            f.write(content)
            tmp = f.name
        try:
            result = lm_studio_helper.check_gguf_file_status(tmp)
            self.assertTrue(result)
        finally:
            os.unlink(tmp)

    def test_returns_false_on_read_error(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * (2 * 1024 * 1024))
            tmp = f.name
        try:
            with patch("builtins.open", side_effect=OSError("perm")):
                result = lm_studio_helper.check_gguf_file_status(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)


class TestCheckLmStudioCompatibility(unittest.TestCase):
    """Tests for lm_studio_helper.check_lm_studio_compatibility."""

    def test_gguf_file_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf = Path(tmpdir) / "model.gguf"
            gguf.touch()
            result = lm_studio_helper.check_lm_studio_compatibility(tmpdir)
            self.assertTrue(result)

    def test_mlx_format_is_not_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").touch()
            (Path(tmpdir) / "model.npz").touch()
            result = lm_studio_helper.check_lm_studio_compatibility(tmpdir)
            self.assertFalse(result)

    def test_unknown_format_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = lm_studio_helper.check_lm_studio_compatibility(tmpdir)
            self.assertFalse(result)

    def test_multiple_gguf_files_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.gguf").touch()
            (Path(tmpdir) / "b.gguf").touch()
            result = lm_studio_helper.check_lm_studio_compatibility(tmpdir)
            self.assertTrue(result)


class TestCheckProjectStructure(unittest.TestCase):
    """Smoke-test that check_project_structure runs without raising."""

    def test_runs_without_error(self):
        # The function uses relative paths, so we just verify it doesn't raise
        try:
            lm_studio_helper.check_project_structure()
        except Exception as exc:
            self.fail(f"check_project_structure raised {exc}")


class TestGetLmStudioInstructions(unittest.TestCase):
    """Tests for lm_studio_helper.get_lm_studio_instructions and get_updated_lm_studio_instructions."""

    def test_get_lm_studio_instructions_returns_string(self):
        result = lm_studio_helper.get_lm_studio_instructions()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_get_updated_lm_studio_instructions_returns_string(self):
        result = lm_studio_helper.get_updated_lm_studio_instructions()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_instructions_mention_gguf(self):
        result = lm_studio_helper.get_lm_studio_instructions()
        self.assertIn("gguf", result.lower())

    def test_updated_instructions_mention_cli(self):
        result = lm_studio_helper.get_updated_lm_studio_instructions()
        self.assertIn("lms_cli", result.lower())


class TestCheckLmStudioStatus(unittest.TestCase):
    """Tests for lm_studio_helper.check_lm_studio_status."""

    def test_runs_when_lms_unavailable(self):
        """Should not raise even when lms CLI is missing."""
        with patch("lm_studio_helper.check_lms_cli_available", return_value=(False, "not found")):
            try:
                lm_studio_helper.check_lm_studio_status()
            except Exception as exc:
                self.fail(f"check_lm_studio_status raised {exc}")

    def test_runs_when_lms_available_and_ls_succeeds(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "phi3-mini  3.8GB  gguf\n"
        with patch("lm_studio_helper.check_lms_cli_available", return_value=(True, "v1.0")):
            with patch("lm_studio_helper.subprocess.run", return_value=mock_result):
                try:
                    lm_studio_helper.check_lm_studio_status()
                except Exception as exc:
                    self.fail(f"check_lm_studio_status raised {exc}")

    def test_runs_when_lms_available_and_ls_fails(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("lm_studio_helper.check_lms_cli_available", return_value=(True, "v1.0")):
            with patch("lm_studio_helper.subprocess.run", return_value=mock_result):
                try:
                    lm_studio_helper.check_lm_studio_status()
                except Exception as exc:
                    self.fail(f"check_lm_studio_status raised {exc}")


if __name__ == "__main__":
    unittest.main()
