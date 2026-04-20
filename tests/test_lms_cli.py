#!/usr/bin/env python3
"""
Tests for lms_cli.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import lms_cli


class TestCheckLmsAvailable(unittest.TestCase):
    """Tests for lms_cli.check_lms_available."""

    def _mock_run(self, returncode=0, stdout="LM Studio 1.0", side_effect=None):
        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout
        if side_effect:
            with patch("lms_cli.subprocess.run", side_effect=side_effect):
                return lms_cli.check_lms_available()
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            return lms_cli.check_lms_available()

    def test_returns_true_when_available(self):
        self.assertTrue(self._mock_run(returncode=0))

    def test_returns_false_when_nonzero_return(self):
        self.assertFalse(self._mock_run(returncode=1))

    def test_returns_false_when_not_found(self):
        self.assertFalse(self._mock_run(side_effect=FileNotFoundError()))

    def test_returns_false_on_timeout(self):
        self.assertFalse(
            self._mock_run(
                side_effect=subprocess.TimeoutExpired(cmd="lms", timeout=10)
            )
        )


class TestImportModelToLmstudio(unittest.TestCase):
    """Tests for lms_cli.import_model_to_lmstudio."""

    def test_returns_false_when_file_not_found(self):
        result = lms_cli.import_model_to_lmstudio("/no/such/model.gguf")
        self.assertFalse(result)

    def test_returns_true_on_successful_import(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = ("imported ok", "")
            mock_proc.stdin = MagicMock()

            with patch("lms_cli.subprocess.Popen", return_value=mock_proc):
                result = lms_cli.import_model_to_lmstudio(tmp, "testcreator", "mymodel")
            self.assertTrue(result)
        finally:
            os.unlink(tmp)

    def test_returns_false_on_nonzero_return(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = ("", "error message")
            mock_proc.stdin = MagicMock()

            with patch("lms_cli.subprocess.Popen", return_value=mock_proc):
                result = lms_cli.import_model_to_lmstudio(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)

    def test_returns_false_on_timeout(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            mock_proc = MagicMock()
            mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
                cmd="lms", timeout=300
            )
            mock_proc.stdin = MagicMock()

            with patch("lms_cli.subprocess.Popen", return_value=mock_proc):
                result = lms_cli.import_model_to_lmstudio(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)

    def test_returns_false_on_exception(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            with patch("lms_cli.subprocess.Popen", side_effect=RuntimeError("unexpected")):
                result = lms_cli.import_model_to_lmstudio(tmp)
            self.assertFalse(result)
        finally:
            os.unlink(tmp)

    def test_model_name_defaults_to_stem(self):
        """When model_name is None, it should be derived from the filename stem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = os.path.join(tmpdir, "phi3-mini.gguf")
            with open(tmp, "wb") as f:
                f.write(b"GGUF" + b"\x00" * 100)
            try:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = ("ok", "")
                mock_proc.stdin = MagicMock()

                with patch("lms_cli.subprocess.Popen", return_value=mock_proc) as mock_popen:
                    lms_cli.import_model_to_lmstudio(tmp, model_name=None)
                    # Verify lms import was called with the file path
                    args = mock_popen.call_args[0][0]
                    self.assertIn("import", args)
                    self.assertIn(tmp, args)
            finally:
                os.unlink(tmp)

    def test_copy_mode_adds_copy_flag(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.stdin = MagicMock()

            with patch("lms_cli.subprocess.Popen", return_value=mock_proc) as mock_popen:
                lms_cli.import_model_to_lmstudio(tmp, copy_mode=True)
                args = mock_popen.call_args[0][0]
                self.assertIn("--copy", args)
        finally:
            os.unlink(tmp)

    def test_move_mode_omits_copy_flag(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
            f.write(b"GGUF" + b"\x00" * 100)
            tmp = f.name
        try:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.stdin = MagicMock()

            with patch("lms_cli.subprocess.Popen", return_value=mock_proc) as mock_popen:
                lms_cli.import_model_to_lmstudio(tmp, copy_mode=False)
                args = mock_popen.call_args[0][0]
                self.assertNotIn("--copy", args)
        finally:
            os.unlink(tmp)


class TestListLmstudioModels(unittest.TestCase):
    """Tests for lms_cli.list_lmstudio_models."""

    def test_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "phi3-mini\n"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertTrue(lms_cli.list_lmstudio_models())

    def test_returns_false_on_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "err"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertFalse(lms_cli.list_lmstudio_models())

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=Exception("fail")):
            self.assertFalse(lms_cli.list_lmstudio_models())


class TestLoadModelInLmstudio(unittest.TestCase):
    """Tests for lms_cli.load_model_in_lmstudio."""

    def test_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "loaded"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertTrue(lms_cli.load_model_in_lmstudio("phi3-mini"))

    def test_returns_false_on_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not found"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertFalse(lms_cli.load_model_in_lmstudio("phi3-mini"))

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=Exception("broken")):
            self.assertFalse(lms_cli.load_model_in_lmstudio("phi3-mini"))


class TestGetLmstudioStatus(unittest.TestCase):
    """Tests for lms_cli.get_lmstudio_status."""

    def test_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "running"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertTrue(lms_cli.get_lmstudio_status())

    def test_returns_false_on_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertFalse(lms_cli.get_lmstudio_status())

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=Exception("down")):
            self.assertFalse(lms_cli.get_lmstudio_status())


class TestListLoadedModels(unittest.TestCase):
    """Tests for lms_cli.list_loaded_models."""

    def test_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "phi3-mini (loaded)"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertTrue(lms_cli.list_loaded_models())

    def test_returns_false_when_command_fails(self):
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stderr = "not supported"
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertFalse(lms_cli.list_loaded_models())

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=RuntimeError("crash")):
            self.assertFalse(lms_cli.list_loaded_models())


class TestAutoImportProjectModels(unittest.TestCase):
    """Tests for lms_cli.auto_import_project_models."""

    def test_returns_false_when_no_gguf_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # models/ dir exists but is empty
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            # Change working directory so auto_import finds the dir
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = lms_cli.auto_import_project_models()
                self.assertFalse(result)
            finally:
                os.chdir(original_cwd)

    def test_returns_true_when_all_imports_succeed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "model_a.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
            (models_dir / "model_b.gguf").write_bytes(b"GGUF" + b"\x00" * 100)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("lms_cli.import_model_to_lmstudio", return_value=True):
                    result = lms_cli.auto_import_project_models()
                self.assertTrue(result)
            finally:
                os.chdir(original_cwd)

    def test_returns_true_when_at_least_one_import_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "good.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
            (models_dir / "bad.gguf").write_bytes(b"GGUF" + b"\x00" * 100)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                side_effects = [True, False]
                with patch(
                    "lms_cli.import_model_to_lmstudio", side_effect=side_effects
                ):
                    result = lms_cli.auto_import_project_models()
                self.assertTrue(result)
            finally:
                os.chdir(original_cwd)

    def test_returns_false_when_all_imports_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "model.gguf").write_bytes(b"GGUF" + b"\x00" * 100)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("lms_cli.import_model_to_lmstudio", return_value=False):
                    result = lms_cli.auto_import_project_models()
                self.assertFalse(result)
            finally:
                os.chdir(original_cwd)


class TestShowLmsHelp(unittest.TestCase):
    """Tests for lms_cli.show_lms_help."""

    def test_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "Usage: lms ..."
        with patch("lms_cli.subprocess.run", return_value=mock_result):
            self.assertTrue(lms_cli.show_lms_help())

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=Exception("no lms")):
            self.assertFalse(lms_cli.show_lms_help())


class TestStartChatSession(unittest.TestCase):
    """Tests for lms_cli.start_chat_session."""

    def test_returns_true_without_model(self):
        with patch("lms_cli.subprocess.run") as mock_run:
            result = lms_cli.start_chat_session()
            self.assertTrue(result)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd, ["lms", "chat"])

    def test_returns_true_with_model_identifier(self):
        with patch("lms_cli.subprocess.run") as mock_run:
            result = lms_cli.start_chat_session("phi3-mini")
            self.assertTrue(result)
            cmd = mock_run.call_args[0][0]
            self.assertIn("phi3-mini", cmd)

    def test_returns_false_on_exception(self):
        with patch("lms_cli.subprocess.run", side_effect=Exception("failed")):
            result = lms_cli.start_chat_session()
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
