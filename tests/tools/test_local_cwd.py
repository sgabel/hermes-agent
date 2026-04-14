"""Regression tests for LocalEnvironment cwd normalization (tilde expansion)."""

from pathlib import Path

from tools.environments.local import LocalEnvironment, _normalize_local_cwd


class TestNormalizeLocalCwd:
    def test_expands_tilde(self):
        assert _normalize_local_cwd("~/hermes") == str(Path.home() / "hermes")

    def test_empty_returns_process_cwd(self):
        import os
        assert _normalize_local_cwd("") == os.getcwd()

    def test_absolute_path_preserved(self):
        assert _normalize_local_cwd("/tmp") == "/tmp"


class TestLocalExecuteTildeExpansion:
    def test_execute_accepts_tilde_cwd(self):
        env = LocalEnvironment(timeout=10)
        try:
            r = env.execute("pwd", cwd="~")
            assert r["returncode"] == 0
            assert r["output"].strip() == str(Path.home())
        finally:
            env.cleanup()
