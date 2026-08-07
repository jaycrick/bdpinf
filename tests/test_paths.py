"""Tests for project-root resolution."""

from pathlib import Path

import pytest

from bdpinf.paths import find_project_root


class TestFindProjectRoot:
    """Test find_project_root's env-var and marker-search resolution."""

    def test_walks_up_to_find_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").touch()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        monkeypatch.delenv("QUARTO_PROJECT_DIR", raising=False)

        assert find_project_root() == tmp_path

    def test_env_var_used_when_marker_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "_quarto.yml").touch()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("QUARTO_PROJECT_DIR", str(tmp_path))

        assert find_project_root() == tmp_path

    def test_env_var_ignored_when_marker_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bogus = tmp_path / "bogus"
        bogus.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        (real_root / "pyproject.toml").touch()
        monkeypatch.chdir(real_root)
        monkeypatch.setenv("QUARTO_PROJECT_DIR", str(bogus))

        assert find_project_root() == real_root

    def test_falls_back_to_cwd_when_no_marker_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("QUARTO_PROJECT_DIR", raising=False)

        assert find_project_root(markers=("no-such-marker-file",)) == tmp_path

    def test_custom_env_var_name_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").touch()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("CUSTOM_ROOT", str(tmp_path))

        assert find_project_root(env_var="CUSTOM_ROOT") == tmp_path
