"""Tests for inf-bp-texts package."""

# ruff: noqa: S101

import tempfile
from pathlib import Path

from inf_bp_texts.utils import ensure_directory


def test_ensure_directory() -> None:
    """Test directory creation utility."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_subdir" / "nested"
        ensure_directory(test_path)
        assert test_path.exists()
        assert test_path.is_dir()


def test_package_import() -> None:
    """Test that the package can be imported."""
    import inf_bp_texts

    assert inf_bp_texts.__version__ == "0.1.0"
