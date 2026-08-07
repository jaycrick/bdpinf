"""Tests for bdpinf package basics."""

import tempfile
from pathlib import Path

from bdpinf.utils import ensure_directory


class TestBasic:
    """Test basic package-level functionality."""

    def test_ensure_directory(self) -> None:
        """Test directory creation utility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_subdir" / "nested"
            ensure_directory(test_path)
            assert test_path.exists()
            assert test_path.is_dir()

    def test_package_import(self) -> None:
        """Test that the package can be imported."""
        import bdpinf

        assert bdpinf.__version__ == "0.1.0"
