"""Tests for bdpinf package basics."""

from importlib.metadata import version


class TestBasic:
    """Test basic package-level functionality."""

    def test_package_import(self) -> None:
        """Test that the package can be imported."""
        import bdpinf

        assert bdpinf.__version__ == version("bdpinf")
