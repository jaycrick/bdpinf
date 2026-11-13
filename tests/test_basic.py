"""Tests for bdpinf package basics."""


class TestBasic:
    """Test basic package-level functionality."""

    def test_package_import(self) -> None:
        """Test that the package can be imported."""
        import bdpinf

        assert bdpinf.__version__ == "0.4.2"
