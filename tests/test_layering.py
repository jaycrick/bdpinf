"""Test that bdpinf's internal imports respect the package's layer ordering."""

import ast
import subprocess
import sys
from pathlib import Path

import bdpinf

assert bdpinf.__file__ is not None
_SRC_ROOT = Path(bdpinf.__file__).resolve().parent

# Layer 0 is the base of the dependency stack; higher layers may import lower
# (or equal) layers, never the reverse. Package `__init__` modules are keyed
# by their dotted package name and take the layer of the submodules they
# aggregate.
_LAYERS: dict[str, int] = {
    "bdpinf.utils": 0,
    "bdpinf.paths": 0,
    "bdpinf.types": 1,
    "bdpinf.models.validation": 1,
    "bdpinf.models.math_extinction": 2,
    "bdpinf.models.math_populations": 2,
    "bdpinf.models.rate_functions": 1,
    "bdpinf.models.math_nested": 1,
    "bdpinf.models.math_age_densities": 2,
    "bdpinf.models.math_general": 2,
    "bdpinf.models.math_age_masses": 2,
    "bdpinf.models": 2,
    "bdpinf.inference.bayesian": 3,
    "bdpinf.inference.constraints": 3,
    "bdpinf.inference.fitting": 3,
    "bdpinf.inference.knots": 3,
    "bdpinf.inference.likelihood": 3,
    "bdpinf.inference.persistence": 3,
    "bdpinf.inference.rate_models": 3,
    "bdpinf.inference.selection": 3,
    "bdpinf.inference.uncertainty": 3,
    "bdpinf.inference": 3,
    "bdpinf.models.genealogy": 4,
    "bdpinf.process": 5,
    "bdpinf.simulation.age_sampling": 6,
    "bdpinf.simulation.conditioned": 6,
    "bdpinf.simulation.forward": 6,
    "bdpinf.simulation.simulator": 6,
    "bdpinf.simulation": 6,
    "bdpinf.analysis.asymptotic_consistency": 7,
    "bdpinf.analysis.bootstrap_date_density": 7,
    "bdpinf.analysis.date_density": 7,
    "bdpinf.analysis.latex_tables": 7,
    "bdpinf.analysis.multi_epoch_recovery": 7,
    "bdpinf.analysis.single_epoch": 7,
    "bdpinf.analysis.single_epoch_recovery": 7,
    "bdpinf.analysis.spike_detection": 7,
    "bdpinf.analysis.subset_fitting": 7,
    "bdpinf.analysis.validation_helpers": 7,
    "bdpinf.analysis": 7,
    "bdpinf.visualization.age_density_plots": 8,
    "bdpinf.visualization.age_fit_plot": 8,
    "bdpinf.visualization.altair_theme": 8,
    "bdpinf.visualization.bayesian_plots": 8,
    "bdpinf.visualization.correlation_plots": 8,
    "bdpinf.visualization.date_distribution_plots": 8,
    "bdpinf.visualization.figure_manifest": 8,
    "bdpinf.visualization.genealogy": 8,
    "bdpinf.visualization.plotting": 8,
    "bdpinf.visualization.splitting_plots": 8,
    "bdpinf.visualization": 8,
    "bdpinf": 9,
}


def _module_name(path: Path) -> tuple[str, bool]:
    """Return a source file's dotted module name and whether it is a package."""
    rel_parts = path.relative_to(_SRC_ROOT.parent).with_suffix("").parts
    is_package = rel_parts[-1] == "__init__"
    if is_package:
        rel_parts = rel_parts[:-1]
    return ".".join(rel_parts), is_package


def _resolve_import_from(
    node: ast.ImportFrom, current_module: str, is_package: bool
) -> str | None:
    if node.level == 0:
        return node.module
    base_parts = current_module.split(".")
    if not is_package:
        base_parts = base_parts[:-1]
    if node.level > 1:
        base_parts = base_parts[: -(node.level - 1)]
    base = ".".join(base_parts)
    return f"{base}.{node.module}" if node.module else base


def _bdpinf_import_edges() -> list[tuple[str, str]]:
    """Collect every (importer, imported) edge naming a `bdpinf.*` module.

    Walks the full AST of every source file, including nested function
    bodies, so lazy imports used to dodge a cycle at module-load time are
    still caught.
    """
    edges: list[tuple[str, str]] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        current_module, is_package = _module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                edges.extend(
                    (current_module, alias.name)
                    for alias in node.names
                    if alias.name == "bdpinf" or alias.name.startswith("bdpinf.")
                )
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_import_from(node, current_module, is_package)
                if resolved and (
                    resolved == "bdpinf" or resolved.startswith("bdpinf.")
                ):
                    edges.append((current_module, resolved))
    return edges


def _layer_of(module: str) -> int:
    if module in _LAYERS:
        return _LAYERS[module]
    raise AssertionError(
        f"{module!r} has no entry in _LAYERS; add it to tests/test_layering.py "
        "so the layering test can check its imports."
    )


class TestLayering:
    """Assert every internal `bdpinf.*` import points at the same or a lower layer."""

    def test_no_import_points_to_a_higher_layer(self) -> None:
        violations = [
            (importer, imported, _layer_of(importer), _layer_of(imported))
            for importer, imported in _bdpinf_import_edges()
            if _layer_of(importer) < _layer_of(imported)
        ]
        assert not violations, (
            "Layer inversions found (importer_layer < imported_layer):\n"
            + "\n".join(
                f"  {importer} (layer {il}) -> {imported} (layer jl={jl})"
                for importer, imported, il, jl in violations
            )
        )

    def test_all_source_modules_are_classified(self) -> None:
        modules = {_module_name(p)[0] for p in _SRC_ROOT.rglob("*.py")}
        unclassified = modules - _LAYERS.keys()
        assert not unclassified, f"Modules missing from _LAYERS: {sorted(unclassified)}"


class TestCoreImportIsolation:
    """Assert the numeric core never pulls in the Altair visualization stack."""

    def test_core_import_does_not_load_altair(self) -> None:
        script = (
            "import sys\n"
            "import bdpinf\n"
            "import bdpinf.simulation\n"
            "import bdpinf.inference\n"
            "assert 'altair' not in sys.modules, sorted(sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
