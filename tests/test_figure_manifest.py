"""Tests for figure manifests and pruning."""

import sys
from collections.abc import Iterator
from pathlib import Path

import altair as alt
import pandas as pd
import pytest

from bdpinf.visualization import figure_manifest as fm
from bdpinf.visualization import save_chart
from bdpinf.visualization.figure_manifest import (
    current_source,
    main,
    prune_figures,
    record_figure,
    source_slug,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "QUARTO_DOCUMENT_FILE",
        "QUARTO_DOCUMENT_PATH",
        "QUARTO_PROJECT_ROOT",
        "QUARTO_PROJECT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    fm._started.clear()
    yield
    fm._started.clear()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _manifest(directory: Path, slug: str, source: str, names: list[str]) -> Path:
    path = directory / ".figure_manifest" / f"{slug}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# source: {source}", *names]) + "\n")
    return path


class TestCurrentSource:
    def test_quarto_document(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc_dir = tmp_path / "qmd" / "sub"
        doc_dir.mkdir(parents=True)
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "nb.qmd")
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(doc_dir))
        monkeypatch.setenv("QUARTO_PROJECT_ROOT", str(tmp_path))
        assert current_source() == "qmd/sub/nb.qmd"

    def test_quarto_project_dir_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "nb.qmd")
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(tmp_path))
        monkeypatch.setenv("QUARTO_PROJECT_DIR", str(tmp_path))
        assert current_source() == "nb.qmd"

    def test_quarto_document_outside_project_is_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "nb.qmd")
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(tmp_path))
        monkeypatch.setenv("QUARTO_PROJECT_ROOT", str(tmp_path / "elsewhere"))
        assert current_source() == (tmp_path / "nb.qmd").resolve().as_posix()

    def test_relative_document_without_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "nb.qmd")
        assert current_source() == (tmp_path / "nb.qmd").resolve().as_posix()

    def test_script(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        script = _touch(tmp_path / "run_me.py")
        monkeypatch.setattr(sys, "argv", [str(script)])
        assert current_source() == script.resolve().as_posix()

    def test_script_inside_project_is_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = Path(__file__).resolve()
        monkeypatch.setattr(sys, "argv", [str(script)])
        assert current_source() == "tests/test_figure_manifest.py"

    def test_interactive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["not_a_file"])
        assert current_source() == "interactive"
        monkeypatch.setattr(sys, "argv", [])
        assert current_source() == "interactive"


class TestSourceSlug:
    def test_separators_replaced(self) -> None:
        assert source_slug("qmd/sub/nb.qmd") == "qmd__sub__nb.qmd"
        assert source_slug("/abs/dir/s.py") == "abs__dir__s.py"

    def test_unsafe_characters(self) -> None:
        assert "/" not in source_slug("a b:c/d")

    def test_empty(self) -> None:
        assert source_slug("") == "interactive"


class TestRecordFigure:
    def _setup(self, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
        monkeypatch.setattr(fm, "current_source", lambda: source)

    def test_first_write_truncates_then_appends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(monkeypatch, "a/nb.qmd")
        manifest = tmp_path / ".figure_manifest" / "a__nb.qmd.txt"
        manifest.parent.mkdir()
        manifest.write_text("# source: a/nb.qmd\nold.pdf\n")

        record_figure(tmp_path / "one.pdf")
        record_figure(tmp_path / "two.pdf")
        record_figure(tmp_path / "one.pdf")
        assert manifest.read_text().splitlines() == [
            "# source: a/nb.qmd",
            "one.pdf",
            "two.pdf",
        ]

    def test_new_process_truncates_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(monkeypatch, "nb.qmd")
        record_figure(tmp_path / "one.pdf")
        fm._started.clear()
        record_figure(tmp_path / "two.pdf")
        manifest = tmp_path / ".figure_manifest" / "nb.qmd.txt"
        assert manifest.read_text().splitlines() == ["# source: nb.qmd", "two.pdf"]

    def test_sources_and_directories_are_independent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(monkeypatch, "a.qmd")
        record_figure(tmp_path / "x.pdf")
        record_figure(tmp_path / "sub" / "y.pdf")
        self._setup(monkeypatch, "b.qmd")
        record_figure(tmp_path / "z.pdf")
        names = sorted(p.name for p in (tmp_path / ".figure_manifest").iterdir())
        assert names == ["a.qmd.txt", "b.qmd.txt"]
        assert (tmp_path / "sub" / ".figure_manifest" / "a.qmd.txt").exists()

    def test_deleted_manifest_is_recreated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(monkeypatch, "nb.qmd")
        record_figure(tmp_path / "one.pdf")
        (tmp_path / ".figure_manifest" / "nb.qmd.txt").unlink()
        record_figure(tmp_path / "two.pdf")
        text = (tmp_path / ".figure_manifest" / "nb.qmd.txt").read_text()
        assert text.splitlines() == ["# source: nb.qmd", "two.pdf"]


class TestSaveChartRecords:
    def test_save_chart_writes_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fm, "current_source", lambda: "nb.qmd")
        chart = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [1, 2]}))
            .mark_line()
            .encode(x="x:Q", y="y:Q")
        )
        path = save_chart(chart, "fig_a", output_dir=tmp_path)
        assert path.exists()
        manifest = tmp_path / ".figure_manifest" / "nb.qmd.txt"
        assert manifest.read_text().splitlines() == ["# source: nb.qmd", "fig_a.pdf"]


class TestPruneFigures:
    def _tree(self, tmp_path: Path) -> tuple[Path, Path]:
        project = tmp_path / "project"
        (project / "nb").mkdir(parents=True)
        (project / "nb" / "live.qmd").write_text("")
        figs = project / "figures"
        keep = _touch(figs / "managed" / "keep.pdf")
        stale = _touch(figs / "managed" / "stale.png")
        _manifest(figs / "managed", "nb__live.qmd", "nb/live.qmd", ["keep.pdf"])
        assert keep.exists()
        assert stale.exists()
        return project, figs

    def test_dry_run_reports_without_deleting(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        report = prune_figures(figs, project_root=project)
        assert report.stale == [figs / "managed" / "stale.png"]
        assert report.unmanaged == []
        assert report.orphan_manifests == []
        assert report.deleted == []
        assert (figs / "managed" / "stale.png").exists()

    def test_apply_deletes_stale_only(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        report = prune_figures(figs, project_root=project, apply=True)
        assert report.deleted == [figs / "managed" / "stale.png"]
        assert not (figs / "managed" / "stale.png").exists()
        assert (figs / "managed" / "keep.pdf").exists()
        assert (figs / "managed" / ".figure_manifest" / "nb__live.qmd.txt").exists()

    def test_non_figure_files_are_ignored(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        _touch(figs / "managed" / "notes.txt")
        report = prune_figures(figs, project_root=project)
        assert report.stale == [figs / "managed" / "stale.png"]

    def test_orphan_manifest_dropped_and_figures_stale(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        orphan = _manifest(
            figs / "managed", "nb__gone.qmd", "nb/gone.qmd", ["gone.svg"]
        )
        _touch(figs / "managed" / "gone.svg")
        report = prune_figures(figs, project_root=project)
        assert report.orphan_manifests == [orphan]
        assert figs / "managed" / "gone.svg" in report.stale

        report = prune_figures(figs, project_root=project, apply=True)
        assert not orphan.exists()
        assert not (figs / "managed" / "gone.svg").exists()
        assert (figs / "managed" / "keep.pdf").exists()

    def test_interactive_manifest_is_never_orphan(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        _manifest(figs / "managed", "interactive", "interactive", ["scratch.pdf"])
        _touch(figs / "managed" / "scratch.pdf")
        report = prune_figures(figs, project_root=project)
        assert report.orphan_manifests == []
        assert figs / "managed" / "scratch.pdf" not in report.stale

    def test_manifest_without_source_line_is_kept(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        path = figs / "managed" / ".figure_manifest" / "odd.txt"
        path.write_text("stale.png\n")
        report = prune_figures(figs, project_root=project)
        assert report.orphan_manifests == []
        assert report.stale == []

    def test_unmanaged_directory(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        loose = _touch(figs / "loose" / "old.pdf")

        report = prune_figures(figs, project_root=project, apply=True)
        assert report.unmanaged == [loose]
        assert loose.exists()

        report = prune_figures(
            figs, project_root=project, apply=True, include_unmanaged=True
        )
        assert loose in report.deleted
        assert not loose.exists()

    def test_nested_directories(self, tmp_path: Path) -> None:
        project, figs = self._tree(tmp_path)
        nested = _touch(figs / "managed" / "deeper" / "x.pdf")
        report = prune_figures(figs, project_root=project)
        assert report.unmanaged == [nested]


class TestCli:
    def test_dry_run_prints_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        figs = tmp_path / "figures"
        _touch(figs / "d" / "a.pdf")
        _manifest(figs / "d", "interactive", "interactive", ["b.pdf"])
        _touch(figs / "loose" / "c.pdf")

        code = main(["prune", str(figs), "--project-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert code == 0
        assert "Stale figures" in out
        assert "a.pdf" in out
        assert "Unmanaged figures" in out
        assert "Dry run" in out
        assert (figs / "d" / "a.pdf").exists()

    def test_apply(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        figs = tmp_path / "figures"
        _touch(figs / "d" / "a.pdf")
        _manifest(figs / "d", "interactive", "interactive", ["b.pdf"])
        _touch(figs / "loose" / "c.pdf")

        code = main(
            [
                "prune",
                str(figs),
                "--project-root",
                str(tmp_path),
                "--apply",
                "--include-unmanaged",
            ]
        )
        assert code == 0
        assert "Deleted: 2" in capsys.readouterr().out
        assert not (figs / "d" / "a.pdf").exists()
        assert not (figs / "loose" / "c.pdf").exists()

    def test_default_project_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["prune", str(tmp_path)]) == 0
        assert "Stale figures" in capsys.readouterr().out
