"""Smoke tests for scripts/bench_jax_vs_torch.py.

Verifies the benchmark script imports correctly and can run a single
quick configuration end-to-end without crashing.  Not marked slow — the
quick path (1 epoch, 10 ages, 1 repeat) completes in seconds once JAX is
compiled.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.bench_jax_vs_torch import app


class TestBenchSmoke:
    """Smoke-test the JAX/torch benchmark CLI."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "JAX" in result.output or "benchmark" in result.output.lower()

    def test_quick_run_cpu_only(self, tmp_path: Path) -> None:
        out = tmp_path / "bench.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--quick",
                "--n-repeats",
                "1",
                "--no-gpu",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists(), "Output JSON was not written"

        data = json.loads(out.read_text())
        assert "cpu_results" in data
        assert "simulation_baseline" in data
        assert len(data["cpu_results"]) > 0

        row = data["cpu_results"][0]
        for key in (
            "epochs",
            "n_ages",
            "jax_ll_cold_s",
            "jax_ll_warm_s",
            "torch_ll_s",
            "speedup_ll",
        ):
            assert key in row, f"Missing key '{key}' in result row"

    def test_result_schema(self, tmp_path: Path) -> None:
        out = tmp_path / "bench2.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--quick",
                "--n-repeats",
                "1",
                "--no-gpu",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out.read_text())

        # Platform block
        assert "platform" in data
        assert "jax_version" in data["platform"]
        assert "torch_version" in data["platform"]

        # Simulation baseline: one row per epoch config
        sim = data["simulation_baseline"]
        assert len(sim) == 3  # 1-epoch, 2-epoch, 6-epoch
        for row in sim:
            assert "epochs" in row
            assert "sim_mean_s" in row

        # All cpu_results rows have positive timings
        for row in data["cpu_results"]:
            assert row["jax_ll_warm_s"] > 0
            assert row["torch_ll_s"] > 0
