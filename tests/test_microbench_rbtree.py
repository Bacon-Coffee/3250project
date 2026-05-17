"""Smoke tests for the RBTree microbenchmark experiment.

The benchmark itself runs at much larger sizes (1k / 10k / 100k) from
its ``__main__`` entry point. Here we exercise the measurement
functions on tiny inputs to pin the public API and confirm both
back-ends are timed.
"""

from __future__ import annotations

import random

import pytest

from src.experiments.microbench_rbtree import (
    benchmark_insert,
    benchmark_pop_leftmost,
    run_benchmarks,
)


def test_benchmark_insert_returns_both_backends():
    rng = random.Random(0)
    result = benchmark_insert(n=128, rng=rng)
    assert "rbtree_ns_per_op" in result
    assert "sortedlist_ns_per_op" in result
    assert result["n"] == 128
    assert result["rbtree_ns_per_op"] > 0
    assert result["sortedlist_ns_per_op"] > 0


def test_benchmark_pop_leftmost_drains_a_fresh_tree():
    rng = random.Random(1)
    result = benchmark_pop_leftmost(n=64, rng=rng)
    assert result["rbtree_ns_per_op"] > 0
    assert result["sortedlist_ns_per_op"] > 0
    assert result["n"] == 64


def test_run_benchmarks_writes_csv_with_one_row_per_size(tmp_path):
    csv_path = tmp_path / "bench.csv"
    rows = run_benchmarks(sizes=[32, 64], seed=42, out_csv=csv_path)
    assert csv_path.exists()
    text = csv_path.read_text()
    header, *data_lines = [line for line in text.splitlines() if line.strip()]
    # 2 sizes * 2 ops = 4 in-memory rows; each is written as 2 CSV lines
    # (one per backend), so we expect 8 data rows on disk.
    assert len(rows) == 2 * 2
    assert len(data_lines) == 2 * 2 * 2
    assert "operation" in header
    # Spot-check both backends are represented for the first size.
    assert any(line.startswith("32,insert,rbtree,") for line in data_lines)
    assert any(line.startswith("32,insert,sortedlist,") for line in data_lines)


@pytest.mark.slow
def test_run_benchmarks_with_plot(tmp_path):
    csv_path = tmp_path / "bench.csv"
    png_path = tmp_path / "bench.png"
    run_benchmarks(sizes=[16, 32], seed=7, out_csv=csv_path, out_png=png_path)
    assert csv_path.exists()
    assert png_path.exists() and png_path.stat().st_size > 0
