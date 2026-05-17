"""Tests for src.workloads.trace_loader (Bitbrains GWA-T-12 parser).

GWA-T-12 format: one CSV per VM, semicolon-separated, header:
    "Timestamp [ms];CPU cores;CPU capacity provisioned [MHz];
     CPU usage [MHz];CPU usage [%];Memory ...;Disk ...;Network ..."
Sample interval is 300 s (5 min).

Our mapping (CLAUDE.md project doc): one VM = one Process.
- arrival_time = (first_timestamp - global_min_timestamp) // tick_seconds
- burst_time   = sum(cpu_usage_pct / 100 * sample_interval_s) // tick_seconds
                 clipped to >= 1

These tests fabricate a tiny trace dir so we don't need the 100 MB dataset.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.workloads.trace_loader import (
    BitbrainsRow,
    load_bitbrains_trace,
    parse_bitbrains_csv,
)

SAMPLE_HEADER = (
    '"Timestamp [ms]";"CPU cores";"CPU capacity provisioned [MHz]";'
    '"CPU usage [MHz]";"CPU usage [%]";"Memory capacity provisioned [KB]";'
    '"Memory usage [KB]";"Disk read throughput [KB/s]";'
    '"Disk write throughput [KB/s]";"Network received throughput [KB/s]";'
    '"Network transmitted throughput [KB/s]"'
)


def _write_vm(path: Path, rows: list[tuple[int, float]]) -> None:
    """Write a minimal Bitbrains CSV: rows are (timestamp_ms, cpu_pct)."""
    lines = [SAMPLE_HEADER]
    for ts_ms, pct in rows:
        lines.append(f"{ts_ms};1;1000;{pct * 10};{pct};0;0;0;0;0;0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestParseSingleCsv:
    def test_parses_header_and_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "1.csv"
        _write_vm(f, [(0, 10.0), (300_000, 50.0), (600_000, 100.0)])
        rows = parse_bitbrains_csv(f)
        assert len(rows) == 3
        assert all(isinstance(r, BitbrainsRow) for r in rows)
        assert rows[0].timestamp_ms == 0
        assert rows[0].cpu_usage_pct == pytest.approx(10.0)
        assert rows[2].cpu_usage_pct == pytest.approx(100.0)

    def test_skips_empty_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "2.csv"
        content = SAMPLE_HEADER + "\n\n0;1;1000;100;10;0;0;0;0;0;0\n\n"
        f.write_text(content, encoding="utf-8")
        rows = parse_bitbrains_csv(f)
        assert len(rows) == 1


class TestLoadDirectory:
    def test_one_process_per_vm(self, tmp_path: Path) -> None:
        _write_vm(tmp_path / "1.csv", [(0, 50.0), (300_000, 50.0)])
        _write_vm(tmp_path / "2.csv", [(0, 20.0), (300_000, 20.0)])
        procs = load_bitbrains_trace(tmp_path)
        assert len(procs) == 2
        assert {p.pid for p in procs} == {0, 1}

    def test_pids_unique_and_dense(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_vm(tmp_path / f"{i}.csv", [(i * 1000, 50.0)])
        procs = load_bitbrains_trace(tmp_path)
        assert sorted(p.pid for p in procs) == [0, 1, 2, 3, 4]

    def test_arrival_time_normalized_to_zero(self, tmp_path: Path) -> None:
        _write_vm(tmp_path / "a.csv", [(1_000_000, 50.0), (1_300_000, 50.0)])
        _write_vm(tmp_path / "b.csv", [(1_300_000, 50.0), (1_600_000, 50.0)])
        procs = load_bitbrains_trace(tmp_path, tick_seconds=60)
        arrivals = sorted(p.arrival_time for p in procs)
        # Global min = 1_000_000 ms = 1000 s. Diff for b = 300 s → 5 ticks.
        assert arrivals[0] == 0
        assert arrivals[1] == 5

    def test_burst_time_proportional_to_cpu_usage(self, tmp_path: Path) -> None:
        # 100% for 1 hour (12 samples * 300 s) at tick=60 s -> 60 ticks of CPU.
        rows = [(i * 300_000, 100.0) for i in range(12)]
        _write_vm(tmp_path / "busy.csv", rows)
        # 10% for the same hour → 6 ticks.
        rows10 = [(i * 300_000, 10.0) for i in range(12)]
        _write_vm(tmp_path / "idle.csv", rows10)
        procs = sorted(load_bitbrains_trace(tmp_path, tick_seconds=60), key=lambda p: p.pid)
        bursts = sorted(p.burst_time for p in procs)
        assert abs(bursts[0] - 6) <= 1
        assert abs(bursts[1] - 60) <= 1

    def test_zero_usage_vm_gets_minimum_burst(self, tmp_path: Path) -> None:
        _write_vm(tmp_path / "ghost.csv", [(0, 0.0), (300_000, 0.0)])
        procs = load_bitbrains_trace(tmp_path)
        assert procs[0].burst_time >= 1

    def test_max_vms_limits_count(self, tmp_path: Path) -> None:
        for i in range(10):
            _write_vm(tmp_path / f"{i}.csv", [(0, 50.0)])
        procs = load_bitbrains_trace(tmp_path, max_vms=3)
        assert len(procs) == 3

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        assert load_bitbrains_trace(tmp_path) == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_bitbrains_trace(tmp_path / "nope")

    def test_arrivals_sorted_after_load(self, tmp_path: Path) -> None:
        _write_vm(tmp_path / "late.csv", [(900_000, 50.0)])
        _write_vm(tmp_path / "early.csv", [(0, 50.0)])
        _write_vm(tmp_path / "mid.csv", [(300_000, 50.0)])
        procs = load_bitbrains_trace(tmp_path)
        arrivals = [p.arrival_time for p in procs]
        assert arrivals == sorted(arrivals)


class TestScaleFactor:
    def test_scale_factor_shrinks_burst(self, tmp_path: Path) -> None:
        rows = [(i * 300_000, 100.0) for i in range(12)]
        _write_vm(tmp_path / "big.csv", rows)
        full = load_bitbrains_trace(tmp_path, tick_seconds=60, scale_factor=1.0)
        half = load_bitbrains_trace(tmp_path, tick_seconds=60, scale_factor=0.5)
        assert half[0].burst_time < full[0].burst_time

    def test_non_positive_scale_factor_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale_factor"):
            load_bitbrains_trace(tmp_path, scale_factor=0)


class TestRealisticSubset:
    def test_constructs_valid_process(self, tmp_path: Path) -> None:
        content = textwrap.dedent(f"""\
            {SAMPLE_HEADER}
            1304870400000;1;2000;400;20;1048576;524288;0;0;0;0
            1304870700000;1;2000;800;40;1048576;524288;0;0;0;0
            1304871000000;1;2000;1200;60;1048576;524288;0;0;0;0
        """)
        (tmp_path / "vm1.csv").write_text(content, encoding="utf-8")
        procs = load_bitbrains_trace(tmp_path)
        assert len(procs) == 1
        p = procs[0]
        assert p.arrival_time == 0
        assert p.burst_time >= 1
        assert p.io_pattern is None
        assert p.nice_value == 0
