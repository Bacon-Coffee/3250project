"""Tests for src.workloads.synthetic.

Phase 7 Task 7.1: Poisson arrivals + lognormal CPU bursts + configurable
I/O on/off cycles, exposed as three profiles (CPU_HEAVY / IO_HEAVY / MIXED).
The contract these tests pin down:

1. Output is ``list[Process]`` with unique pids in ``[0, n)``.
2. ``arrival_time`` is non-decreasing (so the simulator can push events FIFO).
3. ``burst_time >= 1`` and ``nice_value in [-20, 19]`` — Process invariants.
4. Same seed ⇒ identical output (reproducibility for paper figures).
5. Profile semantics:
   - CPU_HEAVY: nobody has ``io_pattern``.
   - IO_HEAVY:  everybody has ``io_pattern`` with short CPU bursts.
   - MIXED:    a substantial fraction of each (>20% of both).
6. Poisson interarrival sanity: mean interarrival ≈ 1/λ within 30% on n=2000.
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.core.process import IOPattern, Process
from src.workloads.synthetic import (
    Profile,
    WorkloadConfig,
    cpu_heavy,
    generate_workload,
    io_heavy,
    mixed,
)


def _assert_valid_process_list(procs: list[Process], n: int) -> None:
    assert len(procs) == n
    pids = [p.pid for p in procs]
    assert sorted(pids) == list(range(n))
    arrivals = [p.arrival_time for p in procs]
    assert arrivals == sorted(arrivals), "arrivals must be non-decreasing"
    for p in procs:
        assert p.burst_time >= 1
        assert -20 <= p.nice_value <= 19
        if p.io_pattern is not None:
            assert p.io_pattern.cpu_burst >= 1
            assert p.io_pattern.io_burst >= 1


class TestGenerateWorkload:
    def test_cpu_heavy_no_io(self) -> None:
        procs = cpu_heavy(n=50, seed=42)
        _assert_valid_process_list(procs, 50)
        assert all(p.io_pattern is None for p in procs)

    def test_io_heavy_all_have_io(self) -> None:
        procs = io_heavy(n=50, seed=42)
        _assert_valid_process_list(procs, 50)
        assert all(isinstance(p.io_pattern, IOPattern) for p in procs)
        ratios = [p.io_pattern.cpu_burst / p.io_pattern.io_burst for p in procs]
        assert sum(ratios) / len(ratios) < 1.0, "IO-heavy should average CPU < IO"

    def test_mixed_has_both_kinds(self) -> None:
        procs = mixed(n=200, seed=42)
        _assert_valid_process_list(procs, 200)
        with_io = sum(1 for p in procs if p.io_pattern is not None)
        without_io = len(procs) - with_io
        assert with_io > 0.2 * len(procs), "mixed should have >20% IO-bound"
        assert without_io > 0.2 * len(procs), "mixed should have >20% CPU-bound"

    def test_reproducible_with_seed(self) -> None:
        a = cpu_heavy(n=100, seed=7)
        b = cpu_heavy(n=100, seed=7)
        assert [p.arrival_time for p in a] == [p.arrival_time for p in b]
        assert [p.burst_time for p in a] == [p.burst_time for p in b]

    def test_different_seeds_differ(self) -> None:
        a = cpu_heavy(n=100, seed=1)
        b = cpu_heavy(n=100, seed=2)
        assert [p.burst_time for p in a] != [p.burst_time for p in b]


class TestPoissonStatistics:
    def test_mean_interarrival_close_to_inverse_lambda(self) -> None:
        rate = 0.5
        cfg = WorkloadConfig(
            profile=Profile.CPU_HEAVY,
            n_processes=2000,
            arrival_rate=rate,
            burst_mu=math.log(10),
            burst_sigma=0.5,
            io_probability=0.0,
            seed=123,
        )
        procs = generate_workload(cfg)
        arrivals = [p.arrival_time for p in procs]
        intervals = [b - a for a, b in itertools.pairwise(arrivals)]
        observed = sum(intervals) / len(intervals)
        expected = 1.0 / rate
        assert abs(observed - expected) / expected < 0.3


class TestWorkloadConfigValidation:
    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="arrival_rate"):
            WorkloadConfig(
                profile=Profile.CPU_HEAVY,
                n_processes=10,
                arrival_rate=-1.0,
                burst_mu=1.0,
                burst_sigma=0.5,
                io_probability=0.0,
            )

    def test_zero_processes_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_processes"):
            WorkloadConfig(
                profile=Profile.CPU_HEAVY,
                n_processes=0,
                arrival_rate=1.0,
                burst_mu=1.0,
                burst_sigma=0.5,
                io_probability=0.0,
            )

    def test_io_probability_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="io_probability"):
            WorkloadConfig(
                profile=Profile.CPU_HEAVY,
                n_processes=10,
                arrival_rate=1.0,
                burst_mu=1.0,
                burst_sigma=0.5,
                io_probability=1.5,
            )


class TestNiceDistribution:
    def test_default_all_zero_nice(self) -> None:
        procs = cpu_heavy(n=50, seed=42)
        assert all(p.nice_value == 0 for p in procs)

    def test_custom_nice_range(self) -> None:
        cfg = WorkloadConfig(
            profile=Profile.CPU_HEAVY,
            n_processes=200,
            arrival_rate=1.0,
            burst_mu=math.log(10),
            burst_sigma=0.4,
            io_probability=0.0,
            nice_range=(-5, 5),
            seed=42,
        )
        procs = generate_workload(cfg)
        nices = {p.nice_value for p in procs}
        assert nices.issubset(set(range(-5, 6)))
        assert len(nices) > 1, "should sample multiple nice values across 200 procs"
