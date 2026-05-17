"""End-to-end Simulator integration tests using stub schedulers.

These tests exercise the entire event-loop without depending on any algorithm
in :mod:`src.algorithms` (Phase 3+). The stubs live in :mod:`tests.conftest`.
"""

from __future__ import annotations

import pytest

from src.core.cpu import LoadBalancer
from src.core.event import Simulator
from src.core.process import IOPattern, Process, ProcessState
from tests.conftest import FIFOStubScheduler, QuantumStubScheduler


def test_single_process_runs_to_completion() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=5)
    sim = Simulator(FIFOStubScheduler(num_cpus=1), [p])
    summary = sim.run()

    assert p.state == ProcessState.TERMINATED
    assert p.cpu_used == 5
    assert p.start_time == 0
    assert p.finish_time == 5
    assert p.response_time == 0
    assert p.turnaround_time == 5
    assert p.wait_time == 0
    assert summary["completed"] == 1
    assert summary["total_ticks"] == 5
    assert summary["cpu_utilization_aggregate"] == pytest.approx(1.0)


def test_two_processes_fcfs_order() -> None:
    p1 = Process(pid=1, arrival_time=0, burst_time=5)
    p2 = Process(pid=2, arrival_time=2, burst_time=3)
    sim = Simulator(FIFOStubScheduler(num_cpus=1), [p1, p2])
    sim.run()

    # P1 runs 0..5; P2 waited from 2 → 5 (3 ticks) then runs 5..8
    assert p1.finish_time == 5
    assert p2.start_time == 5
    assert p2.finish_time == 8
    assert p2.wait_time == 3


def test_idle_period_fast_forwards() -> None:
    p = Process(pid=1, arrival_time=100, burst_time=2)
    sim = Simulator(FIFOStubScheduler(num_cpus=1), [p])
    summary = sim.run()

    assert sim.now == 102
    assert summary["total_ticks"] == 102
    assert summary["cpu_utilization_aggregate"] == pytest.approx(2 / 102)


def test_io_blocking_round_trip() -> None:
    """Process runs 2, blocks 3, runs 2, exits → total CPU 4, wall-clock 7."""
    p = Process(
        pid=1,
        arrival_time=0,
        burst_time=4,
        io_pattern=IOPattern(cpu_burst=2, io_burst=3),
    )
    sim = Simulator(FIFOStubScheduler(num_cpus=1), [p])
    summary = sim.run()

    assert p.state == ProcessState.TERMINATED
    assert p.cpu_used == 4
    assert p.finish_time == 7
    assert summary["cpu_utilization_aggregate"] == pytest.approx(4 / 7)


def test_quantum_preemption_with_two_tasks_alternates() -> None:
    p1 = Process(pid=1, arrival_time=0, burst_time=6)
    p2 = Process(pid=2, arrival_time=0, burst_time=6)
    sim = Simulator(QuantumStubScheduler(num_cpus=1, quantum=2), [p1, p2])
    summary = sim.run()

    assert p1.state == ProcessState.TERMINATED
    assert p2.state == ProcessState.TERMINATED
    assert summary["total_ticks"] == 12
    # With quantum=2 and 6-tick bursts each, both tasks get preempted multiple times
    assert summary["context_switches_total"] >= 5


def test_two_cpus_run_in_parallel() -> None:
    p1 = Process(pid=1, arrival_time=0, burst_time=5)
    p2 = Process(pid=2, arrival_time=0, burst_time=5)

    sched = FIFOStubScheduler(num_cpus=2)
    sched._queues[0].append(p1)
    sched._queues[1].append(p2)

    sim = Simulator(sched, [])
    sim.metrics.register(p1)
    sim.metrics.register(p2)
    p1.enter_ready(0)
    p2.enter_ready(0)

    summary = sim.run()
    assert p1.finish_time == 5
    assert p2.finish_time == 5
    assert summary["completed"] == 2
    assert summary["cpu_utilization_per_cpu"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_load_balancer_pulls_work_to_idle_cpu() -> None:
    procs = [Process(pid=i, arrival_time=0, burst_time=4) for i in range(4)]

    sched = FIFOStubScheduler(num_cpus=2)
    lb = LoadBalancer(sched)
    sim = Simulator(sched, procs, load_balancer=lb)
    summary = sim.run()

    assert all(p.state == ProcessState.TERMINATED for p in procs)
    # 4 procs x 4-tick burst = 16 ticks of work; lower bound on 2 CPUs is 8.
    assert summary["total_ticks"] >= 8
    assert summary["migrations"] >= 1
    assert summary["cpu_utilization_per_cpu"][1] > 0.3


def test_num_cpus_mismatch_rejected() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    with pytest.raises(ValueError, match="num_cpus mismatch"):
        Simulator(sched, [], num_cpus=1)


def test_max_time_positive() -> None:
    sched = FIFOStubScheduler(num_cpus=1)
    with pytest.raises(ValueError, match="max_time"):
        Simulator(sched, [], max_time=0)


def test_max_time_caps_runaway_simulation() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=1000)
    sim = Simulator(FIFOStubScheduler(num_cpus=1), [p], max_time=10)
    summary = sim.run()
    assert summary["total_ticks"] == 10
    assert p.state != ProcessState.TERMINATED
