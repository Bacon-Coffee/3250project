"""Round Robin scheduling tests.

Goal: response_time is bounded by ``(n-1) * quantum`` no matter how
long the bursts are. This is the property that makes RR responsive to
interactive workloads — at the cost of more context switches than FCFS.
"""

from __future__ import annotations

import pytest

from src.algorithms.round_robin import RoundRobin
from src.core.event import Simulator
from src.core.process import Process


def test_rr_invalid_quantum_rejected():
    with pytest.raises(ValueError):
        RoundRobin(num_cpus=1, quantum=0)
    with pytest.raises(ValueError):
        RoundRobin(num_cpus=1, quantum=-1)


def test_rr_quantum_2_interleaves_three_equal_tasks():
    """3 tasks of burst 5, quantum 2 -> strict A/B/C/A/B/C/... interleave."""
    a = Process(pid=1, arrival_time=0, burst_time=5)
    b = Process(pid=2, arrival_time=0, burst_time=5)
    c = Process(pid=3, arrival_time=0, burst_time=5)
    sched = RoundRobin(num_cpus=1, quantum=2)
    Simulator(sched, [a, b, c]).run()
    # Response times bounded by (n-1)*quantum = 4.
    assert a.start_time == 0
    assert b.start_time == 2
    assert c.start_time == 4
    # Predictable finish-time pattern from the strict 2-tick interleave.
    assert a.finish_time == 13
    assert b.finish_time == 14
    assert c.finish_time == 15


def test_rr_task_finishing_within_quantum_does_not_get_preempted():
    """A task whose remaining < quantum should finish in one slice."""
    a = Process(pid=1, arrival_time=0, burst_time=2)
    b = Process(pid=2, arrival_time=0, burst_time=2)
    sched = RoundRobin(num_cpus=1, quantum=4)
    Simulator(sched, [a, b]).run()
    assert a.finish_time == 2
    assert b.finish_time == 4


def test_rr_response_time_bounded_by_n_minus_one_times_quantum():
    """Adding more tasks must still keep the last one's response_time <= (n-1)*q."""
    quantum = 3
    tasks = [Process(pid=i, arrival_time=0, burst_time=10) for i in range(1, 6)]
    sched = RoundRobin(num_cpus=1, quantum=quantum)
    Simulator(sched, tasks).run()
    n = len(tasks)
    for p in tasks:
        assert p.response_time <= (n - 1) * quantum, p


def test_rr_quantum_resets_when_task_is_redispatched():
    """After a task is preempted and later picked again, its slice starts fresh."""
    a = Process(pid=1, arrival_time=0, burst_time=10)
    b = Process(pid=2, arrival_time=0, burst_time=4)
    sched = RoundRobin(num_cpus=1, quantum=2)
    Simulator(sched, [a, b]).run()
    # b finishes at t=8 (a:0..2, b:2..4, a:4..6, b:6..8 done).
    # a then runs uninterrupted for its remaining 6 ticks: finishes at t=14.
    assert b.finish_time == 8
    assert a.finish_time == 14
