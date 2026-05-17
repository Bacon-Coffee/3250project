"""FCFS (First-Come, First-Served) end-to-end behavioural tests.

The canonical example is the **convoy effect** from OSTEP ch. 7: one
long CPU-hog (burst=100) arrives just before three short tasks
(burst=10 each). Under FCFS the shorts wait the full length of the
long task, which inflates average turnaround. This is the baseline the
later algorithms (SJF, RR, CFS) are compared against.
"""

from __future__ import annotations

from src.algorithms.fcfs import FCFS
from src.core.cpu import LoadBalancer
from src.core.event import Simulator
from src.core.process import Process


def _avg(values):
    return sum(values) / len(values)


def test_fcfs_runs_processes_in_arrival_order_on_one_cpu():
    a = Process(pid=1, arrival_time=0, burst_time=5)
    b = Process(pid=2, arrival_time=1, burst_time=3)
    c = Process(pid=3, arrival_time=2, burst_time=2)
    sched = FCFS(num_cpus=1)
    Simulator(sched, [a, b, c]).run()
    # Single-CPU FCFS: completion order matches arrival order.
    assert a.finish_time == 5
    assert b.finish_time == 5 + 3
    assert c.finish_time == 5 + 3 + 2


def test_fcfs_convoy_effect_matches_textbook_numbers():
    """OSTEP-style: 1 long (100) + 3 short (10), all arrive at t=0 in order."""
    long_job = Process(pid=1, arrival_time=0, burst_time=100)
    s1 = Process(pid=2, arrival_time=0, burst_time=10)
    s2 = Process(pid=3, arrival_time=0, burst_time=10)
    s3 = Process(pid=4, arrival_time=0, burst_time=10)
    sched = FCFS(num_cpus=1)
    Simulator(sched, [long_job, s1, s2, s3]).run()
    assert long_job.finish_time == 100
    assert s1.finish_time == 110
    assert s2.finish_time == 120
    assert s3.finish_time == 130
    avg_turnaround = _avg([p.turnaround_time for p in [long_job, s1, s2, s3]])
    assert avg_turnaround == 115.0


def test_fcfs_is_non_preemptive():
    """A short task arriving mid-run waits — FCFS never preempts."""
    long_job = Process(pid=1, arrival_time=0, burst_time=20)
    interloper = Process(pid=2, arrival_time=5, burst_time=2)
    sched = FCFS(num_cpus=1)
    Simulator(sched, [long_job, interloper]).run()
    assert long_job.finish_time == 20
    assert interloper.start_time == 20  # waited for the long job
    assert interloper.finish_time == 22


def test_fcfs_distributes_to_both_cpus_via_load_balancing():
    """With 2 CPUs and the LB pulling stalled work, both CPUs should be used."""
    # All four jobs arrive at t=0; on_arrival places them on CPU 0, then the
    # idle balancer pulls half over to CPU 1.
    jobs = [Process(pid=i, arrival_time=0, burst_time=10) for i in range(1, 5)]
    sched = FCFS(num_cpus=2)
    lb = LoadBalancer(sched)
    Simulator(sched, jobs, load_balancer=lb).run()
    # With balancing every job finishes before the naive 4*10 = 40 ticks.
    finish_times = sorted(p.finish_time for p in jobs)
    assert max(finish_times) < 40
    # At least one task migrated (its last_cpu changed before completion).
    assert any(p.last_cpu == 1 for p in jobs)
