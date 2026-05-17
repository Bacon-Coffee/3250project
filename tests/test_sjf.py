"""Shortest-Job-First (non-preemptive) and SRTF (preemptive) tests.

Two textbook scenarios anchor expected behaviour:

* **SJF undoes the FCFS convoy** — same 1*100 + 3*10 workload as
  :mod:`tests.test_fcfs`, but the shorts now finish first; average
  turnaround drops from 115 to 47.5.
* **SRTF preempts the long task** when a shorter one arrives mid-run.
"""

from __future__ import annotations

from src.algorithms.sjf import SJF
from src.core.event import Simulator
from src.core.process import Process


def _avg(values):
    seq = list(values)
    return sum(seq) / len(seq)


def test_sjf_non_preemptive_breaks_the_fcfs_convoy():
    long_job = Process(pid=1, arrival_time=0, burst_time=100)
    s1 = Process(pid=2, arrival_time=0, burst_time=10)
    s2 = Process(pid=3, arrival_time=0, burst_time=10)
    s3 = Process(pid=4, arrival_time=0, burst_time=10)
    sched = SJF(num_cpus=1, preemptive=False)
    Simulator(sched, [long_job, s1, s2, s3]).run()
    # SJF reorders to s1 -> s2 -> s3 -> long_job (tie-break by pid).
    assert s1.finish_time == 10
    assert s2.finish_time == 20
    assert s3.finish_time == 30
    assert long_job.finish_time == 130
    avg_turnaround = _avg(p.turnaround_time for p in [long_job, s1, s2, s3])
    assert avg_turnaround == 47.5


def test_sjf_non_preemptive_does_not_preempt_mid_run():
    """A short task arriving after a long one started must still wait."""
    long_job = Process(pid=1, arrival_time=0, burst_time=20)
    short = Process(pid=2, arrival_time=5, burst_time=2)
    sched = SJF(num_cpus=1, preemptive=False)
    Simulator(sched, [long_job, short]).run()
    assert long_job.finish_time == 20
    assert short.start_time == 20
    assert short.finish_time == 22


def test_srtf_preempts_when_shorter_task_arrives():
    """SRTF (preemptive SJF): a 2-tick task arriving at t=5 evicts the long job."""
    long_job = Process(pid=1, arrival_time=0, burst_time=100)
    short = Process(pid=2, arrival_time=5, burst_time=2)
    sched = SJF(num_cpus=1, preemptive=True)
    Simulator(sched, [long_job, short]).run()
    # long_job ran for ticks 0..6, then preempted; short runs and finishes 2 ticks later.
    assert short.start_time == 6
    assert short.finish_time == 8
    # long_job had cpu_used=6 at preemption; it needs 100 - 6 = 94 more ticks.
    assert long_job.finish_time == short.finish_time + (long_job.burst_time - 6)


def test_srtf_does_not_preempt_for_equal_or_longer_arrivals():
    """An arriving task with `remaining >= running.remaining` must NOT preempt."""
    long_job = Process(pid=1, arrival_time=0, burst_time=10)
    equal = Process(pid=2, arrival_time=3, burst_time=10)
    sched = SJF(num_cpus=1, preemptive=True)
    Simulator(sched, [long_job, equal]).run()
    # long_job has remaining=7 at t=3; equal arrives with remaining=10 — no preempt.
    assert long_job.finish_time == 10
    assert equal.start_time == 10
