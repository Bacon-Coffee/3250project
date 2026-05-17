"""MLFQ tests covering OSTEP ch. 8 rules.

Rules under test:
  R1.  Higher priority runs first; equal priority round-robins.
  R3.  New arrivals enter at the top level.
  R4.  A task that uses its full quantum at a level is demoted one step.
  R5b. I/O boost — a task that returns from IO is placed at the top level.
  R6.  Periodic priority boost (aging) lifts everyone to the top after S ticks.
"""

from __future__ import annotations

import pytest

from src.algorithms.mlfq import MLFQ
from src.core.event import Simulator
from src.core.process import IOPattern, Process


def test_mlfq_validates_quanta_length():
    with pytest.raises(ValueError):
        MLFQ(num_cpus=1, quanta=(1, 2))  # must be 3 levels


def test_mlfq_new_arrival_lands_at_top_level():
    sched = MLFQ(num_cpus=1)
    p = Process(pid=1, arrival_time=0, burst_time=5)
    sched.on_arrival(p, now=0)
    assert sched.current_level_of(p.pid) == 0


def test_mlfq_cpu_hog_drifts_down_to_bottom_level():
    """After running long enough to exhaust quanta at L0, L1, L2, task pins at L2."""
    a = Process(pid=1, arrival_time=0, burst_time=100)
    sched = MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=10_000)
    Simulator(sched, [a]).run()
    # After 1+2+4 = 7 quanta-exhaustions, A is at L2 and stays there.
    assert sched.current_level_of(a.pid) == 2


def test_mlfq_higher_priority_arrival_preempts_demoted_task():
    """A short task arriving at L0 must immediately preempt a CPU hog at L2."""
    a = Process(pid=1, arrival_time=0, burst_time=200)
    b = Process(pid=2, arrival_time=20, burst_time=2)
    sched = MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=10_000)
    Simulator(sched, [a, b]).run()
    # By t=20, A has been demoted to L2. B arrives at L0 -> immediate preempt
    # on the next tick boundary.
    assert b.start_time == 21
    assert b.finish_time == 23


def test_mlfq_io_boost_resets_level_on_unblock():
    """R5b: with ``io_boost=True``, a returning IO task is placed back at Q0."""
    sched = MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=10_000, io_boost=True)
    p = Process(
        pid=1,
        arrival_time=0,
        burst_time=10,
        io_pattern=IOPattern(cpu_burst=2, io_burst=1),
    )
    sched.on_arrival(p, now=0)
    # Pretend the task has been demoted all the way to L2 by prior CPU bursts.
    sched._level[p.pid] = 2
    sched.on_block(p, now=5)
    sched.on_unblock(p, now=6)
    assert sched.current_level_of(p.pid) == 0


def test_mlfq_io_boost_disabled_keeps_level_on_unblock():
    """With ``io_boost=False`` the task returns to whatever level it was at."""
    sched = MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=10_000, io_boost=False)
    p = Process(
        pid=1,
        arrival_time=0,
        burst_time=10,
        io_pattern=IOPattern(cpu_burst=2, io_burst=1),
    )
    sched.on_arrival(p, now=0)
    sched._level[p.pid] = 2
    sched.on_block(p, now=5)
    sched.on_unblock(p, now=6)
    assert sched.current_level_of(p.pid) == 2


def _run_one_slice(sched: MLFQ, p: Process, cpu: int = 0) -> None:
    """Drive the scheduler the way the simulator would: pick -> tick until
    on_tick returns True -> requeue. Direct unit-style probe."""
    picked = sched.pick_next(cpu_id=cpu, now=0)
    assert picked is p, f"expected to pick {p}, got {picked}"
    safety = 100
    while safety > 0 and not sched.on_tick(now=0, cpu_id=cpu, running=p):
        safety -= 1
    sched.requeue(p, cpu_id=cpu, now=0)


def test_mlfq_periodic_boost_promotes_demoted_task_back_to_top():
    """After boost_interval ticks, a previously-demoted CPU hog is back at L0."""
    sched = MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=30)
    a = Process(pid=1, arrival_time=0, burst_time=100)
    sched.on_arrival(a, now=0)
    # Run enough slice cycles to drift A all the way down to L2.
    for _ in range(5):
        _run_one_slice(sched, a)
    assert sched.current_level_of(a.pid) == 2
    # Fire pick_next at a time past the boost interval -> _maybe_boost triggers.
    sched.pick_next(cpu_id=0, now=30)
    assert sched.current_level_of(a.pid) == 0
