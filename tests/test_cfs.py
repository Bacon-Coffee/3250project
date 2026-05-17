"""CFS scheduler tests — fairness via nice-weighted vruntime.

Behaviour anchors come from ``kernel/sched/core.c`` (sched_prio_to_weight[])
and ``kernel/sched/fair.c`` (vruntime accounting). Three key facts the
paper Methodology section cites:

* ``NICE_0_LOAD = 1024`` — every nice=0 weight slot in the table.
* For two long-runners A (nice=0, w=1024) and B (nice=-5, w=3121), CPU
  share converges to ``w_A : w_B`` over enough ticks.
* The runqueue MUST be the hand-written red-black tree from Phase 2
  (CLAUDE.md invariant #1) — verified by checking the internal handle.
"""

from __future__ import annotations

from src.algorithms.cfs import CFS, NICE_0_LOAD, PRIO_TO_WEIGHT
from src.core.event import Simulator
from src.core.process import Process
from src.core.rbtree import RBTree


def test_prio_to_weight_table_anchored_to_kernel_values():
    """prio_to_weight[20] is the nice=0 weight; matches Linux mainline."""
    assert NICE_0_LOAD == 1024
    assert PRIO_TO_WEIGHT[20] == 1024  # nice = 0
    assert PRIO_TO_WEIGHT[15] == 3121  # nice = -5
    assert PRIO_TO_WEIGHT[39] == 15  # nice = 19
    assert PRIO_TO_WEIGHT[0] == 88761  # nice = -20
    assert len(PRIO_TO_WEIGHT) == 40


def test_cfs_runqueue_is_the_handwritten_rbtree():
    """Invariant #1: CFS uses the from-scratch RBTree, not sortedcontainers."""
    sched = CFS(num_cpus=1)
    assert isinstance(sched._runqueues[0], RBTree)


def test_cfs_two_equal_nice_tasks_share_cpu_evenly():
    """Two nice=0 long tasks should each get half the CPU over a long run."""
    a = Process(pid=1, arrival_time=0, burst_time=500, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=500, nice_value=0)
    sched = CFS(num_cpus=1)
    Simulator(sched, [a, b]).run()
    # 1000 ticks total, no idle.
    assert max(a.finish_time, b.finish_time) == 1000
    assert a.cpu_used == 500
    assert b.cpu_used == 500


def test_cfs_nice_minus_five_gets_proportionally_more_cpu():
    """In a contended window, nice=-5 accumulates CPU ~3x faster than nice=0.

    Set both bursts long enough that A does not terminate inside the
    observation window; at the moment B (nice=-5) finishes, compute the
    ratio of CPU used. Expected ~3121/1024 = 3.05.
    """
    a = Process(pid=1, arrival_time=0, burst_time=10_000, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=2_000, nice_value=-5)
    sched = CFS(num_cpus=1)
    Simulator(sched, [a, b]).run()
    assert b.finish_time < a.finish_time
    # A's cpu_used at B's finish = wall ticks B was NOT on CPU = finish - cpu_used.
    a_used_at_b_finish = b.finish_time - b.cpu_used
    ratio = b.cpu_used / max(1, a_used_at_b_finish)
    weight_ratio = PRIO_TO_WEIGHT[15] / PRIO_TO_WEIGHT[20]  # 3121 / 1024 ~ 3.05
    assert abs(ratio - weight_ratio) / weight_ratio < 0.05, (
        f"observed CPU ratio {ratio:.3f} vs expected weight ratio {weight_ratio:.3f}"
    )


def test_cfs_vruntime_advances_inversely_proportional_to_weight():
    """One tick of CPU adds 1024/weight to vruntime — pure arithmetic check."""
    sched = CFS(num_cpus=1)
    p = Process(pid=1, arrival_time=0, burst_time=10, nice_value=0)
    sched.on_arrival(p, now=0)
    assert p.weight == NICE_0_LOAD
    sched.pick_next(cpu_id=0, now=0)
    sched.on_tick(now=1, cpu_id=0, running=p)
    # nice=0 -> delta_vruntime = 1024 / 1024 = 1.0
    assert abs(p.vruntime - 1.0) < 1e-9

    sched2 = CFS(num_cpus=1)
    q = Process(pid=2, arrival_time=0, burst_time=10, nice_value=-5)
    sched2.on_arrival(q, now=0)
    sched2.pick_next(cpu_id=0, now=0)
    sched2.on_tick(now=1, cpu_id=0, running=q)
    assert abs(q.vruntime - (1024.0 / 3121.0)) < 1e-9


# ===========================================================================
# Phase 4.2 — dynamic time slice: max(sched_latency / nr_running, min_granularity)
# weighted by share of total runqueue weight.
# ===========================================================================


def test_cfs_dynamic_slice_two_equal_weight_tasks_split_sched_latency():
    sched = CFS(num_cpus=1, sched_latency=8, min_granularity=2)
    a = Process(pid=1, arrival_time=0, burst_time=100)
    b = Process(pid=2, arrival_time=0, burst_time=100)
    sched.on_arrival(a, now=0)
    sched.on_arrival(b, now=0)
    sched.pick_next(cpu_id=0, now=0)  # take A out of the tree
    # nr_running=2, period=8, slice = 8 * 1024 / 2048 = 4.
    assert sched._compute_slice(cpu_id=0, running=a) == 4


def test_cfs_dynamic_slice_floored_at_min_granularity_when_overcommitted():
    sched = CFS(num_cpus=1, sched_latency=8, min_granularity=2)
    procs = [Process(pid=i + 1, arrival_time=0, burst_time=100) for i in range(8)]
    for p in procs:
        sched.on_arrival(p, now=0)
    sched.pick_next(cpu_id=0, now=0)
    # nr_running=8 -> period = max(8, 8*2) = 16; slice = 16 * 1024 / 8192 = 2.
    assert sched._compute_slice(cpu_id=0, running=procs[0]) == 2


def test_cfs_dynamic_slice_proportional_to_weight():
    """nice=-5 gets a ~3x larger slice than nice=0 under the same nr_running."""
    # Measure slice_a with A running (B in tree).
    sched1 = CFS(num_cpus=1, sched_latency=12, min_granularity=1)
    a1 = Process(pid=1, arrival_time=0, burst_time=100, nice_value=0)
    b1 = Process(pid=2, arrival_time=0, burst_time=100, nice_value=-5)
    sched1.on_arrival(a1, now=0)
    sched1.on_arrival(b1, now=0)
    sched1.pick_next(cpu_id=0, now=0)  # tie-break: A is leftmost (inserted first)
    slice_a = sched1._compute_slice(cpu_id=0, running=a1)

    # Measure slice_b with B running (A in tree) — separate sched to isolate state.
    sched2 = CFS(num_cpus=1, sched_latency=12, min_granularity=1)
    b2 = Process(pid=11, arrival_time=0, burst_time=100, nice_value=-5)
    a2 = Process(pid=12, arrival_time=0, burst_time=100, nice_value=0)
    sched2.on_arrival(b2, now=0)  # insert B first so it wins the tie
    sched2.on_arrival(a2, now=0)
    sched2.pick_next(cpu_id=0, now=0)
    slice_b = sched2._compute_slice(cpu_id=0, running=b2)

    # Total weight = 4145; slice_a = 12*1024/4145 ~ 2.96; slice_b = 12*3121/4145 ~ 9.04.
    assert slice_a in (2, 3)
    assert slice_b in (8, 9, 10)
    assert slice_b > slice_a


# ===========================================================================
# Phase 4.3 — wake-up compensation: place_entity()
# ===========================================================================


def test_cfs_wakeup_compensation_gives_small_credit_relative_to_min_vruntime():
    """A long-asleep task wakes up with vruntime ~ min_vruntime - sched_latency/2."""
    sched = CFS(num_cpus=1, sched_latency=8)
    sched._min_vruntime[0] = 100.0
    p = Process(pid=1, arrival_time=0, burst_time=10)
    p.weight = NICE_0_LOAD
    p.vruntime = 0.0  # stale, far behind
    sched.on_unblock(p, now=50)
    # max(0, 100 - 8/2) = 96
    assert p.vruntime == 96.0


def test_cfs_wakeup_compensation_does_not_demote_already_ahead_task():
    """A task that is already AHEAD of min_vruntime must not have it lowered."""
    sched = CFS(num_cpus=1, sched_latency=8)
    sched._min_vruntime[0] = 100.0
    p = Process(pid=1, arrival_time=0, burst_time=10)
    p.weight = NICE_0_LOAD
    p.vruntime = 200.0
    sched.on_unblock(p, now=50)
    assert p.vruntime == 200.0
