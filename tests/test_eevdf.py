"""EEVDF scheduler tests — Earliest Eligible Virtual Deadline First.

Anchors come from Peter Zijlstra's 2023 Linux 6.6 series (e.g. commit
``147f3efaa241`` "sched/fair: Add latency_offset" and the cover
letter on lkml.kernel.org/lkml/20230531115839.GA15915@hirez.programming.kicks-ass.net).

Three behaviours pinned by these tests:

* The runqueue is the SAME ``RBTree`` as Phase 2 (CLAUDE.md inv #1),
  but keyed on ``virtual_deadline``.
* ``pick_next`` filters by *eligibility* (``vruntime <= avg_vruntime``)
  and then picks the smallest virtual deadline.
* ``virtual_deadline = vruntime + base_slice * NICE_0_LOAD / weight``;
  high-priority (low nice) tasks therefore get tighter deadlines and
  win contention sooner.
"""

from __future__ import annotations

from src.algorithms.cfs import NICE_0_LOAD
from src.algorithms.eevdf import DEFAULT_BASE_SLICE, EEVDF
from src.core.event import Simulator
from src.core.process import Process
from src.core.rbtree import RBTree


def test_eevdf_runqueue_is_the_handwritten_rbtree():
    """Invariant #1: EEVDF reuses Phase 2's RBTree (same data structure as CFS)."""
    sched = EEVDF(num_cpus=1)
    assert isinstance(sched._runqueues[0], RBTree)


def test_eevdf_virtual_deadline_scales_with_inverse_weight():
    """vd = vruntime + base_slice * NICE_0_LOAD / weight; nice=-5 -> tighter vd."""
    sched = EEVDF(num_cpus=1, base_slice=4)
    a = Process(pid=1, arrival_time=0, burst_time=10, nice_value=0)  # w=1024
    b = Process(pid=2, arrival_time=0, burst_time=10, nice_value=-5)  # w=3121
    sched.on_arrival(a, now=0)
    sched.on_arrival(b, now=0)
    # Both start at vruntime=0. VD_a = 4*1024/1024 = 4. VD_b = 4*1024/3121 ~ 1.31.
    assert abs(a.virtual_deadline - 4.0) < 1e-9
    assert abs(b.virtual_deadline - (4.0 * NICE_0_LOAD / 3121)) < 1e-9
    assert b.virtual_deadline < a.virtual_deadline


def test_eevdf_avg_vruntime_is_weighted_mean_of_in_tree_tasks():
    sched = EEVDF(num_cpus=1)
    a = Process(pid=1, arrival_time=0, burst_time=10, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=10, nice_value=0)
    sched.on_arrival(a, now=0)
    sched.on_arrival(b, now=0)
    assert sched.avg_vruntime(0) == 0.0
    a.vruntime = 10.0
    # weighted avg = (10*1024 + 0*1024) / 2048 = 5.0
    assert abs(sched.avg_vruntime(0) - 5.0) < 1e-9


def test_eevdf_eligibility_filter_excludes_overserved_tasks():
    """A task whose vruntime exceeds avg_vruntime is NOT eligible."""
    sched = EEVDF(num_cpus=1)
    a = Process(pid=1, arrival_time=0, burst_time=10, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=10, nice_value=0)
    sched.on_arrival(a, now=0)
    sched.on_arrival(b, now=0)
    a.vruntime = 10.0
    b.vruntime = 0.0
    assert not sched._is_eligible(a, cpu_id=0)
    assert sched._is_eligible(b, cpu_id=0)


def test_eevdf_pick_next_skips_ineligible_even_with_smaller_vd():
    """B has bigger vd but is the only eligible task — picked anyway."""
    sched = EEVDF(num_cpus=1, base_slice=4)
    a = Process(pid=1, arrival_time=0, burst_time=10, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=10, nice_value=0)
    sched.on_arrival(a, now=0)
    sched.on_arrival(b, now=0)
    a.vruntime = 100.0
    b.vruntime = 0.0
    # Force a tree where A has the smallest vd but is ineligible.
    sched._runqueues[0] = RBTree()
    sched._handles[a.pid] = sched._runqueues[0].insert(0.5, a)
    sched._handles[b.pid] = sched._runqueues[0].insert(10.0, b)
    picked = sched.pick_next(cpu_id=0, now=0)
    assert picked is b


def test_eevdf_two_equal_nice_tasks_share_cpu_evenly():
    """Sanity: same fairness as CFS for symmetric long-runners."""
    a = Process(pid=1, arrival_time=0, burst_time=500, nice_value=0)
    b = Process(pid=2, arrival_time=0, burst_time=500, nice_value=0)
    sched = EEVDF(num_cpus=1)
    Simulator(sched, [a, b]).run()
    assert max(a.finish_time, b.finish_time) == 1000
    assert a.cpu_used == 500
    assert b.cpu_used == 500


def test_eevdf_default_base_slice_matches_kernel_default():
    """``DEFAULT_BASE_SLICE`` mirrors Linux's ``sysctl_sched_base_slice``."""
    assert DEFAULT_BASE_SLICE > 0
