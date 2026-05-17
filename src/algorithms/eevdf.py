"""EEVDF — Earliest Eligible Virtual Deadline First.

Linux 6.6 (October 2023) replaced CFS with EEVDF as the default
scheduler. Peter Zijlstra's series (cover letter on lkml,
``20230531115839.GA15915@hirez.programming.kicks-ass.net``) reframed
fair scheduling around two ideas the paper relies on:

1. **Eligibility** — a task is eligible to run only if its accumulated
   ``vruntime`` is *not greater* than the runqueue's weighted average
   (``avg_vruntime``). Tasks that have been over-served wait until the
   average catches up to them.

2. **Virtual deadline** — among eligible tasks, the one with the
   smallest ``virtual_deadline = vruntime + base_slice * NICE_0_LOAD /
   weight`` is picked next. High-priority (low-nice) tasks therefore
   have *tighter* deadlines and pre-empt long-running siblings sooner.

CLAUDE.md invariant #1: the runqueue MUST be the same hand-written
:class:`src.core.rbtree.RBTree` used by CFS, only with a different key
(``virtual_deadline`` instead of ``vruntime``). The eligibility filter
makes ``pick_next`` an O(n) walk in this implementation — the kernel
augments the tree with ``min_ve`` per subtree to drop it to O(log n);
we trade that complexity for code that maps 1:1 onto the published
formulas. The paper Discussion calls this out explicitly.

Relevant kernel commits (anchored in the Methodology bibliography):

* ``86bfbb7ce4f6``  sched/fair: Add lag based placement
* ``147f3efaa241``  sched/fair: Add latency_offset
* ``e8f331bcc270``  sched/eevdf: Replace CFS with EEVDF default
"""

from __future__ import annotations

from src.algorithms.cfs import NICE_0_LOAD, weight_of
from src.core.process import Process
from src.core.rbtree import RBTree
from src.core.scheduler_base import SchedulerBase

DEFAULT_BASE_SLICE: int = 4
"""``sysctl_sched_base_slice`` analogue, in simulation ticks.

The kernel default is 750us; mapped to our integer-tick model with one
tick representing roughly a CFS sched_min_granularity unit, ``4`` keeps
the deadline arithmetic clean while still smaller than the typical
``sched_latency`` window (8) used by CFS in this project.
"""


class EEVDF(SchedulerBase):
    """Earliest Eligible Virtual Deadline First scheduler."""

    def __init__(self, num_cpus: int = 1, base_slice: int = DEFAULT_BASE_SLICE) -> None:
        super().__init__(num_cpus=num_cpus)
        if base_slice <= 0:
            raise ValueError(f"base_slice must be positive, got {base_slice}")
        self.base_slice = base_slice

        self._runqueues: list[RBTree] = [RBTree() for _ in range(num_cpus)]
        self._handles: dict[int, object] = {}

    # ------------------------------------------------------------------
    # Public/debug observability — analogous to /proc/sched_debug.
    # ------------------------------------------------------------------

    def avg_vruntime(self, cpu_id: int) -> float:
        """Weighted mean ``vruntime`` over the tasks currently in the tree.

        The eligibility test uses this as the threshold. We recompute
        lazily by walking the tree (O(n)); the kernel maintains it
        incrementally via a running sum.
        """
        tree = self._runqueues[cpu_id]
        weighted_sum = 0.0
        total_weight = 0
        for _, p in tree.iter_inorder():
            weighted_sum += p.vruntime * p.weight
            total_weight += p.weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Arrival / IO transitions
    # ------------------------------------------------------------------

    def on_arrival(self, process: Process, now: int) -> None:
        process.weight = weight_of(process.nice_value)
        # New tasks start at the current avg_vruntime so they neither earn
        # an unjustified credit nor a deficit (analog of CFS place_entity).
        process.vruntime = max(process.vruntime, self.avg_vruntime(0))
        self._set_deadline(process)
        self._enqueue(process, cpu_id=0)

    def on_unblock(self, process: Process, now: int) -> None:
        cpu = process.last_cpu if process.last_cpu is not None else 0
        process.vruntime = max(process.vruntime, self.avg_vruntime(cpu))
        self._set_deadline(process)
        self._enqueue(process, cpu_id=cpu)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        process.vruntime = max(process.vruntime, self.avg_vruntime(target_cpu))
        self._set_deadline(process)
        self._enqueue(process, cpu_id=target_cpu)

    def on_block(self, process: Process, now: int) -> None:
        # Nothing to clean up — simulator already removed it from CPU.
        pass

    # ------------------------------------------------------------------
    # Preemption / dispatch
    # ------------------------------------------------------------------

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        # vruntime accounting — identical to CFS.
        running.vruntime += NICE_0_LOAD / running.weight
        # The deadline is the "by when this slice must be done" boundary.
        # Once we cross it, request a new slice (yield iff someone else waits).
        if running.vruntime >= running.virtual_deadline:
            return self._runqueues[cpu_id].leftmost() is not None
        return False

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        tree = self._runqueues[cpu_id]
        if len(tree) == 0:
            return None
        # Walk in vd-order (ascending) and return the first eligible task.
        for _, p in tree.iter_inorder():
            if self._is_eligible(p, cpu_id):
                self._remove(p, cpu_id)
                return p
        # Fallback — no one eligible (can happen transiently before any task
        # has run). Pick the leftmost (smallest vd) regardless.
        _, p = tree.leftmost()
        self._remove(p, cpu_id)
        return p

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        # Preempted-but-not-blocked: request a fresh slice and re-insert.
        self._set_deadline(process)
        self._enqueue(process, cpu_id=cpu_id)

    # ------------------------------------------------------------------
    # Load balancing hooks
    # ------------------------------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        # Steal the rightmost (largest vd) — the task least likely to win
        # an EEVDF contest on its current CPU, so least disruptive to move.
        rightmost = self._runqueues[cpu_id].rightmost()
        if rightmost is None:
            return None
        return rightmost[1]

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self._remove(process, cpu_id)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._runqueues[cpu_id])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_eligible(self, process: Process, cpu_id: int) -> bool:
        return process.vruntime <= self.avg_vruntime(cpu_id) + 1e-9

    def _set_deadline(self, process: Process) -> None:
        process.virtual_deadline = (
            process.vruntime + self.base_slice * NICE_0_LOAD / process.weight
        )

    def _enqueue(self, process: Process, cpu_id: int) -> None:
        tree = self._runqueues[cpu_id]
        handle = tree.insert(process.virtual_deadline, process)
        self._handles[process.pid] = handle

    def _remove(self, process: Process, cpu_id: int) -> None:
        handle = self._handles.pop(process.pid, None)
        if handle is not None:
            self._runqueues[cpu_id].delete(handle)
