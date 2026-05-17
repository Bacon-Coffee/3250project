"""Completely Fair Scheduler (CFS).

Linux's default scheduler from 2007 (commit ``bf0f6f24a1ec``, Ingo
Molnar) until October 2023 (replaced by EEVDF in 6.6). This module
faithfully rebuilds the bits that matter for the paper's argument:

* **Per-CPU runqueue is the hand-written red-black tree** from
  :mod:`src.core.rbtree` (CLAUDE.md invariant #1), keyed on
  ``vruntime``. Leftmost == next task; ``pick_next`` is O(1) on the
  cache and O(log n) on rebalance.

* **prio_to_weight[]** is copied verbatim from
  ``kernel/sched/core.c`` (40 entries, nice -20..19). nice=0 maps to
  ``NICE_0_LOAD = 1024``. The formula::

      vruntime += delta_exec * NICE_0_LOAD / weight

  is what makes higher-priority (lower-nice) tasks accumulate
  ``vruntime`` more slowly and therefore stay leftmost in the tree
  more often.

* **min_vruntime** monotonically tracks the smallest ``vruntime`` ever
  observed on the runqueue; new arrivals are placed at this baseline
  so they neither steal CPU from existing tasks nor are forever locked
  out by an accumulated deficit.

Phase 4.1 implements weight bookkeeping, vruntime accounting and a
*coarse* preemption rule (``min_granularity``-bounded). Phase 4.2 will
replace the coarse rule with the kernel's dynamic
``max(sched_latency / nr_running, min_granularity)`` slice and Phase
4.3 will add wakeup compensation (``place_entity``).
"""

from __future__ import annotations

from src.core.process import Process
from src.core.rbtree import RBTree
from src.core.scheduler_base import SchedulerBase

# ---------------------------------------------------------------------------
# Constants copied from the Linux kernel.
# Source: ``kernel/sched/core.c`` -> ``sched_prio_to_weight[]`` (the table is
# stable across the 2.6 .. 6.5 era; commit ``8a25d5debf83`` is one anchor).
# ---------------------------------------------------------------------------

NICE_0_LOAD: int = 1024
"""Weight assigned to a default-nice (0) task.

Source: ``include/linux/sched/prio.h`` (Linux kernel).
"""

PRIO_TO_WEIGHT: tuple[int, ...] = (
    # nice -20 .. -16
    88761, 71755, 56483, 46273, 36291,
    # nice -15 .. -11
    29154, 23254, 18705, 14949, 11916,
    # nice -10 .. -6
    9548, 7620, 6100, 4904, 3906,
    # nice -5 .. -1
    3121, 2501, 1991, 1586, 1277,
    # nice  0 ..  4
    1024, 820, 655, 526, 423,
    # nice  5 ..  9
    335, 272, 215, 172, 137,
    # nice 10 .. 14
    110, 87, 70, 56, 45,
    # nice 15 .. 19
    36, 29, 23, 18, 15,
)
"""Linux ``sched_prio_to_weight[40]`` — index 0..39 maps nice -20..+19.

Verbatim from ``kernel/sched/core.c``. Required by the paper's
Methodology section as evidence that the simulator's weight choices
are not hand-tuned but kernel-canonical.
"""


def weight_of(nice: int) -> int:
    """Look up the kernel-canonical weight for a nice value in [-20, 19]."""
    if not -20 <= nice <= 19:
        raise ValueError(f"nice must be in [-20, 19], got {nice}")
    return PRIO_TO_WEIGHT[nice + 20]


# ---------------------------------------------------------------------------
# Tunables (Linux defaults from kernel/sched/fair.c). Expressed in ticks so
# the simulator can treat them as integers without floating-point drift.
# ---------------------------------------------------------------------------

DEFAULT_SCHED_LATENCY: int = 8
"""Target period over which every runnable task should run once."""

DEFAULT_MIN_GRANULARITY: int = 2
"""Lower bound on a task's slice — caps preemption overhead at high nr_running."""


class CFS(SchedulerBase):
    """CFS — vruntime-fair, RB-tree backed, weight-aware."""

    def __init__(
        self,
        num_cpus: int = 1,
        sched_latency: int = DEFAULT_SCHED_LATENCY,
        min_granularity: int = DEFAULT_MIN_GRANULARITY,
    ) -> None:
        super().__init__(num_cpus=num_cpus)
        if sched_latency <= 0:
            raise ValueError(f"sched_latency must be positive, got {sched_latency}")
        if min_granularity <= 0:
            raise ValueError(f"min_granularity must be positive, got {min_granularity}")
        self.sched_latency = sched_latency
        self.min_granularity = min_granularity

        self._runqueues: list[RBTree] = [RBTree() for _ in range(num_cpus)]
        self._min_vruntime: list[float] = [0.0 for _ in range(num_cpus)]
        # pid -> RBTree node handle (so we can O(log n) delete on requeue).
        self._handles: dict[int, object] = {}
        # pid -> ticks consumed since last dispatch (for min_granularity bound).
        self._slice_used: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Arrival / IO transitions
    # ------------------------------------------------------------------

    def on_arrival(self, process: Process, now: int) -> None:
        process.weight = weight_of(process.nice_value)
        self._place_entity(process, cpu_id=0)
        self._enqueue(process, cpu_id=0)

    def on_unblock(self, process: Process, now: int) -> None:
        cpu = process.last_cpu if process.last_cpu is not None else 0
        self._place_entity(process, cpu_id=cpu)
        self._enqueue(process, cpu_id=cpu)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        # Renormalize vruntime against the destination CPU's baseline so the
        # migrating task is neither advantaged nor penalized by raw-vruntime drift.
        self._place_entity(process, cpu_id=target_cpu)
        self._enqueue(process, cpu_id=target_cpu)

    def on_block(self, process: Process, now: int) -> None:
        # Voluntary yield. Slice counter no longer relevant.
        self._slice_used.pop(process.pid, None)

    # ------------------------------------------------------------------
    # Preemption / dispatch
    # ------------------------------------------------------------------

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        # vruntime accounting — the heart of CFS fairness.
        running.vruntime += NICE_0_LOAD / running.weight
        used = self._slice_used.get(running.pid, 0) + 1
        self._slice_used[running.pid] = used

        # Dynamic slice: each task gets its weight-proportional share of
        # ``sched_period`` ticks, floored at ``min_granularity``.
        ideal = self._compute_slice(cpu_id, running)
        if used < ideal:
            return False
        # Slice consumed — yield iff another task is waiting.
        return self._runqueues[cpu_id].leftmost() is not None

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        tree = self._runqueues[cpu_id]
        leftmost = tree.leftmost()
        if leftmost is None:
            return None
        _, picked = leftmost
        # Remove via the cached handle (true O(log n) delete).
        tree.delete(self._handles.pop(picked.pid))
        self._slice_used[picked.pid] = 0
        # Update min_vruntime monotonically.
        next_leftmost = tree.leftmost()
        if next_leftmost is not None:
            self._min_vruntime[cpu_id] = max(self._min_vruntime[cpu_id], next_leftmost[0])
        return picked

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        # Process was preempted but still READY. Re-insert with the (now
        # advanced) vruntime so it competes fairly with siblings.
        self._slice_used.pop(process.pid, None)
        self._enqueue(process, cpu_id=cpu_id)

    # ------------------------------------------------------------------
    # Load balancing hooks
    # ------------------------------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        # Steal the RIGHTMOST entry: the task that has earned the most
        # vruntime, i.e. the one losing the least by being migrated.
        rightmost = self._runqueues[cpu_id].rightmost()
        if rightmost is None:
            return None
        return rightmost[1]

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        handle = self._handles.pop(process.pid, None)
        if handle is None:
            return
        self._runqueues[cpu_id].delete(handle)
        self._slice_used.pop(process.pid, None)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._runqueues[cpu_id])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, process: Process, cpu_id: int) -> None:
        tree = self._runqueues[cpu_id]
        handle = tree.insert(process.vruntime, process)
        self._handles[process.pid] = handle

    def _compute_slice(self, cpu_id: int, running: Process) -> int:
        """Kernel formula::

            sched_period = max(sched_latency, nr_running * min_granularity)
            slice        = max(min_granularity,
                               sched_period * weight / total_weight)

        ``nr_running`` counts the currently running task plus every task
        sitting in the RB-tree on ``cpu_id``. ``running`` is the task whose
        slice we want and is assumed to be OUT of the tree (the contract
        held by ``on_tick``, which is the only production caller).
        """
        tree = self._runqueues[cpu_id]
        nr_running = len(tree) + 1  # +1 for the on-CPU task
        period = max(self.sched_latency, nr_running * self.min_granularity)
        total_weight = running.weight + sum(p.weight for _, p in tree.iter_inorder())
        if total_weight <= 0:
            return self.min_granularity
        share = period * running.weight // total_weight
        return max(self.min_granularity, share)

    def _place_entity(self, process: Process, cpu_id: int) -> None:
        """``place_entity`` — wake-up compensation.

        Kernel reference: ``kernel/sched/fair.c::place_entity`` — a task
        returning from sleep gets a small credit (``sched_latency / 2``)
        relative to the CPU's ``min_vruntime``, but is never allowed to
        drop below its own previously-accumulated vruntime.
        """
        credit = self.sched_latency / 2
        baseline = self._min_vruntime[cpu_id] - credit
        process.vruntime = max(process.vruntime, baseline)
