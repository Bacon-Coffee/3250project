"""Per-CPU bookkeeping + simplified load balancer (idle-balance only).

CLAUDE.md invariant #3: the LoadBalancer is algorithm-agnostic — it only
interacts with the scheduler through ``peek_steal_candidate`` and
``pop_for_migration``. The scheduler decides what "cheapest to migrate" means
(CFS/EEVDF: rightmost vruntime; FCFS/RR: tail of FIFO; MLFQ: lowest level).

Phase 1 scope: skeleton + idle-steal algorithm. The simulator wires it in;
Phase 6 actually triggers migrations and validates the H4 (multi-core) hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


@dataclass
class CPU:
    """Bookkeeping for one logical CPU. Runqueue itself lives in the scheduler."""

    cpu_id: int
    running: Process | None = None

    @property
    def is_idle(self) -> bool:
        return self.running is None


MIGRATION_COST_TICKS: int = 1
"""Migration penalty (ticks) charged when a task is stolen across CPUs.

Models a coarse-grained L1/L2 cache miss; tuned by Phase 6 experiments.
"""


class LoadBalancer:
    """Linux-style ``idle_balance``: when a CPU runs out of work, try to steal.

    Algorithm (executed when ``target_cpu`` has nothing in its runqueue):
        1. Find the BUSIEST peer CPU by :meth:`SchedulerBase.runqueue_size`.
        2. If its runqueue is strictly larger, call ``peek_steal_candidate``.
        3. If a candidate exists, call ``pop_for_migration`` to remove it from
           the source queue, mark ``last_cpu`` for cost accounting, and return it.

    Returns the stolen process (or ``None`` if nothing was stealable).

    The simulator is responsible for the actual ``MIGRATION_DONE`` event and
    tick cost — this class is pure policy.
    """

    def __init__(self, scheduler: SchedulerBase) -> None:
        self.scheduler = scheduler

    def find_busiest_peer(self, target_cpu: int) -> int | None:
        """Return the peer CPU id whose runqueue is strictly larger than target's."""
        target_size = self.scheduler.runqueue_size(target_cpu)
        busiest_id: int | None = None
        busiest_size = target_size
        for cpu_id in range(self.scheduler.num_cpus):
            if cpu_id == target_cpu:
                continue
            size = self.scheduler.runqueue_size(cpu_id)
            if size > busiest_size:
                busiest_id = cpu_id
                busiest_size = size
        return busiest_id

    def try_steal(self, target_cpu: int) -> tuple[Process, int] | None:
        """Attempt one idle-balance pull into ``target_cpu``.

        Returns ``(process, source_cpu)`` on success, ``None`` if nothing was
        stealable. Caller charges :data:`MIGRATION_COST_TICKS` ticks of delay.
        """
        if target_cpu < 0 or target_cpu >= self.scheduler.num_cpus:
            raise IndexError(f"target_cpu {target_cpu} out of range")
        source = self.find_busiest_peer(target_cpu)
        if source is None:
            return None
        candidate = self.scheduler.peek_steal_candidate(source)
        if candidate is None:
            return None
        self.scheduler.pop_for_migration(candidate, source)
        candidate.last_cpu = source
        return candidate, source
