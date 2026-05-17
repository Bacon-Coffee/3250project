"""Abstract scheduler interface.

CLAUDE.md invariant #2: every algorithm in :mod:`src.algorithms` MUST implement
the full interface below. The simulator NEVER branches on ``isinstance(sched, CFS)``.

Interface contract (called by :class:`src.core.event.Simulator`):

    on_arrival(p, now)        — a NEW process has entered the system
    on_unblock(p, now)        — a WAITING process has completed IO and is ready
    on_block(p, now)          — RUNNING process is about to WAIT for IO
    on_tick(now, cpu, p)      — one CPU tick was consumed by p on cpu; may preempt
    pick_next(cpu, now)       — return next READY process for cpu, or None
    requeue(p, cpu, now)      — preempted-but-not-blocked p goes back into queue
    peek_steal_candidate(cpu) — read-only: which task on cpu would migrate cheapest
    pop_for_migration(p, cpu) — remove p from cpu's runqueue for stealing (Phase 6)
    runqueue_size(cpu)        — current ready-queue length on cpu (for LB heuristics)

Per-CPU state ownership: each scheduler owns its own per-CPU runqueue
(an array of len = ``num_cpus``). Algorithms differ ONLY in what data structure
backs each runqueue and what key is used to pick next.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.process import Process


class SchedulerBase(ABC):
    """Algorithm-agnostic scheduler interface."""

    def __init__(self, num_cpus: int = 1) -> None:
        if num_cpus < 1:
            raise ValueError(f"num_cpus must be >= 1, got {num_cpus}")
        self.num_cpus = num_cpus

    @abstractmethod
    def on_arrival(self, process: Process, now: int) -> None:
        """A NEW process entered the system at ``now``. Place it on some runqueue."""

    @abstractmethod
    def on_unblock(self, process: Process, now: int) -> None:
        """A WAITING process completed IO at ``now`` and is now READY."""

    def on_block(self, process: Process, now: int) -> None:  # noqa: B027
        """Optional hook: ``process`` is about to enter WAITING. Default: no-op.

        Most algorithms just need to forget the RUNNING task; the simulator
        already removes it from the CPU before calling this.
        """

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        """A migrated process has landed on ``target_cpu`` after MIGRATION_DONE.

        Default: route through :meth:`on_unblock`. Algorithms that own their
        per-CPU runqueues (CFS, EEVDF, MLFQ) should override to insert the
        process specifically into ``target_cpu``'s queue (and, for CFS/EEVDF,
        re-normalize vruntime against the target's ``min_vruntime``).
        """
        self.on_unblock(process, now)

    @abstractmethod
    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        """One CPU tick was just consumed by ``running`` on ``cpu_id``.

        Returns ``True`` if the simulator should preempt ``running`` after this tick
        (e.g. RR quantum expired, CFS vruntime exceeded sibling's, EEVDF deadline
        crossed). Returns ``False`` to keep running.
        """

    @abstractmethod
    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        """Pop and return the next READY process for ``cpu_id``, or ``None`` if empty.

        On success the returned process should be marked RUNNING by the simulator.
        """

    @abstractmethod
    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        """Reinsert a preempted (still-READY) process into ``cpu_id``'s runqueue."""

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        """Return a candidate for load balancing without removing it. Default: None.

        Implementations should return the task that is "cheapest to migrate"
        (CFS/EEVDF: rightmost of the RB-tree; FCFS/RR: tail of FIFO).
        """
        return None

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        """Remove ``process`` from ``cpu_id``'s runqueue for migration. Default: NotImpl."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support migration; "
            "implement pop_for_migration to enable load balancing."
        )

    def runqueue_size(self, cpu_id: int) -> int:  # pragma: no cover - default
        """Number of READY tasks queued on ``cpu_id``. Default: 0."""
        return 0
