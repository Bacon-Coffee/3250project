"""Shortest-Job-First scheduler with optional SRTF preemption.

Two policies in one class, toggled by ``preemptive``:

* ``preemptive=False`` — classic SJF: ``pick_next`` selects the READY
  task with the smallest *remaining* CPU time; once dispatched it runs
  to completion (or IO block).
* ``preemptive=True``  — Shortest Remaining Time First (SRTF): on every
  tick we check the runqueue; if any queued task has strictly less
  remaining CPU than the currently running task, ``on_tick`` returns
  True and the simulator yields the running task back into the queue.

Ties are broken by ``arrival_time`` then ``pid`` for determinism.
Steal candidate for load balancing is the LONGEST-remaining task —
it would be picked last anyway, so migrating it costs the least.
"""

from __future__ import annotations

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


def _sort_key(p: Process) -> tuple[int, int, int]:
    return (p.remaining, p.arrival_time, p.pid)


class SJF(SchedulerBase):
    """SJF / SRTF — algorithm-agnostic LoadBalancer compatible."""

    def __init__(self, num_cpus: int = 1, preemptive: bool = False) -> None:
        super().__init__(num_cpus=num_cpus)
        self.preemptive = preemptive
        self._queues: list[list[Process]] = [[] for _ in range(num_cpus)]

    # --- arrivals / IO --------------------------------------------------

    def on_arrival(self, process: Process, now: int) -> None:
        self._queues[0].append(process)

    def on_unblock(self, process: Process, now: int) -> None:
        cpu = process.last_cpu if process.last_cpu is not None else 0
        self._queues[cpu].append(process)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        self._queues[target_cpu].append(process)

    # --- preemption / dispatch -----------------------------------------

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        if not self.preemptive:
            return False
        # SRTF: preempt iff some queued task has strictly less remaining work.
        return any(p.remaining < running.remaining for p in self._queues[cpu_id])

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        q = self._queues[cpu_id]
        if not q:
            return None
        idx_min = min(range(len(q)), key=lambda i: _sort_key(q[i]))
        return q.pop(idx_min)

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        self._queues[cpu_id].append(process)

    # --- load balancing hooks ------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        q = self._queues[cpu_id]
        if not q:
            return None
        # Longest remaining => last one we'd pick anyway, so cheapest to migrate.
        return max(q, key=lambda p: (p.remaining, p.pid))

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self._queues[cpu_id].remove(process)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._queues[cpu_id])
