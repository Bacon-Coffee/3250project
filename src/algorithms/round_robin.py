"""Round Robin scheduler with parameterized quantum.

Per-CPU FIFO queue + a per-PID slice counter. ``on_tick`` returns True
once a task has spent ``quantum`` ticks on CPU since being dispatched;
the simulator then yields it back via :meth:`requeue`, which appends to
the tail and clears the slice counter.

Choosing ``quantum`` is a fundamental RR tradeoff:

* small quantum   -> low response_time, high context-switch overhead
* large quantum   -> behaves more like FCFS

This module exposes the parameter so the experiments harness can sweep
it (Phase 8 produces an RR-quantum-vs-response-time plot for the paper).
"""

from __future__ import annotations

from collections import deque

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


class RoundRobin(SchedulerBase):
    """Classic per-CPU Round Robin with explicit quantum."""

    def __init__(self, num_cpus: int = 1, quantum: int = 4) -> None:
        super().__init__(num_cpus=num_cpus)
        if quantum <= 0:
            raise ValueError(f"quantum must be positive, got {quantum}")
        self.quantum = quantum
        self._queues: list[deque[Process]] = [deque() for _ in range(num_cpus)]
        self._slice_used: dict[int, int] = {}  # pid -> ticks consumed in current slice

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
        used = self._slice_used.get(running.pid, 0) + 1
        self._slice_used[running.pid] = used
        return used >= self.quantum

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        if not self._queues[cpu_id]:
            return None
        picked = self._queues[cpu_id].popleft()
        # Fresh slice starts now.
        self._slice_used[picked.pid] = 0
        return picked

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        # End of quantum: clear the counter and re-queue at the tail.
        self._slice_used.pop(process.pid, None)
        self._queues[cpu_id].append(process)

    def on_block(self, process: Process, now: int) -> None:
        # Forget the slice counter; we'll start fresh when IO completes.
        self._slice_used.pop(process.pid, None)

    # --- load balancing hooks ------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self._queues[cpu_id][-1] if self._queues[cpu_id] else None

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self._queues[cpu_id].remove(process)
        self._slice_used.pop(process.pid, None)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._queues[cpu_id])
