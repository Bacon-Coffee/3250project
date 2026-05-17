"""First-Come, First-Served (FCFS) scheduler.

The simplest possible policy and the paper's convoy-effect baseline.
One FIFO queue per CPU; tasks are dispatched in insertion order and run
to completion (or until they block for IO). Non-preemptive.

Multi-CPU behaviour: on arrival a task is always placed on CPU 0; the
:class:`src.core.cpu.LoadBalancer` is responsible for pulling work to
the other CPUs when they go idle (CLAUDE.md invariant #3).
"""

from __future__ import annotations

from collections import deque

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


class FCFS(SchedulerBase):
    """Per-CPU FIFO, non-preemptive."""

    def __init__(self, num_cpus: int = 1) -> None:
        super().__init__(num_cpus=num_cpus)
        self._queues: list[deque[Process]] = [deque() for _ in range(num_cpus)]

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
        # FCFS never preempts; the simulator handles natural exit / IO-block.
        return False

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        if not self._queues[cpu_id]:
            return None
        return self._queues[cpu_id].popleft()

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        # Defensive: FCFS shouldn't trigger preemption, but if some external
        # event ever caused a yield, put the task back at the FRONT so it
        # retains its FCFS position.
        self._queues[cpu_id].appendleft(process)

    # --- load balancing hooks ------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self._queues[cpu_id][-1] if self._queues[cpu_id] else None

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self._queues[cpu_id].remove(process)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._queues[cpu_id])
