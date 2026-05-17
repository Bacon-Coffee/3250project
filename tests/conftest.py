"""Shared test fixtures and stub schedulers used to exercise the engine.

The real algorithms live in :mod:`src.algorithms` (Phase 3+). For Phase 1 we
need a minimal SchedulerBase implementation that exercises every code path
of the simulator: arrivals, IO, preemption, multi-CPU dispatch.
"""

from __future__ import annotations

from collections import deque

import pytest

from src.core.cpu import LoadBalancer
from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


class FIFOStubScheduler(SchedulerBase):
    """Single FIFO queue per CPU; non-preemptive. New arrivals go to CPU 0."""

    def __init__(self, num_cpus: int = 1) -> None:
        super().__init__(num_cpus=num_cpus)
        self._queues: list[deque[Process]] = [deque() for _ in range(num_cpus)]

    def on_arrival(self, process: Process, now: int) -> None:
        self._queues[0].append(process)

    def on_unblock(self, process: Process, now: int) -> None:
        cpu = process.last_cpu if process.last_cpu is not None else 0
        self._queues[cpu].append(process)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        self._queues[target_cpu].append(process)

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        return False

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        if not self._queues[cpu_id]:
            return None
        return self._queues[cpu_id].popleft()

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        self._queues[cpu_id].append(process)

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self._queues[cpu_id][-1] if self._queues[cpu_id] else None

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self._queues[cpu_id].remove(process)

    def runqueue_size(self, cpu_id: int) -> int:
        return len(self._queues[cpu_id])


class QuantumStubScheduler(FIFOStubScheduler):
    """FIFO + fixed-quantum preemption. Used to exercise the requeue path."""

    def __init__(self, num_cpus: int = 1, quantum: int = 2) -> None:
        super().__init__(num_cpus=num_cpus)
        self.quantum = quantum
        self._slice_used: dict[int, int] = {}

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        picked = super().pick_next(cpu_id, now)
        if picked is not None:
            self._slice_used[picked.pid] = 0
        return picked

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        self._slice_used.pop(process.pid, None)
        super().requeue(process, cpu_id, now)

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        used = self._slice_used.get(running.pid, 0) + 1
        self._slice_used[running.pid] = used
        return used >= self.quantum


@pytest.fixture
def fifo_scheduler() -> FIFOStubScheduler:
    return FIFOStubScheduler(num_cpus=1)


@pytest.fixture
def fifo_dual_scheduler() -> FIFOStubScheduler:
    return FIFOStubScheduler(num_cpus=2)


@pytest.fixture
def quantum_scheduler() -> QuantumStubScheduler:
    return QuantumStubScheduler(num_cpus=1, quantum=2)


@pytest.fixture
def load_balancer_factory():
    def _make(scheduler: SchedulerBase) -> LoadBalancer:
        return LoadBalancer(scheduler)

    return _make
