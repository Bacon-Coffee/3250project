"""Multi-Level Feedback Queue scheduler (3 levels, OSTEP ch. 8 rules).

Rules implemented:

  R1.  Highest non-empty level wins; equal-level tasks run round-robin.
  R3.  New arrivals enter at the top level (Q0).
  R4.  A task that exhausts its quantum at level L is demoted to L+1
       (capped at the bottom level).
  R5b. I/O boost — a task that returns from IO is placed back at Q0
       (toggleable via ``io_boost``; default ON).
  R6.  Periodic priority boost — every ``boost_interval`` ticks, every
       known task is restored to Q0 (anti-starvation).

Per-level quantum defaults to ``(1, 2, 4)`` (small at top, larger at
the bottom — interactive tasks pay almost no preemption cost, while
CPU hogs accumulate longer slices once they sink).
"""

from __future__ import annotations

from collections import deque

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


class MLFQ(SchedulerBase):
    """3-level Multi-Level Feedback Queue."""

    NUM_LEVELS: int = 3

    def __init__(
        self,
        num_cpus: int = 1,
        quanta: tuple[int, int, int] = (1, 2, 4),
        boost_interval: int = 100,
        io_boost: bool = True,
    ) -> None:
        super().__init__(num_cpus=num_cpus)
        if len(quanta) != self.NUM_LEVELS:
            raise ValueError(f"quanta must have {self.NUM_LEVELS} entries, got {len(quanta)}")
        if any(q <= 0 for q in quanta):
            raise ValueError(f"all quanta must be positive, got {quanta}")
        if boost_interval <= 0:
            raise ValueError(f"boost_interval must be positive, got {boost_interval}")
        self.quanta = tuple(quanta)
        self.boost_interval = boost_interval
        self.io_boost = io_boost

        self._queues: list[list[deque[Process]]] = [
            [deque() for _ in range(self.NUM_LEVELS)] for _ in range(num_cpus)
        ]
        self._level: dict[int, int] = {}
        self._slice_used: dict[int, int] = {}
        self._last_boost: int = 0

    # ------------------------------------------------------------------
    # Diagnostics — analog of /proc inspection in the real kernel.
    # ------------------------------------------------------------------

    def current_level_of(self, pid: int) -> int | None:
        """Return the level a task is currently classified at, or None if unknown."""
        return self._level.get(pid)

    # --- arrivals / IO --------------------------------------------------

    def on_arrival(self, process: Process, now: int) -> None:
        # R3: new arrivals get the top priority.
        self._level[process.pid] = 0
        self._slice_used[process.pid] = 0
        self._queues[0][0].append(process)

    def on_unblock(self, process: Process, now: int) -> None:
        if self.io_boost:
            self._level[process.pid] = 0
        self._slice_used[process.pid] = 0
        cpu = process.last_cpu if process.last_cpu is not None else 0
        level = self._level.get(process.pid, 0)
        self._queues[cpu][level].append(process)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        # Migrated tasks keep their current level.
        level = self._level.get(process.pid, 0)
        self._queues[target_cpu][level].append(process)

    def on_block(self, process: Process, now: int) -> None:
        # Voluntary yield (rule 4b would keep the level; we already do, since
        # we only demote on slice exhaustion). Just clear the slice counter.
        self._slice_used.pop(process.pid, None)

    # --- preemption / dispatch -----------------------------------------

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        self._maybe_boost(now)
        level = self._level.get(running.pid, 0)

        # R1: yield immediately if a strictly higher-priority task is waiting.
        for higher in range(level):
            if self._queues[cpu_id][higher]:
                # Don't demote — we were displaced, not exhausted.
                self._slice_used[running.pid] = 0
                return True

        used = self._slice_used.get(running.pid, 0) + 1
        self._slice_used[running.pid] = used
        if used >= self.quanta[level]:
            # R4: quantum exhausted, demote one step (capped).
            self._level[running.pid] = min(level + 1, self.NUM_LEVELS - 1)
            self._slice_used[running.pid] = 0
            return True
        return False

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        self._maybe_boost(now)
        for level in range(self.NUM_LEVELS):
            q = self._queues[cpu_id][level]
            if q:
                return q.popleft()
        return None

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        level = self._level.get(process.pid, 0)
        self._queues[cpu_id][level].append(process)

    # --- load balancing hooks ------------------------------------------

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        # Prefer to steal LOW-priority work (least disruptive). Walk from the
        # bottom level upward; within a level, steal the tail (FIFO-friendly).
        for level in range(self.NUM_LEVELS - 1, -1, -1):
            q = self._queues[cpu_id][level]
            if q:
                return q[-1]
        return None

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        level = self._level.get(process.pid, 0)
        self._queues[cpu_id][level].remove(process)

    def runqueue_size(self, cpu_id: int) -> int:
        return sum(len(q) for q in self._queues[cpu_id])

    # ------------------------------------------------------------------
    # Internal — priority boost / aging.
    # ------------------------------------------------------------------

    def _maybe_boost(self, now: int) -> None:
        if now - self._last_boost < self.boost_interval:
            return
        self._last_boost = now
        # R6: lift every known task back to Q0 and reset their slice budget.
        for pid in self._level:
            self._level[pid] = 0
            self._slice_used[pid] = 0
        for cpu_id in range(self.num_cpus):
            levels = self._queues[cpu_id]
            for level in range(1, self.NUM_LEVELS):
                while levels[level]:
                    levels[0].append(levels[level].popleft())
