"""Event-driven simulator engine.

Main loop (tick-based event-driven hybrid):
    1. ``_fire_events_at_now``  — drain all events whose time <= now
                                  (arrivals, IO completions, migration completions)
    2. ``_fill_idle_cpus``      — call ``scheduler.pick_next`` for each idle CPU;
                                  if empty AND a LoadBalancer is wired, try to steal.
    3. ``_tick_once``           — for each non-idle CPU: charge 1 tick of work,
                                  notify scheduler via ``on_tick``, decide fate
                                  (TERMINATE / block-for-IO / preempt).
    4. If all CPUs idle and events remain in queue: fast-forward to the next
       event's time (counting the gap as idle on every CPU). Otherwise advance
       1 tick and loop. Termination: all CPUs idle AND queue empty, or
       ``now`` reaches ``max_time``.

Event types (CLAUDE.md Task 1.2):
    PROCESS_ARRIVAL   — scheduled at process.arrival_time for every input process
    IO_COMPLETE       — scheduled when a process blocks; fires when IO burst ends
    MIGRATION_DONE    — scheduled when LoadBalancer steals; fires after cache-miss cost
    PROCESS_EXIT      — currently handled inline; reserved for future async exit hooks
    TIMER_INTERRUPT   — reserved for algorithms with explicit periodic timers
                        (CFS preemption is detected via ``on_tick`` return value, so
                        we don't currently emit TIMER_INTERRUPT events ourselves)
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from src.core.cpu import CPU, MIGRATION_COST_TICKS, LoadBalancer
from src.core.metrics import Metrics
from src.core.process import Process, ProcessState
from src.core.scheduler_base import SchedulerBase


class EventType(Enum):
    PROCESS_ARRIVAL = auto()
    TIMER_INTERRUPT = auto()
    IO_COMPLETE = auto()
    PROCESS_EXIT = auto()
    MIGRATION_DONE = auto()


@dataclass
class Event:
    time: int
    event_type: EventType
    process: Process | None = None
    cpu_id: int | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"event.time must be non-negative, got {self.time}")


class EventQueue:
    """Min-heap of pending events, stable on equal time by insertion order."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Event]] = []
        self._seq = itertools.count()

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.time, next(self._seq), event))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[2]

    def peek(self) -> Event | None:
        return self._heap[0][2] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


class Simulator:
    """Tick-based event-driven simulator.

    Single-threaded (CLAUDE.md invariant #6); ``num_cpus`` simulates parallel
    CPUs by giving each its own runqueue inside the scheduler, but actually
    advances one tick at a time across all CPUs in lock-step.
    """

    def __init__(
        self,
        scheduler: SchedulerBase,
        processes: list[Process],
        *,
        num_cpus: int | None = None,
        load_balancer: LoadBalancer | None = None,
        max_time: int = 10**7,
    ) -> None:
        cpus_target = num_cpus if num_cpus is not None else scheduler.num_cpus
        if scheduler.num_cpus != cpus_target:
            raise ValueError(
                f"num_cpus mismatch: scheduler.num_cpus={scheduler.num_cpus}, "
                f"simulator num_cpus={cpus_target}"
            )
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, got {max_time}")

        self.scheduler = scheduler
        self.processes = list(processes)
        self.num_cpus = cpus_target
        self.load_balancer = load_balancer
        self.max_time = max_time

        self.now: int = 0
        self.cpus: list[CPU] = [CPU(cpu_id=c) for c in range(self.num_cpus)]
        self.queue: EventQueue = EventQueue()
        self.metrics: Metrics = Metrics(num_cpus=self.num_cpus)
        for p in self.processes:
            self.metrics.register(p)

    def run(self) -> dict[str, Any]:
        for p in self.processes:
            self.queue.push(
                Event(
                    time=p.arrival_time,
                    event_type=EventType.PROCESS_ARRIVAL,
                    process=p,
                )
            )

        while self.now < self.max_time:
            self._fire_events_at_now()
            self._fill_idle_cpus()

            if all(c.is_idle for c in self.cpus):
                if not self.queue:
                    break
                self._fast_forward_to_next_event()
                continue

            self._tick_once()

        return self.metrics.summary()

    def _fire_events_at_now(self) -> None:
        while self.queue and self.queue.peek().time <= self.now:
            self._handle_event(self.queue.pop())

    def _handle_event(self, ev: Event) -> None:
        if ev.event_type == EventType.PROCESS_ARRIVAL:
            p = ev.process
            assert p is not None
            p.enter_ready(self.now)
            self.scheduler.on_arrival(p, self.now)
        elif ev.event_type == EventType.IO_COMPLETE:
            p = ev.process
            assert p is not None
            p.enter_ready(self.now)
            self.scheduler.on_unblock(p, self.now)
        elif ev.event_type == EventType.MIGRATION_DONE:
            p = ev.process
            assert p is not None
            assert ev.cpu_id is not None
            p.enter_ready(self.now)
            p.last_cpu = ev.cpu_id
            self.scheduler.on_migration_arrival(p, ev.cpu_id, self.now)
        # TIMER_INTERRUPT / PROCESS_EXIT: reserved, no-op in Phase 1

    def _fill_idle_cpus(self) -> None:
        for cpu in self.cpus:
            if not cpu.is_idle:
                continue
            picked = self.scheduler.pick_next(cpu.cpu_id, self.now)
            if picked is None and self.load_balancer is not None:
                stolen = self.load_balancer.try_steal(cpu.cpu_id)
                if stolen is not None:
                    process, _source = stolen
                    self.metrics.record_migration()
                    self.queue.push(
                        Event(
                            time=self.now + MIGRATION_COST_TICKS,
                            event_type=EventType.MIGRATION_DONE,
                            process=process,
                            cpu_id=cpu.cpu_id,
                        )
                    )
                    continue
            if picked is not None:
                self._dispatch(picked, cpu)

    def _dispatch(self, p: Process, cpu: CPU) -> None:
        p.leave_ready(self.now)
        p.state = ProcessState.RUNNING
        p.on_cpu = cpu.cpu_id
        if p.start_time is None:
            p.start_time = self.now
        cpu.running = p
        self.metrics.record_context_switch(cpu.cpu_id)

    def _tick_once(self) -> None:
        for cpu in self.cpus:
            running = cpu.running
            if running is None:
                self.metrics.record_idle(cpu.cpu_id, 1)
                continue
            running.cpu_used += 1
            running.cpu_burst_progress += 1
            self.metrics.record_run(cpu.cpu_id, 1)
            preempt = self.scheduler.on_tick(self.now + 1, cpu.cpu_id, running)

            # Fate priority: TERMINATE > IO-block > preempt
            if running.cpu_used >= running.burst_time:
                self._terminate(running, cpu)
            elif running.needs_io_now():
                self._block_for_io(running, cpu)
            elif preempt:
                self._yield_back(running, cpu)

        self.metrics.total_ticks += 1
        self.now += 1

    def _terminate(self, p: Process, cpu: CPU) -> None:
        cpu.running = None
        p.on_cpu = None
        p.last_cpu = cpu.cpu_id
        p.finish_time = self.now + 1
        p.state = ProcessState.TERMINATED

    def _block_for_io(self, p: Process, cpu: CPU) -> None:
        assert p.io_pattern is not None
        cpu.running = None
        p.on_cpu = None
        p.last_cpu = cpu.cpu_id
        p.state = ProcessState.WAITING
        p.cpu_burst_progress = 0
        self.scheduler.on_block(p, self.now + 1)
        self.queue.push(
            Event(
                time=self.now + 1 + p.io_pattern.io_burst,
                event_type=EventType.IO_COMPLETE,
                process=p,
            )
        )

    def _yield_back(self, p: Process, cpu: CPU) -> None:
        cpu.running = None
        p.on_cpu = None
        p.last_cpu = cpu.cpu_id
        p.enter_ready(self.now + 1)
        self.scheduler.requeue(p, cpu.cpu_id, self.now + 1)

    def _fast_forward_to_next_event(self) -> None:
        next_event = self.queue.peek()
        assert next_event is not None
        if next_event.time <= self.now:
            return
        gap = next_event.time - self.now
        for cpu in self.cpus:
            self.metrics.record_idle(cpu.cpu_id, gap)
        self.metrics.total_ticks += gap
        self.now = next_event.time
