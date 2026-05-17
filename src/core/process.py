"""Process data structure and state machine.

A Process is the unit of work for the simulator. Its state transitions are:

    NEW --(arrival)--> READY <--(preempt)-- RUNNING --(burst done)--> TERMINATED
                          ^                    |
                          |                    v
                          +-- (IO complete) WAITING

Statistics fields are written by the simulator; algorithms only consume the
scheduler-state fields (vruntime / lag / virtual_deadline / weight).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProcessState(Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"


@dataclass(slots=True)
class IOPattern:
    """Alternating CPU/IO burst pattern.

    The process runs ``cpu_burst`` CPU ticks, then blocks for ``io_burst`` ticks,
    repeating until total CPU time used equals :attr:`Process.burst_time`.
    """

    cpu_burst: int
    io_burst: int

    def __post_init__(self) -> None:
        if self.cpu_burst <= 0 or self.io_burst <= 0:
            raise ValueError("cpu_burst and io_burst must both be positive")


@dataclass
class Process:
    """A schedulable task.

    Identity / immutable workload:
        pid:           unique identifier
        arrival_time:  tick at which the process enters the system
        burst_time:    TOTAL CPU ticks required to complete
        priority:      lower = higher priority (MLFQ initial level)
        nice_value:    Linux nice in [-20, 19]; used by CFS / EEVDF
        io_pattern:    optional CPU/IO alternation pattern

    Runtime state (mutated by simulator):
        state:                NEW / READY / RUNNING / WAITING / TERMINATED
        cpu_used:             total CPU ticks completed across the run
        cpu_burst_progress:   CPU ticks completed in the CURRENT burst (resets on IO)
        on_cpu:               id of the CPU currently running this process (else None)
        last_cpu:             id of the most recent CPU (for migration cost / affinity)

    Statistics (set by simulator):
        start_time:   first tick on which it ran (origin of response_time)
        finish_time:  tick on which it TERMINATED
        wait_time:    total ticks spent in READY
        _ready_since: bookkeeping for accumulating wait_time (private)

    Scheduler-specific state (set by the algorithm; simulator does not touch):
        vruntime:          weighted virtual runtime (CFS, kernel/sched/fair.c)
        lag:               service deficit (EEVDF, Peter Zijlstra 2023 commits)
        virtual_deadline:  vd = vruntime + request_size * weight_inv (EEVDF)
        weight:            derived from nice_value via prio_to_weight[]
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 0
    nice_value: int = 0
    io_pattern: IOPattern | None = None

    state: ProcessState = ProcessState.NEW
    cpu_used: int = 0
    cpu_burst_progress: int = 0
    on_cpu: int | None = None
    last_cpu: int | None = None

    start_time: int | None = None
    finish_time: int | None = None
    wait_time: int = 0
    _ready_since: int | None = None

    vruntime: float = 0.0
    lag: float = 0.0
    virtual_deadline: float = 0.0
    weight: int = 1024

    def __post_init__(self) -> None:
        if self.burst_time <= 0:
            raise ValueError(f"burst_time must be positive, got {self.burst_time}")
        if self.arrival_time < 0:
            raise ValueError(f"arrival_time must be non-negative, got {self.arrival_time}")
        if not -20 <= self.nice_value <= 19:
            raise ValueError(f"nice_value must be in [-20, 19], got {self.nice_value}")

    @property
    def response_time(self) -> int | None:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self) -> int | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def remaining(self) -> int:
        return max(0, self.burst_time - self.cpu_used)

    @property
    def is_io_bound(self) -> bool:
        return self.io_pattern is not None

    def needs_io_now(self) -> bool:
        """True if the process has exhausted its current CPU burst and should block."""
        if self.io_pattern is None:
            return False
        return self.cpu_burst_progress >= self.io_pattern.cpu_burst

    def enter_ready(self, now: int) -> None:
        """Transition into READY at tick ``now`` (called by simulator)."""
        self.state = ProcessState.READY
        self._ready_since = now

    def leave_ready(self, now: int) -> None:
        """Charge accumulated ready-queue wait time and clear the marker."""
        if self._ready_since is not None:
            self.wait_time += now - self._ready_since
            self._ready_since = None

    def __repr__(self) -> str:
        return f"P{self.pid}({self.state.value}, cpu={self.cpu_used}/{self.burst_time})"
