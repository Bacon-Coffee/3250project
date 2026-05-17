"""Gantt chart rendering for per-CPU scheduling timelines.

Records dispatch events while a simulation runs and renders the resulting
intervals as a matplotlib ``broken_barh`` chart. Useful as an appendix figure
when the paper needs to demonstrate, e.g., CFS slice rotation or RR quantum
boundaries on a tiny worked example.

Wiring -- the simulator does not natively emit a timeline, so this module
provides a thin :class:`GanttRecorder` that wraps :class:`SchedulerBase` and
forwards every method while accumulating ``(cpu_id, pid, start, end)``
intervals from successive ``pick_next`` -> dispatch-end transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.process import Process
from src.core.scheduler_base import SchedulerBase


@dataclass
class Interval:
    cpu_id: int
    pid: int
    start: int
    end: int


class GanttRecorder(SchedulerBase):
    """Decorator-scheduler that records dispatch intervals.

    Subclasses ``SchedulerBase`` so the simulator can plug it in transparently.
    Every interface method delegates to ``self.inner``; ``pick_next`` and
    related transitions additionally maintain the timeline state.
    """

    def __init__(self, inner: SchedulerBase) -> None:
        super().__init__(num_cpus=inner.num_cpus)
        self.inner = inner
        self.intervals: list[Interval] = []
        # cpu_id -> (pid, start_time) for the currently dispatched task.
        self._running: dict[int, tuple[int, int]] = {}

    # --- pass-throughs --------------------------------------------------

    def on_arrival(self, process: Process, now: int) -> None:
        self.inner.on_arrival(process, now)

    def on_unblock(self, process: Process, now: int) -> None:
        self.inner.on_unblock(process, now)

    def on_block(self, process: Process, now: int) -> None:
        self._close_interval(process.on_cpu, now)
        self.inner.on_block(process, now)

    def on_migration_arrival(self, process: Process, target_cpu: int, now: int) -> None:
        self.inner.on_migration_arrival(process, target_cpu, now)

    def on_tick(self, now: int, cpu_id: int, running: Process) -> bool:
        return self.inner.on_tick(now, cpu_id, running)

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        # Close any prior interval on this CPU before logging the new dispatch.
        self._close_interval(cpu_id, now)
        picked = self.inner.pick_next(cpu_id, now)
        if picked is not None:
            self._running[cpu_id] = (picked.pid, now)
        return picked

    def requeue(self, process: Process, cpu_id: int, now: int) -> None:
        self._close_interval(cpu_id, now)
        self.inner.requeue(process, cpu_id, now)

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self.inner.peek_steal_candidate(cpu_id)

    def pop_for_migration(self, process: Process, cpu_id: int) -> None:
        self.inner.pop_for_migration(process, cpu_id)

    def runqueue_size(self, cpu_id: int) -> int:
        return self.inner.runqueue_size(cpu_id)

    # --- timeline bookkeeping ------------------------------------------

    def _close_interval(self, cpu_id: int | None, now: int) -> None:
        if cpu_id is None:
            return
        entry = self._running.pop(cpu_id, None)
        if entry is None:
            return
        pid, start = entry
        if now > start:
            self.intervals.append(Interval(cpu_id=cpu_id, pid=pid, start=start, end=now))

    def finalize(self, end_time: int) -> list[Interval]:
        for cpu_id in list(self._running):
            self._close_interval(cpu_id, end_time)
        return list(self.intervals)


def render_gantt(
    intervals: list[Interval],
    *,
    title: str,
    out_path: Path,
    end_time: int | None = None,
    pid_limit: int | None = None,
) -> None:
    """Render dispatch intervals as a broken_barh chart, one row per pid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not intervals:
        return
    pids = sorted({iv.pid for iv in intervals})
    if pid_limit is not None:
        pids = pids[:pid_limit]
    pid_to_row = {pid: i for i, pid in enumerate(pids)}

    cpu_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, max(2, 0.3 * len(pids))))
    max_cpu = max(iv.cpu_id for iv in intervals)
    for iv in intervals:
        if iv.pid not in pid_to_row:
            continue
        ax.broken_barh(
            [(iv.start, iv.end - iv.start)],
            (pid_to_row[iv.pid] - 0.4, 0.8),
            facecolors=cpu_colors[iv.cpu_id % len(cpu_colors)],
        )

    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f"P{p}" for p in pids])
    ax.set_xlabel("tick")
    ax.set_title(title)
    if end_time is not None:
        ax.set_xlim(0, end_time)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cpu_colors[c % len(cpu_colors)])
        for c in range(max_cpu + 1)
    ]
    ax.legend(handles, [f"CPU {c}" for c in range(len(handles))], loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


