"""Per-simulation metrics collection.

Collects raw per-process and per-CPU counters during the run and produces a
summary dict at the end. Used by Phase 8 ``run_all.py`` to populate result CSVs.

Indicators (CLAUDE.md Task 1.4):

    Per-process latency:  mean / p50 / p95 / p99 of wait / turnaround / response
    Throughput:           completed processes per unit time
    Context switches:     per CPU
    Fairness:             Jain's Fairness Index + Min-Max Ratio over CPU shares
    CPU utilization:      per CPU + aggregate
    Migrations:           total count (Phase 6 load balancer)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.core.process import Process, ProcessState


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over a PRE-SORTED list. ``pct`` in [0, 100]."""
    if not sorted_values:
        return float("nan")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def jains_fairness(shares: list[float]) -> float:
    """Jain's Fairness Index: J = (Σx)² / (n · Σx²), in (0, 1].

    1.0 = perfectly fair; 1/n = maximally unfair (one party gets everything).
    Returns ``nan`` if no positive shares exist.
    """
    positives = [x for x in shares if x > 0]
    if not positives:
        return float("nan")
    sum_x = sum(positives)
    sum_x_sq = sum(x * x for x in positives)
    return (sum_x * sum_x) / (len(positives) * sum_x_sq)


def min_max_ratio(shares: list[float]) -> float:
    """max(x) / min(x) over positive shares. Returns ``nan`` if fewer than 2 positives."""
    positives = [x for x in shares if x > 0]
    if len(positives) < 2:
        return float("nan")
    return max(positives) / min(positives)


@dataclass
class Metrics:
    """Mutable collector — call ``record_*`` during the run, ``summary()`` at end."""

    num_cpus: int
    processes: list[Process] = field(default_factory=list)
    running_ticks: list[int] = field(default_factory=list)
    idle_ticks: list[int] = field(default_factory=list)
    context_switches: list[int] = field(default_factory=list)
    migrations: int = 0
    total_ticks: int = 0

    def __post_init__(self) -> None:
        if self.num_cpus < 1:
            raise ValueError(f"num_cpus must be >= 1, got {self.num_cpus}")
        for lst_name in ("running_ticks", "idle_ticks", "context_switches"):
            lst = getattr(self, lst_name)
            if len(lst) == 0:
                setattr(self, lst_name, [0] * self.num_cpus)
            elif len(lst) != self.num_cpus:
                raise ValueError(
                    f"{lst_name} length {len(lst)} does not match num_cpus {self.num_cpus}"
                )

    def record_run(self, cpu_id: int, ticks: int = 1) -> None:
        self.running_ticks[cpu_id] += ticks

    def record_idle(self, cpu_id: int, ticks: int = 1) -> None:
        self.idle_ticks[cpu_id] += ticks

    def record_context_switch(self, cpu_id: int) -> None:
        self.context_switches[cpu_id] += 1

    def record_migration(self) -> None:
        self.migrations += 1

    def register(self, process: Process) -> None:
        self.processes.append(process)

    def summary(self) -> dict[str, Any]:
        """Compute aggregate summary. Safe to call mid-run; just less meaningful."""
        completed = [p for p in self.processes if p.state == ProcessState.TERMINATED]
        n_completed = len(completed)

        wait = sorted(p.wait_time for p in completed)
        turn = sorted(p.turnaround_time for p in completed if p.turnaround_time is not None)
        resp = sorted(p.response_time for p in completed if p.response_time is not None)

        shares = [float(p.cpu_used) for p in completed]

        util = [
            (self.running_ticks[c] / self.total_ticks) if self.total_ticks > 0 else 0.0
            for c in range(self.num_cpus)
        ]
        agg_util = (
            sum(self.running_ticks) / (self.num_cpus * self.total_ticks)
            if self.total_ticks > 0
            else 0.0
        )

        return {
            "total_ticks": self.total_ticks,
            "completed": n_completed,
            "throughput": (n_completed / self.total_ticks) if self.total_ticks > 0 else 0.0,
            "wait_time": _latency_stats(wait),
            "turnaround_time": _latency_stats(turn),
            "response_time": _latency_stats(resp),
            "cpu_utilization_per_cpu": util,
            "cpu_utilization_aggregate": agg_util,
            "context_switches_per_cpu": list(self.context_switches),
            "context_switches_total": sum(self.context_switches),
            "migrations": self.migrations,
            "jains_fairness": jains_fairness(shares),
            "min_max_ratio": min_max_ratio(shares),
        }


def _latency_stats(sorted_values: list[float]) -> dict[str, float]:
    if not sorted_values:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "mean": statistics.fmean(sorted_values),
        "p50": percentile(sorted_values, 50),
        "p95": percentile(sorted_values, 95),
        "p99": percentile(sorted_values, 99),
    }
