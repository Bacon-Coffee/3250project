"""Synthetic workload generator: Poisson arrivals + lognormal CPU bursts.

CLAUDE.md Phase 7 Task 7.1. Three profiles drive the H1-H3 experiments:

* CPU_HEAVY — long compute-bound tasks, no IO. Stresses the scheduler's
  fairness machinery (vruntime accounting, RB-tree maintenance).
* IO_HEAVY  — short CPU bursts alternating with longer IO waits. Exercises
  the wake-up path (CFS sleeper compensation, EEVDF lag re-injection).
* MIXED     — a 50/50 blend. The realistic case the paper builds its main
  comparison around.

All sampling goes through ``numpy.random.Generator``: seeding once gives
bit-exact reproducibility, which the paper figures rely on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.core.process import IOPattern, Process


class Profile(Enum):
    CPU_HEAVY = "cpu_heavy"
    IO_HEAVY = "io_heavy"
    MIXED = "mixed"


@dataclass(frozen=True)
class WorkloadConfig:
    """All knobs needed to deterministically build a workload."""

    profile: Profile
    n_processes: int
    arrival_rate: float
    """Poisson rate λ in arrivals per tick. Inter-arrival ~ Exp(1/λ)."""

    burst_mu: float
    """ln(burst_time) mean (lognormal mu parameter)."""

    burst_sigma: float
    """ln(burst_time) std (lognormal sigma parameter)."""

    io_probability: float
    """Per-process probability of having an IOPattern attached."""

    io_cpu_burst_mean: int = 5
    """Mean CPU burst length (ticks) of the on/off cycle."""

    io_io_burst_mean: int = 5
    """Mean IO burst length (ticks) of the on/off cycle."""

    nice_range: tuple[int, int] | None = None
    """Inclusive integer range to draw nice from. ``None`` ⇒ all zero."""

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_processes <= 0:
            raise ValueError(f"n_processes must be positive, got {self.n_processes}")
        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be positive, got {self.arrival_rate}")
        if self.burst_sigma <= 0:
            raise ValueError(f"burst_sigma must be positive, got {self.burst_sigma}")
        if not 0.0 <= self.io_probability <= 1.0:
            raise ValueError(
                f"io_probability must be in [0, 1], got {self.io_probability}"
            )
        if self.io_cpu_burst_mean < 1 or self.io_io_burst_mean < 1:
            raise ValueError("io_cpu_burst_mean and io_io_burst_mean must be >= 1")
        if self.nice_range is not None:
            lo, hi = self.nice_range
            if lo > hi:
                raise ValueError(f"nice_range low > high: {self.nice_range}")
            if lo < -20 or hi > 19:
                raise ValueError(f"nice_range must lie in [-20, 19], got {self.nice_range}")


def generate_workload(cfg: WorkloadConfig) -> list[Process]:
    """Build ``cfg.n_processes`` processes by sampling from cfg's distributions."""
    rng = np.random.default_rng(cfg.seed)

    inter_arrivals = rng.exponential(scale=1.0 / cfg.arrival_rate, size=cfg.n_processes)
    arrival_times = np.cumsum(inter_arrivals).round().astype(int)

    bursts = rng.lognormal(mean=cfg.burst_mu, sigma=cfg.burst_sigma, size=cfg.n_processes)
    bursts = np.clip(bursts.round().astype(int), 1, None)

    if cfg.nice_range is None:
        nices = np.zeros(cfg.n_processes, dtype=int)
    else:
        lo, hi = cfg.nice_range
        nices = rng.integers(lo, hi + 1, size=cfg.n_processes)

    io_decisions = rng.random(size=cfg.n_processes) < cfg.io_probability
    cpu_bursts = np.clip(rng.poisson(lam=cfg.io_cpu_burst_mean, size=cfg.n_processes), 1, None)
    io_bursts = np.clip(rng.poisson(lam=cfg.io_io_burst_mean, size=cfg.n_processes), 1, None)

    processes: list[Process] = []
    for pid in range(cfg.n_processes):
        io_pattern = (
            IOPattern(cpu_burst=int(cpu_bursts[pid]), io_burst=int(io_bursts[pid]))
            if io_decisions[pid]
            else None
        )
        processes.append(
            Process(
                pid=pid,
                arrival_time=int(arrival_times[pid]),
                burst_time=int(bursts[pid]),
                nice_value=int(nices[pid]),
                io_pattern=io_pattern,
            )
        )
    return processes


def cpu_heavy(n: int, *, seed: int | None = None) -> list[Process]:
    """Compute-bound workload, no IO. Average burst ~= exp(mu + sigma^2 / 2) ticks."""
    return generate_workload(
        WorkloadConfig(
            profile=Profile.CPU_HEAVY,
            n_processes=n,
            arrival_rate=0.3,
            burst_mu=math.log(50),
            burst_sigma=0.7,
            io_probability=0.0,
            seed=seed,
        )
    )


def io_heavy(n: int, *, seed: int | None = None) -> list[Process]:
    """IO-bound workload: every process alternates short CPU / longer IO bursts."""
    return generate_workload(
        WorkloadConfig(
            profile=Profile.IO_HEAVY,
            n_processes=n,
            arrival_rate=1.0,
            burst_mu=math.log(20),
            burst_sigma=0.5,
            io_probability=1.0,
            io_cpu_burst_mean=2,
            io_io_burst_mean=8,
            seed=seed,
        )
    )


def mixed(n: int, *, seed: int | None = None) -> list[Process]:
    """50/50 blend of compute-bound and IO-bound processes."""
    return generate_workload(
        WorkloadConfig(
            profile=Profile.MIXED,
            n_processes=n,
            arrival_rate=0.5,
            burst_mu=math.log(30),
            burst_sigma=0.6,
            io_probability=0.5,
            io_cpu_burst_mean=4,
            io_io_burst_mean=4,
            seed=seed,
        )
    )
