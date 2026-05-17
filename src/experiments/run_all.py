"""Phase 8 driver — sweep 6 algorithms x 4 workloads x N seeds.

Produces a tidy long-format CSV (``results/csv/run_all.csv``) so the figure
scripts can pivot it however they need. Each row is one (algorithm, workload,
seed) trial summarised by the metrics dict from :class:`src.core.metrics.Metrics`.

Usage::

    python -m src.experiments.run_all
    python -m src.experiments.run_all --seeds 7 --processes 80 --dual-core
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.algorithms.cfs import CFS
from src.algorithms.eevdf import EEVDF
from src.algorithms.fcfs import FCFS
from src.algorithms.mlfq import MLFQ
from src.algorithms.round_robin import RoundRobin
from src.algorithms.sjf import SJF
from src.core.cpu import LoadBalancer
from src.core.event import Simulator
from src.core.process import Process
from src.core.scheduler_base import SchedulerBase
from src.workloads.synthetic import (
    Profile,
    WorkloadConfig,
    cpu_heavy,
    generate_workload,
    io_heavy,
    mixed,
)

# ---------------------------------------------------------------------------
# Algorithm registry -- name -> factory(num_cpus) -> SchedulerBase
# ---------------------------------------------------------------------------

AlgoFactory = Callable[[int], SchedulerBase]

ALGORITHMS: dict[str, AlgoFactory] = {
    "FCFS": lambda n: FCFS(num_cpus=n),
    "SJF": lambda n: SJF(num_cpus=n, preemptive=False),
    "SRTF": lambda n: SJF(num_cpus=n, preemptive=True),
    "RR": lambda n: RoundRobin(num_cpus=n, quantum=4),
    "MLFQ": lambda n: MLFQ(num_cpus=n, quanta=(1, 2, 4), boost_interval=100),
    "CFS": lambda n: CFS(num_cpus=n),
    "EEVDF": lambda n: EEVDF(num_cpus=n),
}


# ---------------------------------------------------------------------------
# Workload registry -- name -> factory(n_processes, seed) -> list[Process]
# ---------------------------------------------------------------------------

WorkloadFactory = Callable[[int, int], list[Process]]


def _nice_mixed(n: int, seed: int) -> list[Process]:
    """Mixed profile with non-zero nice spread -- exercises CFS/EEVDF weights."""
    return generate_workload(
        WorkloadConfig(
            profile=Profile.MIXED,
            n_processes=n,
            arrival_rate=0.5,
            burst_mu=math.log(30),
            burst_sigma=0.6,
            io_probability=0.3,
            io_cpu_burst_mean=4,
            io_io_burst_mean=4,
            nice_range=(-5, 5),
            seed=seed,
        )
    )


WORKLOADS: dict[str, WorkloadFactory] = {
    "cpu_heavy": lambda n, s: cpu_heavy(n, seed=s),
    "io_heavy": lambda n, s: io_heavy(n, seed=s),
    "mixed": lambda n, s: mixed(n, seed=s),
    "nice_mixed": _nice_mixed,
}


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrialResult:
    algorithm: str
    workload: str
    seed: int
    n_cpus: int
    n_processes: int
    wall_seconds: float
    summary: dict[str, Any]


def run_trial(
    algorithm: str,
    workload: str,
    seed: int,
    *,
    n_processes: int,
    n_cpus: int,
    max_time: int,
    enable_load_balancer: bool,
) -> TrialResult:
    procs = WORKLOADS[workload](n_processes, seed)
    scheduler = ALGORITHMS[algorithm](n_cpus)
    lb = LoadBalancer(scheduler) if (enable_load_balancer and n_cpus > 1) else None
    sim = Simulator(
        scheduler=scheduler,
        processes=procs,
        num_cpus=n_cpus,
        load_balancer=lb,
        max_time=max_time,
    )
    t0 = time.perf_counter()
    summary = sim.run()
    elapsed = time.perf_counter() - t0
    return TrialResult(
        algorithm=algorithm,
        workload=workload,
        seed=seed,
        n_cpus=n_cpus,
        n_processes=n_processes,
        wall_seconds=elapsed,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Sweep + CSV emit
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "algorithm",
    "workload",
    "seed",
    "n_cpus",
    "n_processes",
    "wall_seconds",
    "total_ticks",
    "completed",
    "throughput",
    "wait_mean",
    "wait_p50",
    "wait_p95",
    "wait_p99",
    "turnaround_mean",
    "turnaround_p95",
    "turnaround_p99",
    "response_mean",
    "response_p95",
    "response_p99",
    "cpu_utilization_aggregate",
    "context_switches_total",
    "migrations",
    "jains_fairness",
    "min_max_ratio",
]


def _flatten(trial: TrialResult) -> dict[str, Any]:
    s = trial.summary
    return {
        "algorithm": trial.algorithm,
        "workload": trial.workload,
        "seed": trial.seed,
        "n_cpus": trial.n_cpus,
        "n_processes": trial.n_processes,
        "wall_seconds": f"{trial.wall_seconds:.4f}",
        "total_ticks": s["total_ticks"],
        "completed": s["completed"],
        "throughput": f"{s['throughput']:.6f}",
        "wait_mean": f"{s['wait_time']['mean']:.4f}",
        "wait_p50": f"{s['wait_time']['p50']:.4f}",
        "wait_p95": f"{s['wait_time']['p95']:.4f}",
        "wait_p99": f"{s['wait_time']['p99']:.4f}",
        "turnaround_mean": f"{s['turnaround_time']['mean']:.4f}",
        "turnaround_p95": f"{s['turnaround_time']['p95']:.4f}",
        "turnaround_p99": f"{s['turnaround_time']['p99']:.4f}",
        "response_mean": f"{s['response_time']['mean']:.4f}",
        "response_p95": f"{s['response_time']['p95']:.4f}",
        "response_p99": f"{s['response_time']['p99']:.4f}",
        "cpu_utilization_aggregate": f"{s['cpu_utilization_aggregate']:.4f}",
        "context_switches_total": s["context_switches_total"],
        "migrations": s["migrations"],
        "jains_fairness": f"{s['jains_fairness']:.6f}",
        "min_max_ratio": f"{s['min_max_ratio']:.6f}",
    }


def run_sweep(
    algorithms: list[str],
    workloads: list[str],
    seeds: list[int],
    *,
    n_processes: int,
    n_cpus: int,
    max_time: int,
    enable_load_balancer: bool,
    verbose: bool = True,
) -> list[TrialResult]:
    trials: list[TrialResult] = []
    total = len(algorithms) * len(workloads) * len(seeds)
    idx = 0
    for wl in workloads:
        for algo in algorithms:
            for seed in seeds:
                idx += 1
                if verbose:
                    print(
                        f"[{idx:>3}/{total}] {algo:<6} on {wl:<12} seed={seed}",
                        end="",
                        flush=True,
                    )
                trial = run_trial(
                    algorithm=algo,
                    workload=wl,
                    seed=seed,
                    n_processes=n_processes,
                    n_cpus=n_cpus,
                    max_time=max_time,
                    enable_load_balancer=enable_load_balancer,
                )
                trials.append(trial)
                if verbose:
                    s = trial.summary
                    print(
                        f"  wait_p99={s['wait_time']['p99']:>8.1f}  "
                        f"resp_p99={s['response_time']['p99']:>8.1f}  "
                        f"jain={s['jains_fairness']:.3f}  "
                        f"({trial.wall_seconds:.2f}s)"
                    )
    return trials


def write_csv(trials: list[TrialResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for t in trials:
            w.writerow(_flatten(t))


def _default_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "csv" / "run_all.csv"


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 -- 6x4xN sweep")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOADS.keys()),
        choices=list(WORKLOADS.keys()),
    )
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds")
    parser.add_argument("--processes", type=int, default=60)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--max-time", type=int, default=200_000)
    parser.add_argument(
        "--dual-core",
        action="store_true",
        help="shorthand for --cpus 2 with load balancer enabled",
    )
    parser.add_argument("--no-load-balancer", action="store_true")
    parser.add_argument("--out", type=Path, default=_default_csv())
    args = parser.parse_args()

    n_cpus = 2 if args.dual_core else args.cpus
    seeds = [20260517 + i for i in range(args.seeds)]
    enable_lb = not args.no_load_balancer

    trials = run_sweep(
        algorithms=args.algorithms,
        workloads=args.workloads,
        seeds=seeds,
        n_processes=args.processes,
        n_cpus=n_cpus,
        max_time=args.max_time,
        enable_load_balancer=enable_lb,
    )
    write_csv(trials, args.out)
    print(f"\nwrote {len(trials)} rows -> {args.out}")


if __name__ == "__main__":
    _cli()
