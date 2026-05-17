"""Phase 8 -- Render the 5 main paper figures.

Inputs:
    results/csv/run_all.csv             -- sweep produced by run_all.py
    results/csv/rbtree_microbench.csv   -- microbench produced by
                                            src/experiments/microbench_rbtree.py

Auxiliary experiments executed inline (small enough to keep co-located):
    * Zijlstra short-vs-long case (Figure 2): one CFS and one EEVDF run on a
      hand-crafted workload where a long task is paired with periodic short
      bursts. We record per-process vruntime over time and plot two curves.
    * Dual-core load-balancing on/off (Figure 5): the same workload is run
      twice on the 2-CPU simulator -- once with the LoadBalancer enabled,
      once without. Per-CPU running-time fractions over time produce the
      utilization curve.

Run::

    python -m src.experiments.make_figures
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from src.algorithms.cfs import CFS
from src.algorithms.eevdf import EEVDF
from src.algorithms.fcfs import FCFS
from src.algorithms.mlfq import MLFQ
from src.algorithms.round_robin import RoundRobin
from src.algorithms.sjf import SJF
from src.core.cpu import LoadBalancer
from src.core.event import Simulator
from src.core.process import IOPattern, Process
from src.core.scheduler_base import SchedulerBase
from src.visualization.comparison import cdf_plot, dual_panel, line_plot
from src.workloads.synthetic import (
    Profile,
    WorkloadConfig,
    cpu_heavy,
    generate_workload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV_DIR = REPO_ROOT / "results" / "csv"
RESULTS_FIG_DIR = REPO_ROOT / "results" / "figures"


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def load_run_all(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate(
    rows: list[dict[str, str]],
    *,
    workloads: Sequence[str],
    metric: str,
) -> dict[str, dict[str, float]]:
    """Return ``out[workload][algorithm] = mean(metric across seeds)``."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["workload"] not in workloads:
            continue
        val = float(r[metric])
        if math.isnan(val):
            continue
        buckets[r["workload"]][r["algorithm"]].append(val)
    out: dict[str, dict[str, float]] = {}
    for wl in workloads:
        out[wl] = {algo: statistics.fmean(vs) for algo, vs in buckets[wl].items() if vs}
    return out


# ---------------------------------------------------------------------------
# Figure 1 -- P99 response time + Jain's fairness, 6 algos x 3 workloads.
# ---------------------------------------------------------------------------


def fig1_p99_and_fairness(rows: list[dict[str, str]], out_path: Path) -> None:
    workloads = ("cpu_heavy", "io_heavy", "mixed")
    p99 = aggregate(rows, workloads=workloads, metric="response_p99")
    fairness = aggregate(rows, workloads=workloads, metric="jains_fairness")
    dual_panel(
        top_values=p99,
        bottom_values=fairness,
        top_title="P99 response time (ticks)",
        bottom_title="Jain's fairness index",
        top_ylabel="P99 response (ticks)",
        bottom_ylabel="J  (1 = perfectly fair)",
        out_path=out_path,
        top_log=True,
    )


# ---------------------------------------------------------------------------
# Figure 2 -- CFS vs EEVDF on a Zijlstra-style short-vs-long case.
# Two long CPU-bound tasks + one short-burst interactive task.
# ---------------------------------------------------------------------------


class VruntimeProbe(SchedulerBase):
    """Decorator: log (now, pid, vruntime) on every dispatch and tick."""

    def __init__(self, inner: SchedulerBase) -> None:
        super().__init__(num_cpus=inner.num_cpus)
        self.inner = inner
        self.samples: list[tuple[int, int, float]] = []

    def on_arrival(self, p: Process, now: int) -> None:
        self.inner.on_arrival(p, now)
        self.samples.append((now, p.pid, p.vruntime))

    def on_unblock(self, p: Process, now: int) -> None:
        self.inner.on_unblock(p, now)
        self.samples.append((now, p.pid, p.vruntime))

    def on_block(self, p: Process, now: int) -> None:
        self.inner.on_block(p, now)
        self.samples.append((now, p.pid, p.vruntime))

    def on_migration_arrival(self, p: Process, target_cpu: int, now: int) -> None:
        self.inner.on_migration_arrival(p, target_cpu, now)

    def on_tick(self, now: int, cpu_id: int, p: Process) -> bool:
        preempt = self.inner.on_tick(now, cpu_id, p)
        self.samples.append((now, p.pid, p.vruntime))
        return preempt

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        return self.inner.pick_next(cpu_id, now)

    def requeue(self, p: Process, cpu_id: int, now: int) -> None:
        self.inner.requeue(p, cpu_id, now)
        self.samples.append((now, p.pid, p.vruntime))

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self.inner.peek_steal_candidate(cpu_id)

    def pop_for_migration(self, p: Process, cpu_id: int) -> None:
        self.inner.pop_for_migration(p, cpu_id)

    def runqueue_size(self, cpu_id: int) -> int:
        return self.inner.runqueue_size(cpu_id)


def _zijlstra_workload() -> list[Process]:
    """Two long CPU tasks (P0, P1) + one short interactive task (P2)."""
    long0 = Process(pid=0, arrival_time=0, burst_time=200, nice_value=0)
    long1 = Process(pid=1, arrival_time=0, burst_time=200, nice_value=0)
    short = Process(
        pid=2,
        arrival_time=0,
        burst_time=120,
        nice_value=-5,
        io_pattern=IOPattern(cpu_burst=2, io_burst=6),
    )
    return [long0, long1, short]


def _series_for_pid(
    samples: list[tuple[int, int, float]], pid: int
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for t, p, v in samples:
        if p == pid:
            xs.append(t)
            ys.append(v)
    return xs, ys


def fig2_cfs_vs_eevdf(out_path: Path) -> None:
    procs_cfs = _zijlstra_workload()
    procs_eevdf = _zijlstra_workload()

    probe_cfs = VruntimeProbe(CFS(num_cpus=1))
    sim_cfs = Simulator(scheduler=probe_cfs, processes=procs_cfs, max_time=2000)
    sim_cfs.run()

    probe_eevdf = VruntimeProbe(EEVDF(num_cpus=1))
    sim_eevdf = Simulator(scheduler=probe_eevdf, processes=procs_eevdf, max_time=2000)
    sim_eevdf.run()

    series: dict[str, tuple[Sequence[float], Sequence[float]]] = {}
    for pid, label in [(2, "short P2"), (0, "long P0"), (1, "long P1")]:
        xs, ys = _series_for_pid(probe_cfs.samples, pid)
        if xs:
            series[f"CFS / {label}"] = (xs, ys)
    for pid, label in [(2, "short P2"), (0, "long P0"), (1, "long P1")]:
        xs, ys = _series_for_pid(probe_eevdf.samples, pid)
        if xs:
            series[f"EEVDF / {label}"] = (xs, ys)

    line_plot(
        series=series,
        title="vruntime trajectory -- CFS vs EEVDF on the Zijlstra short/long case",
        xlabel="simulation tick",
        ylabel="vruntime (weighted virtual time)",
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Figure 3 -- RB-tree microbenchmark (consumes existing CSV if present).
# ---------------------------------------------------------------------------


def fig3_rbtree_microbench(microbench_csv: Path, out_path: Path) -> None:
    if not microbench_csv.exists():
        from src.experiments.microbench_rbtree import run_benchmarks

        run_benchmarks(
            sizes=[1_000, 10_000, 100_000],
            seed=20260517,
            out_csv=microbench_csv,
            out_png=out_path,
        )
        return

    rows = list(csv.DictReader(microbench_csv.open("r")))
    by_op_backend: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_op_backend[r["operation"]][r["backend"]].append(
            (int(r["size"]), float(r["ns_per_op"]))
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = sorted(by_op_backend.keys())
    fig, axes = plt.subplots(1, len(ops), figsize=(5 * len(ops), 4), sharey=True)
    if len(ops) == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10").colors
    styles = {"rbtree": ("o-", palette[0]), "sortedlist": ("s--", palette[1])}

    for ax, op in zip(axes, ops, strict=True):
        for backend, points in by_op_backend[op].items():
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            marker, color = styles.get(backend, ("o-", "gray"))
            ax.plot(xs, ys, marker, color=color, label=backend)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("ns / op")
        ax.set_title(op)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle("Hand-written RBTree vs sortedcontainers")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- CDF of per-process wait time under a "real-trace-like" workload.
# Bitbrains data isn't checked into the repo (see data/bitbrains/.gitkeep), so
# we synthesize a heavy-tailed VM-like workload that mimics the marginal shape
# of GWA-T-12 (long-tail bursts, sparse arrivals).
# ---------------------------------------------------------------------------


def _bitbrains_like(n: int, seed: int) -> list[Process]:
    return generate_workload(
        WorkloadConfig(
            profile=Profile.CPU_HEAVY,
            n_processes=n,
            arrival_rate=0.05,
            burst_mu=math.log(80),
            burst_sigma=1.4,
            io_probability=0.0,
            seed=seed,
        )
    )


def fig4_bitbrains_cdf(out_path: Path) -> None:
    factories = {
        "FCFS": lambda: FCFS(num_cpus=1),
        "SJF": lambda: SJF(num_cpus=1, preemptive=False),
        "SRTF": lambda: SJF(num_cpus=1, preemptive=True),
        "RR": lambda: RoundRobin(num_cpus=1, quantum=4),
        "MLFQ": lambda: MLFQ(num_cpus=1, quanta=(1, 2, 4), boost_interval=100),
        "CFS": lambda: CFS(num_cpus=1),
        "EEVDF": lambda: EEVDF(num_cpus=1),
    }

    series: dict[str, list[float]] = {}
    for name, factory in factories.items():
        procs = _bitbrains_like(80, seed=20260517)
        sim = Simulator(scheduler=factory(), processes=procs, max_time=500_000)
        sim.run()
        series[name] = [p.wait_time for p in procs if p.finish_time is not None]

    cdf_plot(
        series=series,
        title="Wait-time CDF on a Bitbrains-style heavy-tailed workload",
        xlabel="wait time (ticks)",
        out_path=out_path,
        log_x=True,
    )


# ---------------------------------------------------------------------------
# Figure 5 -- Dual-core CPU utilization with/without load balancing.
# ---------------------------------------------------------------------------


class UtilizationProbe(SchedulerBase):
    """Decorator: snapshot per-CPU running pid every tick."""

    def __init__(self, inner: SchedulerBase) -> None:
        super().__init__(num_cpus=inner.num_cpus)
        self.inner = inner
        self.snapshots: list[tuple[int, list[int | None]]] = []
        self._last_seen: list[int | None] = [None] * inner.num_cpus

    def on_arrival(self, p: Process, now: int) -> None:
        self.inner.on_arrival(p, now)

    def on_unblock(self, p: Process, now: int) -> None:
        self.inner.on_unblock(p, now)

    def on_block(self, p: Process, now: int) -> None:
        self.inner.on_block(p, now)

    def on_migration_arrival(self, p: Process, target_cpu: int, now: int) -> None:
        self.inner.on_migration_arrival(p, target_cpu, now)

    def on_tick(self, now: int, cpu_id: int, p: Process) -> bool:
        self._last_seen[cpu_id] = p.pid
        self.snapshots.append((now, list(self._last_seen)))
        return self.inner.on_tick(now, cpu_id, p)

    def pick_next(self, cpu_id: int, now: int) -> Process | None:
        picked = self.inner.pick_next(cpu_id, now)
        if picked is None:
            self._last_seen[cpu_id] = None
        return picked

    def requeue(self, p: Process, cpu_id: int, now: int) -> None:
        self.inner.requeue(p, cpu_id, now)

    def peek_steal_candidate(self, cpu_id: int) -> Process | None:
        return self.inner.peek_steal_candidate(cpu_id)

    def pop_for_migration(self, p: Process, cpu_id: int) -> None:
        self.inner.pop_for_migration(p, cpu_id)

    def runqueue_size(self, cpu_id: int) -> int:
        return self.inner.runqueue_size(cpu_id)


def _utilization_curve(
    snapshots: list[tuple[int, list[int | None]]],
    num_cpus: int,
    bin_size: int,
) -> tuple[list[int], list[list[float]]]:
    if not snapshots:
        return [], [[] for _ in range(num_cpus)]
    max_t = snapshots[-1][0] + 1
    n_bins = max(1, max_t // bin_size)
    busy = [[0] * n_bins for _ in range(num_cpus)]
    total = [[0] * n_bins for _ in range(num_cpus)]
    for t, state in snapshots:
        b = min(n_bins - 1, t // bin_size)
        for c, pid in enumerate(state):
            total[c][b] += 1
            if pid is not None:
                busy[c][b] += 1
    xs = [i * bin_size for i in range(n_bins)]
    out = [
        [(busy[c][i] / total[c][i]) if total[c][i] else 0.0 for i in range(n_bins)]
        for c in range(num_cpus)
    ]
    return xs, out


def fig5_dual_core_lb(out_path: Path) -> None:
    def run(with_lb: bool) -> UtilizationProbe:
        procs = cpu_heavy(40, seed=20260517)
        sched = CFS(num_cpus=2)
        probe = UtilizationProbe(sched)
        lb = LoadBalancer(probe.inner) if with_lb else None
        sim = Simulator(
            scheduler=probe,
            processes=procs,
            num_cpus=2,
            load_balancer=lb,
            max_time=200_000,
        )
        sim.run()
        return probe

    probe_off = run(with_lb=False)
    probe_on = run(with_lb=True)

    def util_means(probe: UtilizationProbe) -> tuple[float, float]:
        _xs, curves = _utilization_curve(probe.snapshots, 2, bin_size=500)
        if not curves[0]:
            return 0.0, 0.0
        return statistics.fmean(curves[0]), statistics.fmean(curves[1])

    off_a, off_b = util_means(probe_off)
    on_a, on_b = util_means(probe_on)
    print(
        f"  fig5 -- LB off: CPU0={off_a:.2%} CPU1={off_b:.2%} | "
        f"LB on: CPU0={on_a:.2%} CPU1={on_b:.2%}"
    )

    xs_off, curves_off = _utilization_curve(probe_off.snapshots, 2, bin_size=500)
    xs_on, curves_on = _utilization_curve(probe_on.snapshots, 2, bin_size=500)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_off, ax_on) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    for ax, xs, curves, title in [
        (ax_off, xs_off, curves_off, "Without load balancing"),
        (ax_on, xs_on, curves_on, "With idle-balance (CFS rightmost steal)"),
    ]:
        ax.plot(xs, curves[0], label="CPU 0", lw=1.8, color="#1f77b4")
        ax.plot(xs, curves[1], label="CPU 1", lw=1.8, color="#ff7f0e")
        ax.set_ylabel("CPU utilization")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right")
    ax_on.set_xlabel("simulation tick")
    fig.suptitle("Dual-core CFS -- utilization over time, idle-balance on/off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 -- render the 5 main figures")
    parser.add_argument(
        "--run-all-csv",
        type=Path,
        default=RESULTS_CSV_DIR / "run_all.csv",
    )
    parser.add_argument(
        "--microbench-csv",
        type=Path,
        default=RESULTS_CSV_DIR / "rbtree_microbench.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS_FIG_DIR)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["1", "2", "3", "4", "5"],
        help="Render only a subset (default: all 5)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only) if args.only else {"1", "2", "3", "4", "5"}

    if "1" in selected:
        if not args.run_all_csv.exists():
            raise SystemExit(
                f"missing {args.run_all_csv} -- run src.experiments.run_all first"
            )
        rows = load_run_all(args.run_all_csv)
        fig1_p99_and_fairness(rows, args.out_dir / "fig1_p99_fairness.png")
        print(f"  fig1 -> {args.out_dir / 'fig1_p99_fairness.png'}")

    if "2" in selected:
        fig2_cfs_vs_eevdf(args.out_dir / "fig2_cfs_vs_eevdf.png")
        print(f"  fig2 -> {args.out_dir / 'fig2_cfs_vs_eevdf.png'}")

    if "3" in selected:
        fig3_rbtree_microbench(
            args.microbench_csv, args.out_dir / "fig3_rbtree_microbench.png"
        )
        print(f"  fig3 -> {args.out_dir / 'fig3_rbtree_microbench.png'}")

    if "4" in selected:
        fig4_bitbrains_cdf(args.out_dir / "fig4_bitbrains_cdf.png")
        print(f"  fig4 -> {args.out_dir / 'fig4_bitbrains_cdf.png'}")

    if "5" in selected:
        fig5_dual_core_lb(args.out_dir / "fig5_dual_core_lb.png")
        print(f"  fig5 -> {args.out_dir / 'fig5_dual_core_lb.png'}")


if __name__ == "__main__":
    _cli()
