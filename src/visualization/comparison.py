"""Cross-algorithm comparison plots driven by the ``run_all.csv`` schema.

Each helper takes already-aggregated data (so the figure script controls the
seed-averaging policy) and writes one PNG to ``out_path``. Matplotlib's
``Agg`` backend is forced so the entire pipeline can run headless.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ALGO_ORDER = ["FCFS", "SJF", "SRTF", "RR", "MLFQ", "CFS", "EEVDF"]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def grouped_bar(
    values: Mapping[str, Mapping[str, float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    algo_order: Sequence[str] = ALGO_ORDER,
    log_y: bool = False,
) -> None:
    """Grouped bar chart -- outer group = workload, inner = algorithm.

    ``values[workload][algorithm] = scalar``. Missing entries are skipped.
    """
    plt = _setup_matplotlib()
    import numpy as np

    workloads = list(values.keys())
    algos = [a for a in algo_order if any(a in values[w] for w in workloads)]

    n_workloads = len(workloads)
    n_algos = len(algos)
    width = 0.8 / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(2 + 1.4 * n_workloads, 4))
    palette = plt.get_cmap("tab10").colors
    x = np.arange(n_workloads)

    for i, algo in enumerate(algos):
        ys = [values[w].get(algo, float("nan")) for w in workloads]
        offset = (i - (n_algos - 1) / 2) * width
        ax.bar(x + offset, ys, width=width, label=algo, color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=min(4, len(algos)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def dual_panel(
    top_values: Mapping[str, Mapping[str, float]],
    bottom_values: Mapping[str, Mapping[str, float]],
    *,
    top_title: str,
    bottom_title: str,
    top_ylabel: str,
    bottom_ylabel: str,
    out_path: Path,
    algo_order: Sequence[str] = ALGO_ORDER,
    top_log: bool = False,
    bottom_log: bool = False,
) -> None:
    """Two stacked grouped-bar panels sharing the X axis (workload groups)."""
    plt = _setup_matplotlib()
    import numpy as np

    workloads = list(top_values.keys())
    algos = [
        a
        for a in algo_order
        if any(a in top_values[w] for w in workloads)
        or any(a in bottom_values[w] for w in workloads)
    ]
    n_workloads = len(workloads)
    n_algos = len(algos)
    width = 0.8 / max(n_algos, 1)
    x = np.arange(n_workloads)
    palette = plt.get_cmap("tab10").colors

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(2 + 1.6 * n_workloads, 7), sharex=True
    )

    for ax, vals, title, ylab, log_y in [
        (ax_top, top_values, top_title, top_ylabel, top_log),
        (ax_bot, bottom_values, bottom_title, bottom_ylabel, bottom_log),
    ]:
        for i, algo in enumerate(algos):
            ys = [vals[w].get(algo, float("nan")) for w in workloads]
            offset = (i - (n_algos - 1) / 2) * width
            ax.bar(x + offset, ys, width=width, label=algo, color=palette[i % len(palette)])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(workloads, rotation=15)
    ax_top.legend(fontsize=8, ncol=min(4, len(algos)), loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def cdf_plot(
    series: Mapping[str, Iterable[float]],
    *,
    title: str,
    xlabel: str,
    out_path: Path,
    log_x: bool = False,
) -> None:
    """Empirical CDF, one line per algorithm.

    ``series[algo] = iterable of per-process values`` (e.g. wait times).
    """
    plt = _setup_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = plt.get_cmap("tab10").colors
    keys = [k for k in ALGO_ORDER if k in series] + [
        k for k in series if k not in ALGO_ORDER
    ]
    for i, algo in enumerate(keys):
        data = np.asarray(list(series[algo]), dtype=float)
        if len(data) == 0:
            continue
        data.sort()
        ys = np.linspace(0, 1, len(data), endpoint=True)
        ax.plot(data, ys, label=algo, color=palette[i % len(palette)], lw=1.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def line_plot(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Generic XY line plot: ``series[label] = (xs, ys)``."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = plt.get_cmap("tab10").colors
    for i, (label, (xs, ys)) in enumerate(series.items()):
        ax.plot(xs, ys, label=label, color=palette[i % len(palette)], lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def stacked_area(
    xs: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Per-CPU utilization over time, one stacked region per CPU."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = plt.get_cmap("tab10").colors
    labels = list(series.keys())
    values = [series[k] for k in labels]
    ax.stackplot(xs, values, labels=labels, colors=palette[: len(labels)], alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
