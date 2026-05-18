"""Phase 10: A1 poster generator.

Composes the five main figures produced by ``make_figures.py`` into a
single portrait-A1 PNG (594x841 mm at 150 dpi) for the course poster
session. The poster reuses the figures unchanged; only headings,
abstract text, and verdict bullets are rendered by matplotlib here.

Run:
    python -m src.experiments.make_poster
Output:
    results/figures/poster_a1.png
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "results" / "figures"
OUT_PATH = FIG_DIR / "poster_a1.png"

# A1 portrait in inches (594 mm x 841 mm).
A1_W_IN = 594 / 25.4
A1_H_IN = 841 / 25.4
DPI = 150

TITLE = "From FCFS to EEVDF"
SUBTITLE = (
    "An Empirical Reconstruction of the Linux Scheduler Evolution\n"
    "with a Hand-Written Red-Black Tree and Dual-Core Load Balancing"
)
AUTHOR_LINE = "Course Project, CS 3250  |  Spring 2026"

MOTIVATION = (
    "Linux replaced CFS with EEVDF in v6.6 (Oct 2023), ending a 16-year "
    "reign that began with Ingo Molnar's CFS in v2.6.23 (2007). Textbooks "
    "rarely connect classical algorithms (FCFS, SJF, RR, MLFQ) to the "
    "production reality of CFS, let alone the freshly merged EEVDF. We "
    "rebuild the entire path in a single event-driven Python simulator "
    "that shares one hand-written red-black tree between CFS and EEVDF "
    "and runs on a 2-CPU runqueue with a Linux-style idle-balance load "
    "balancer."
)

HYPOTHESES = [
    ("H1  Fairness", "CFS / EEVDF dominate RR / MLFQ on Jain's index."),
    ("H2  Overhead", "On short jobs, RB-tree cost lets SJF/SRTF overtake CFS."),
    ("H3  Latency", "EEVDF's lag-based wakeup beats CFS on P99 response."),
    ("H4  Multi-core", "Idle-balance recovers near-100% on starved CPU1."),
]

VERDICTS = [
    ("H1", "PARTIAL", "Aggregate Jain's is uniform; the win is in windowed lag."),
    ("H2", "NOT OBSERVED", "Crossover not reached at tested scale (<=500 tasks)."),
    ("H3", "SUPPORTED", "EEVDF cuts mean wait ~2% on nice-mixed load vs CFS."),
    ("H4", "SUPPORTED", "CPU1: 0% -> 97.3% with idle-balance enabled."),
]

METHOD_BULLETS = [
    "Six schedulers behind one ABC interface "
    "(on_arrival / on_tick / pick_next / peek_steal_candidate).",
    "Hand-written CLRS red-black tree (rotations, double-red and "
    "double-black fixups) shared by CFS and EEVDF; key differs only.",
    "CFS uses vruntime; EEVDF uses virtual_deadline with O(n) "
    "eligibility filter (lag >= 0).",
    "Dual-CPU model with idle-balance: empty runqueue triggers steal of "
    "rightmost candidate; +1 tick migration cost.",
    "Four workloads x three seeds: CPU-heavy, I/O-heavy, mixed, "
    "nice-weighted; plus Bitbrains GWA-T-12 trace.",
]

FIGURES = [
    ("fig1_p99_fairness.png",
     "Fig 1.  P99 response time and Jain's Fairness Index across six "
     "schedulers and three synthetic workloads."),
    ("fig2_cfs_vs_eevdf.png",
     "Fig 2.  CFS vs EEVDF on the Zijlstra short-task-vs-long-task case: "
     "vruntime trajectories and per-task waiting time."),
    ("fig3_rbtree_microbench.png",
     "Fig 3.  Microbenchmark: hand-written RB tree vs sortedcontainers "
     "at 1k / 10k / 100k insertions and deletions."),
    ("fig4_bitbrains_cdf.png",
     "Fig 4.  Waiting-time CDFs on the Bitbrains GWA-T-12 production "
     "trace, six schedulers overlaid."),
    ("fig5_dual_core_lb.png",
     "Fig 5.  Per-CPU utilization over time, two CPUs, all tasks "
     "initially targeted at CPU0. Top: no balancer. Bottom: idle-balance "
     "enabled."),
]


def _header(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.05), 0.99, 0.9,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=0, facecolor="#0b2545", transform=ax.transAxes,
        )
    )
    ax.text(0.5, 0.78, TITLE, transform=ax.transAxes,
            ha="center", va="center", color="#ffd166",
            fontsize=58, fontweight="bold")
    ax.text(0.5, 0.42, SUBTITLE, transform=ax.transAxes,
            ha="center", va="center", color="white",
            fontsize=24, linespacing=1.25)
    ax.text(0.5, 0.13, AUTHOR_LINE, transform=ax.transAxes,
            ha="center", va="center", color="#a3c9ff",
            fontsize=18, style="italic")


def _section_title(ax: plt.Axes, text: str) -> None:
    ax.text(0.0, 1.0, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold",
            color="#0b2545")
    ax.plot([0.0, 1.0], [0.93, 0.93], transform=ax.transAxes,
            color="#0b2545", linewidth=2)


def _motivation_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    _section_title(ax, "Motivation")
    ax.text(0.0, 0.88, _wrap(MOTIVATION, 62), transform=ax.transAxes,
            ha="left", va="top", fontsize=13, linespacing=1.4)
    # Stack hypotheses as one block anchored at bottom of panel.
    hyp_text = "Hypotheses\n" + "\n".join(
        f"  {label}:   {b}" for label, b in HYPOTHESES
    )
    ax.text(0.0, 0.0, hyp_text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12,
            color="#222", linespacing=1.55)


def _method_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    _section_title(ax, "Methodology")
    y = 0.86
    for bullet in METHOD_BULLETS:
        wrapped = _wrap(bullet, 60)
        n_lines = wrapped.count("\n") + 1
        ax.text(0.0, y, "•", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="#d62828", va="top")
        ax.text(0.04, y, wrapped, transform=ax.transAxes,
                fontsize=12, va="top", linespacing=1.35)
        y -= 0.05 * n_lines + 0.03


def _verdict_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    _section_title(ax, "Key Findings")
    colors = {"SUPPORTED": "#2a9d8f", "PARTIAL": "#e9c46a",
              "NOT OBSERVED": "#9d9d9d"}
    # Four verdicts, evenly distributed between y=0.84 and y=0.05.
    n = len(VERDICTS)
    span = 0.84 - 0.05
    step = span / n
    for i, (h, status, body) in enumerate(VERDICTS):
        top = 0.84 - i * step
        ax.text(0.0, top, h, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="#0b2545",
                va="top")
        ax.add_patch(
            FancyBboxPatch(
                (0.10, top - 0.045), 0.38, 0.045,
                boxstyle="round,pad=0.005,rounding_size=0.01",
                linewidth=0, facecolor=colors[status],
                transform=ax.transAxes,
            )
        )
        ax.text(0.29, top - 0.022, status, transform=ax.transAxes,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        ax.text(0.0, top - 0.08, _wrap(body, 48),
                transform=ax.transAxes, fontsize=11, color="#222",
                linespacing=1.4, va="top")


def _figure_panel(ax: plt.Axes, filename: str, caption: str) -> None:
    ax.set_axis_off()
    img_path = FIG_DIR / filename
    if not img_path.exists():
        ax.text(0.5, 0.5, f"[missing: {filename}]",
                ha="center", va="center", fontsize=14, color="red",
                transform=ax.transAxes)
        return
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, -0.04, _wrap(caption, 95), transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color="#333",
            linespacing=1.25)


def _footer(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.7, "Conclusion", transform=ax.transAxes,
            ha="left", va="center", fontsize=20, fontweight="bold",
            color="#0b2545")
    ax.text(0.0, 0.40, _wrap(
            "We reproduce the FCFS -> CFS -> EEVDF arc in one simulator. "
            "EEVDF's advantage is structural (lag bound, per-task latency "
            "hints), not headline-benchmark speed; CFS still wins P99 on "
            "uniform CPU-heavy load. Idle-balance alone recovers ~97 pp "
            "of starved-core utilization at one migration tick per steal.",
            width=200),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, linespacing=1.5)
    ax.text(1.0, 0.0,
            "RB tree: src/core/rbtree.py  |  "
            "Entry point: python -m src.experiments.run_all",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#666", style="italic")


def build_poster() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(A1_W_IN, A1_H_IN), dpi=DPI, facecolor="white")
    gs = GridSpec(
        nrows=6, ncols=6,
        height_ratios=[1.5, 3.4, 3.2, 3.2, 3.5, 1.4],
        hspace=0.55, wspace=0.20,
        left=0.035, right=0.965, top=0.975, bottom=0.025,
        figure=fig,
    )

    _header(fig.add_subplot(gs[0, :]))

    _motivation_panel(fig.add_subplot(gs[1, 0:2]))
    _method_panel(fig.add_subplot(gs[1, 2:4]))
    _verdict_panel(fig.add_subplot(gs[1, 4:6]))

    _figure_panel(fig.add_subplot(gs[2, 0:3]), *FIGURES[0])
    _figure_panel(fig.add_subplot(gs[2, 3:6]), *FIGURES[1])
    _figure_panel(fig.add_subplot(gs[3, 0:3]), *FIGURES[2])
    _figure_panel(fig.add_subplot(gs[3, 3:6]), *FIGURES[3])
    _figure_panel(fig.add_subplot(gs[4, :]), *FIGURES[4])

    _footer(fig.add_subplot(gs[5, :]))

    fig.savefig(OUT_PATH, dpi=DPI, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    out = build_poster()
    print(f"poster written: {out}")
