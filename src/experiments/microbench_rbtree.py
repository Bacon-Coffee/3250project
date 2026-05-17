"""Microbenchmark: hand-written RBTree vs ``sortedcontainers.SortedList``.

Background — this experiment generates the Discussion-chapter figure
that quantifies the cost of CLAUDE.md invariant #1 ("the RB tree is
hand-rolled and is the paper's centerpiece"). Two workloads matter for
a CFS-style runqueue:

* **insert**       — bulk arrivals before any scheduling has happened.
* **pop_leftmost** — the steady-state ``pick_next`` cost; this is the
  hot path the kernel cares about most.

Each workload is measured at ``n = 1k / 10k / 100k`` and the per-op
latency is reported in nanoseconds. Results are written to a CSV and
optionally to a PNG line chart.

Run it::

    python -m src.experiments.microbench_rbtree
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

from sortedcontainers import SortedList

from src.core.rbtree import RBTree


def _gen_keys(n: int, rng: random.Random) -> list[int]:
    """Random keys drawn from a range wide enough to keep tie rates low."""
    upper = max(10 * n, 1024)
    return [rng.randrange(upper) for _ in range(n)]


def benchmark_insert(n: int, rng: random.Random) -> dict[str, Any]:
    """Time bulk insertion of ``n`` random keys into both backends."""
    keys = _gen_keys(n, rng)

    tree = RBTree()
    t0 = time.perf_counter_ns()
    for k in keys:
        tree.insert(k, None)
    rb_total = time.perf_counter_ns() - t0

    sl: SortedList = SortedList()
    t0 = time.perf_counter_ns()
    for k in keys:
        sl.add(k)
    sl_total = time.perf_counter_ns() - t0

    return {
        "n": n,
        "operation": "insert",
        "rbtree_ns_per_op": rb_total / n,
        "sortedlist_ns_per_op": sl_total / n,
    }


def benchmark_pop_leftmost(n: int, rng: random.Random) -> dict[str, Any]:
    """Time draining ``n`` elements via leftmost / index-0 access."""
    keys = _gen_keys(n, rng)

    tree = RBTree()
    for k in keys:
        tree.insert(k, None)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        tree.pop_leftmost()
    rb_total = time.perf_counter_ns() - t0

    sl: SortedList = SortedList(keys)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        sl.pop(0)
    sl_total = time.perf_counter_ns() - t0

    return {
        "n": n,
        "operation": "pop_leftmost",
        "rbtree_ns_per_op": rb_total / n,
        "sortedlist_ns_per_op": sl_total / n,
    }


def run_benchmarks(
    sizes: list[int],
    seed: int,
    out_csv: str | Path,
    out_png: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run insert + pop_leftmost at each ``n``, write CSV (and optional plot)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for n in sizes:
        # Different draws per size, deterministic seed.
        rows.append(benchmark_insert(n, random.Random(seed + n)))
        rows.append(benchmark_pop_leftmost(n, random.Random(seed + n + 1)))

    fieldnames = ["size", "operation", "backend", "ns_per_op"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for backend in ("rbtree", "sortedlist"):
                w.writerow(
                    {
                        "size": r["n"],
                        "operation": r["operation"],
                        "backend": backend,
                        "ns_per_op": f"{r[f'{backend}_ns_per_op']:.2f}",
                    }
                )

    if out_png is not None:
        _plot(rows, out_png)

    return rows


def _plot(rows: list[dict[str, Any]], out_png: str | Path) -> None:
    """Render a log-log line chart per operation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = sorted({r["operation"] for r in rows})
    fig, axes = plt.subplots(1, len(ops), figsize=(5 * len(ops), 4), sharey=True)
    if len(ops) == 1:
        axes = [axes]

    for ax, op in zip(axes, ops, strict=True):
        op_rows = sorted([r for r in rows if r["operation"] == op], key=lambda r: r["n"])
        sizes = [r["n"] for r in op_rows]
        ax.plot(
            sizes,
            [r["rbtree_ns_per_op"] for r in op_rows],
            "o-",
            label="hand-written RBTree",
        )
        ax.plot(
            sizes,
            [r["sortedlist_ns_per_op"] for r in op_rows],
            "s--",
            label="sortedcontainers",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("ns / op")
        ax.set_title(op)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle("RBTree microbenchmark — hand-written vs sortedcontainers")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _default_csv_path() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "results" / "csv" / "rbtree_microbench.csv"


def _default_png_path() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "results" / "figures" / "rbtree_microbench.png"


def _cli() -> None:
    parser = argparse.ArgumentParser(description="RBTree vs sortedcontainers microbenchmark")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 100_000],
        help="problem sizes to evaluate (default: 1k 10k 100k)",
    )
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--out-csv", type=Path, default=_default_csv_path())
    parser.add_argument("--out-png", type=Path, default=_default_png_path())
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = run_benchmarks(
        sizes=args.sizes,
        seed=args.seed,
        out_csv=args.out_csv,
        out_png=None if args.no_plot else args.out_png,
    )
    print(f"wrote {len(rows) * 2} rows to {args.out_csv}")
    for r in rows:
        ratio = r["sortedlist_ns_per_op"] / r["rbtree_ns_per_op"]
        print(
            f"  n={r['n']:>7} op={r['operation']:<12} "
            f"rbtree={r['rbtree_ns_per_op']:>9.1f} ns/op  "
            f"sortedlist={r['sortedlist_ns_per_op']:>9.1f} ns/op  "
            f"ratio={ratio:.2f}x"
        )


if __name__ == "__main__":
    _cli()
