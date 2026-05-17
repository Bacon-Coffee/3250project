"""Bitbrains GWA-T-12 trace loader.

CLAUDE.md Phase 7 Task 7.2. GWA-T-12 ships one CSV per VM with a 300 s
sample interval. Per-VM columns (only first / fifth used here):

    Timestamp [ms]; CPU cores; CPU capacity provisioned [MHz];
    CPU usage [MHz]; CPU usage [%]; Memory ...; Disk ...; Network ...

Mapping to the simulator's Process model (paper's Methodology section
will state the caveat that VM-level trace ≠ process-level trace):

    arrival_time = (first_timestamp_ms - global_min_ms) // 1000 // tick_s
    burst_time   = round(sum(pct_i/100 * sample_interval_s) * scale_factor / tick_s)
                   clamped to >= 1
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.core.process import Process


@dataclass(frozen=True, slots=True)
class BitbrainsRow:
    timestamp_ms: int
    cpu_usage_pct: float


def parse_bitbrains_csv(path: Path) -> list[BitbrainsRow]:
    """Parse one VM CSV. Locates columns by header name (column order varies)."""
    rows: list[BitbrainsRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            return rows
        clean = [c.strip().strip('"') for c in header]
        try:
            ts_idx = clean.index("Timestamp [ms]")
            pct_idx = clean.index("CPU usage [%]")
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: missing required column "
                f"('Timestamp [ms]' or 'CPU usage [%]') in header {clean}"
            ) from exc

        for raw in reader:
            if not raw or all(c.strip() == "" for c in raw):
                continue
            rows.append(
                BitbrainsRow(
                    timestamp_ms=int(raw[ts_idx]),
                    cpu_usage_pct=float(raw[pct_idx]),
                )
            )
    return rows


def load_bitbrains_trace(
    directory: Path | str,
    *,
    tick_seconds: int = 60,
    sample_interval_s: int = 300,
    scale_factor: float = 1.0,
    max_vms: int | None = None,
) -> list[Process]:
    """Build one Process per VM CSV under ``directory``.

    Args:
        directory: folder containing per-VM ``*.csv`` files.
        tick_seconds: how many wall seconds one simulator tick represents.
        sample_interval_s: spacing between consecutive samples in a VM CSV
            (GWA-T-12 ships 300 s).
        scale_factor: multiplier on integrated CPU seconds. Useful to shrink
            an hour of trace into a quick smoke test.
        max_vms: stop after loading this many CSV files (sorted by filename).

    Returns:
        Processes sorted by ``arrival_time``, with dense ``pid in [0, len)``.
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)

    files = sorted(directory.glob("*.csv"))
    if max_vms is not None:
        files = files[:max_vms]

    vm_data: list[tuple[str, list[BitbrainsRow]]] = []
    for f in files:
        rows = parse_bitbrains_csv(f)
        if rows:
            vm_data.append((f.name, rows))

    if not vm_data:
        return []

    global_min_ms = min(rows[0].timestamp_ms for _, rows in vm_data)

    processes: list[Process] = []
    for _name, rows in vm_data:
        first_ts = rows[0].timestamp_ms
        arrival_ticks = (first_ts - global_min_ms) // 1000 // tick_seconds

        cpu_seconds = sum(r.cpu_usage_pct / 100.0 * sample_interval_s for r in rows)
        cpu_seconds *= scale_factor
        burst_ticks = max(1, round(cpu_seconds / tick_seconds))

        processes.append(
            Process(
                pid=0,
                arrival_time=int(arrival_ticks),
                burst_time=burst_ticks,
            )
        )

    processes.sort(key=lambda p: p.arrival_time)
    for i, p in enumerate(processes):
        p.pid = i
    return processes
