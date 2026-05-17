"""Tests for Metrics collection + fairness math."""

from __future__ import annotations

import math

import pytest

from src.core.metrics import Metrics, jains_fairness, min_max_ratio, percentile
from src.core.process import Process, ProcessState


def test_percentile_basic() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 50) == 3.0


def test_percentile_interpolates() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    # k = 3 * 0.5 = 1.5 -> halfway between xs[1]=2.0 and xs[2]=3.0
    assert percentile(xs, 50) == 2.5


def test_percentile_empty_returns_nan() -> None:
    assert math.isnan(percentile([], 50))


def test_percentile_single_value() -> None:
    assert percentile([42.0], 99) == 42.0


def test_percentile_bounds_validation() -> None:
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 150)


def test_jains_fairness_perfect() -> None:
    assert jains_fairness([10, 10, 10, 10]) == pytest.approx(1.0)


def test_jains_fairness_skewed() -> None:
    # 4 positive parties with one >> others → significantly less than 1
    assert jains_fairness([1, 1, 1, 100]) < 0.5


def test_jains_fairness_empty_or_all_zero() -> None:
    assert math.isnan(jains_fairness([]))
    assert math.isnan(jains_fairness([0, 0, 0]))


def test_min_max_ratio() -> None:
    assert min_max_ratio([2, 4, 8]) == 4.0
    assert math.isnan(min_max_ratio([5.0]))
    assert math.isnan(min_max_ratio([]))


def test_metrics_validates_num_cpus() -> None:
    with pytest.raises(ValueError):
        Metrics(num_cpus=0)


def test_metrics_initializes_per_cpu_lists() -> None:
    m = Metrics(num_cpus=3)
    assert m.running_ticks == [0, 0, 0]
    assert m.idle_ticks == [0, 0, 0]
    assert m.context_switches == [0, 0, 0]


def test_metrics_record_counters() -> None:
    m = Metrics(num_cpus=2)
    m.record_run(0, 5)
    m.record_run(1, 3)
    m.record_idle(0, 2)
    m.record_context_switch(0)
    m.record_context_switch(0)
    m.record_migration()
    assert m.running_ticks == [5, 3]
    assert m.idle_ticks == [2, 0]
    assert m.context_switches == [2, 0]
    assert m.migrations == 1


def test_metrics_summary_handles_no_completion() -> None:
    m = Metrics(num_cpus=1)
    s = m.summary()
    assert s["completed"] == 0
    assert s["throughput"] == 0.0
    assert math.isnan(s["wait_time"]["p99"])


def test_metrics_summary_computes_utilization() -> None:
    m = Metrics(num_cpus=2)
    m.total_ticks = 10
    m.record_run(0, 8)
    m.record_run(1, 4)
    s = m.summary()
    assert s["cpu_utilization_per_cpu"] == [0.8, 0.4]
    assert s["cpu_utilization_aggregate"] == pytest.approx(0.6)


def test_metrics_summary_with_completed_processes() -> None:
    m = Metrics(num_cpus=1)
    for pid, (arr, burst, start, fin) in enumerate(
        [(0, 5, 0, 5), (1, 3, 5, 8), (2, 2, 8, 10)]
    ):
        p = Process(pid=pid, arrival_time=arr, burst_time=burst)
        p.start_time = start
        p.finish_time = fin
        p.cpu_used = burst
        p.wait_time = start - arr
        p.state = ProcessState.TERMINATED
        m.register(p)
    m.total_ticks = 10
    m.record_run(0, 10)
    s = m.summary()
    assert s["completed"] == 3
    assert s["throughput"] == pytest.approx(0.3)
    assert s["wait_time"]["mean"] == pytest.approx((0 + 4 + 6) / 3)
    assert s["jains_fairness"] > 0
