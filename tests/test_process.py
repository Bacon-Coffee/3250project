"""Tests for Process state machine and derived statistics."""

from __future__ import annotations

import pytest

from src.core.process import IOPattern, Process, ProcessState


def test_default_state_is_new() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=5)
    assert p.state == ProcessState.NEW
    assert p.cpu_used == 0
    assert p.wait_time == 0
    assert p.start_time is None
    assert p.finish_time is None


def test_invalid_burst_time_rejected() -> None:
    with pytest.raises(ValueError, match="burst_time"):
        Process(pid=1, arrival_time=0, burst_time=0)


def test_invalid_arrival_time_rejected() -> None:
    with pytest.raises(ValueError, match="arrival_time"):
        Process(pid=1, arrival_time=-1, burst_time=5)


def test_nice_value_bounds_enforced() -> None:
    with pytest.raises(ValueError, match="nice_value"):
        Process(pid=1, arrival_time=0, burst_time=5, nice_value=20)
    with pytest.raises(ValueError, match="nice_value"):
        Process(pid=1, arrival_time=0, burst_time=5, nice_value=-21)


def test_response_and_turnaround_undefined_before_run() -> None:
    p = Process(pid=1, arrival_time=10, burst_time=5)
    assert p.response_time is None
    assert p.turnaround_time is None


def test_response_and_turnaround_after_run() -> None:
    p = Process(pid=1, arrival_time=10, burst_time=5)
    p.start_time = 13
    p.finish_time = 20
    assert p.response_time == 3
    assert p.turnaround_time == 10


def test_remaining_clamped_at_zero() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=5)
    p.cpu_used = 100
    assert p.remaining == 0


def test_io_pattern_validates() -> None:
    with pytest.raises(ValueError):
        IOPattern(cpu_burst=0, io_burst=3)
    with pytest.raises(ValueError):
        IOPattern(cpu_burst=2, io_burst=-1)


def test_needs_io_now_only_for_io_bound() -> None:
    cpu_only = Process(pid=1, arrival_time=0, burst_time=10)
    assert not cpu_only.is_io_bound
    assert not cpu_only.needs_io_now()

    io_proc = Process(
        pid=2, arrival_time=0, burst_time=10, io_pattern=IOPattern(cpu_burst=3, io_burst=2)
    )
    assert io_proc.is_io_bound
    assert not io_proc.needs_io_now()
    io_proc.cpu_burst_progress = 3
    assert io_proc.needs_io_now()


def test_wait_time_accumulates_via_ready_markers() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=5)
    p.enter_ready(now=5)
    p.leave_ready(now=8)
    assert p.wait_time == 3

    p.enter_ready(now=10)
    p.leave_ready(now=15)
    assert p.wait_time == 8


def test_leave_ready_without_entering_is_noop() -> None:
    p = Process(pid=1, arrival_time=0, burst_time=5)
    p.leave_ready(now=10)
    assert p.wait_time == 0
