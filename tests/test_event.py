"""Tests for the Event / EventQueue primitives."""

from __future__ import annotations

import pytest

from src.core.event import Event, EventQueue, EventType
from src.core.process import Process


def test_event_negative_time_rejected() -> None:
    with pytest.raises(ValueError):
        Event(time=-1, event_type=EventType.PROCESS_ARRIVAL)


def test_event_queue_pops_in_time_order() -> None:
    q = EventQueue()
    q.push(Event(time=5, event_type=EventType.PROCESS_ARRIVAL))
    q.push(Event(time=1, event_type=EventType.PROCESS_ARRIVAL))
    q.push(Event(time=3, event_type=EventType.PROCESS_ARRIVAL))

    times = [q.pop().time for _ in range(3)]
    assert times == [1, 3, 5]
    assert not q


def test_event_queue_stable_on_equal_time() -> None:
    q = EventQueue()
    procs = [Process(pid=i, arrival_time=0, burst_time=1) for i in range(5)]
    for p in procs:
        q.push(Event(time=10, event_type=EventType.PROCESS_ARRIVAL, process=p))

    popped = [q.pop().process.pid for _ in range(5)]
    assert popped == [0, 1, 2, 3, 4]


def test_event_queue_peek_does_not_remove() -> None:
    q = EventQueue()
    q.push(Event(time=7, event_type=EventType.PROCESS_ARRIVAL))
    first = q.peek()
    assert first is not None
    assert first.time == 7
    assert len(q) == 1
    assert q.pop() is first


def test_event_queue_empty_peek() -> None:
    q = EventQueue()
    assert q.peek() is None
    assert not q
    assert len(q) == 0
