"""Tests for CPU bookkeeping + LoadBalancer steal logic."""

from __future__ import annotations

import pytest

from src.core.cpu import CPU, LoadBalancer
from src.core.process import Process
from tests.conftest import FIFOStubScheduler


def test_cpu_idle_when_no_running() -> None:
    cpu = CPU(cpu_id=0)
    assert cpu.is_idle
    cpu.running = Process(pid=1, arrival_time=0, burst_time=1)
    assert not cpu.is_idle


def test_load_balancer_no_peer_busier_returns_none() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    lb = LoadBalancer(sched)
    assert lb.try_steal(0) is None
    assert lb.find_busiest_peer(0) is None


def test_load_balancer_finds_busiest_peer() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    for pid in range(3):
        p = Process(pid=pid, arrival_time=0, burst_time=1)
        sched._queues[1].append(p)
    lb = LoadBalancer(sched)
    assert lb.find_busiest_peer(0) == 1


def test_load_balancer_steals_tail_of_busier_queue() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    procs = [Process(pid=pid, arrival_time=0, burst_time=1) for pid in range(3)]
    for p in procs:
        sched._queues[1].append(p)

    lb = LoadBalancer(sched)
    result = lb.try_steal(0)
    assert result is not None
    stolen, source = result
    # FIFOStub.peek_steal_candidate returns the tail of the FIFO
    assert stolen.pid == 2
    assert source == 1
    assert stolen.last_cpu == 1
    assert sched.runqueue_size(1) == 2


def test_load_balancer_only_steals_when_strictly_busier() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    sched._queues[0].append(Process(pid=0, arrival_time=0, burst_time=1))
    sched._queues[1].append(Process(pid=1, arrival_time=0, burst_time=1))
    lb = LoadBalancer(sched)
    assert lb.try_steal(0) is None


def test_load_balancer_rejects_out_of_range_target() -> None:
    sched = FIFOStubScheduler(num_cpus=2)
    lb = LoadBalancer(sched)
    with pytest.raises(IndexError):
        lb.try_steal(-1)
    with pytest.raises(IndexError):
        lb.try_steal(5)


def test_pop_for_migration_default_raises() -> None:
    class MinimalScheduler(FIFOStubScheduler):
        def pop_for_migration(self, process: Process, cpu_id: int) -> None:
            raise NotImplementedError("explicit opt-out")

    sched = MinimalScheduler(num_cpus=2)
    sched._queues[1].append(Process(pid=0, arrival_time=0, burst_time=1))
    sched._queues[1].append(Process(pid=1, arrival_time=0, burst_time=1))
    lb = LoadBalancer(sched)
    with pytest.raises(NotImplementedError):
        lb.try_steal(0)
