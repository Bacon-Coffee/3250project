"""Unit tests for the hand-written red-black tree in :mod:`src.core.rbtree`.

CLAUDE.md invariant #1: this tree MUST be implemented from scratch (not
backed by ``sortedcontainers``). These tests pin both the public API used
by CFS/EEVDF and the structural RB invariants from CLRS chapter 13.

Tests are written FIRST under TDD; they should all fail until
``src/core/rbtree.py`` is implemented.
"""

from __future__ import annotations

import random

import pytest
from sortedcontainers import SortedList

from src.core.rbtree import RBTree

# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_empty_tree_has_no_extremes():
    tree = RBTree()
    assert len(tree) == 0
    assert tree.leftmost() is None
    assert tree.rightmost() is None
    assert list(tree.iter_inorder()) == []


def test_single_insert_visible_at_both_ends():
    tree = RBTree()
    tree.insert(5, "a")
    assert len(tree) == 1
    assert tree.leftmost() == (5, "a")
    assert tree.rightmost() == (5, "a")
    assert list(tree.iter_inorder()) == [(5, "a")]


def test_insert_returns_node_handle_used_for_delete():
    """CFS needs O(log n) delete by handle — vruntime advances, we re-insert."""
    tree = RBTree()
    node = tree.insert(42, "task")
    assert node is not None
    tree.delete(node)
    assert len(tree) == 0
    assert tree.leftmost() is None


def test_inorder_iteration_is_sorted():
    tree = RBTree()
    for k in [50, 10, 70, 30, 20, 60, 80, 40]:
        tree.insert(k, f"v{k}")
    keys = [k for k, _ in tree.iter_inorder()]
    assert keys == sorted(keys)
    assert tree.leftmost()[0] == 10
    assert tree.rightmost()[0] == 80


def test_leftmost_is_cached_and_updates_on_insert():
    """CFS's pick_next is hot — leftmost must be O(1), tracked across inserts."""
    tree = RBTree()
    tree.insert(50, "fifty")
    tree.insert(30, "thirty")
    tree.insert(10, "ten")
    assert tree.leftmost() == (10, "ten")
    tree.insert(5, "five")
    assert tree.leftmost() == (5, "five")
    # Inserting a larger key must NOT change leftmost.
    tree.insert(99, "ninetynine")
    assert tree.leftmost() == (5, "five")


def test_pop_leftmost_advances_leftmost():
    tree = RBTree()
    for k in [3, 1, 4, 1, 5, 9, 2, 6]:
        tree.insert(k, k)
    popped_keys = []
    while len(tree) > 0:
        popped_keys.append(tree.pop_leftmost()[0])
    assert popped_keys == sorted([3, 1, 4, 1, 5, 9, 2, 6])
    assert tree.leftmost() is None


def test_rightmost_is_cached_for_load_balancing_steal():
    """LoadBalancer steals the rightmost (highest vruntime, cheapest to migrate)."""
    tree = RBTree()
    for k in [10, 20, 5, 15, 25, 1]:
        tree.insert(k, k)
    assert tree.rightmost() == (25, 25)
    tree.pop_rightmost()
    assert tree.rightmost() == (20, 20)


def test_duplicate_keys_are_allowed_and_all_preserved():
    """Two tasks can have identical vruntime; both must survive insertion."""
    tree = RBTree()
    nodes = [tree.insert(7, f"task-{i}") for i in range(4)]
    assert len(tree) == 4
    in_order = list(tree.iter_inorder())
    assert [k for k, _ in in_order] == [7, 7, 7, 7]
    # Deleting one handle removes exactly one entry.
    tree.delete(nodes[1])
    assert len(tree) == 3


def test_delete_by_handle_does_not_disturb_other_keys():
    tree = RBTree()
    handles = {k: tree.insert(k, k) for k in [50, 30, 70, 20, 40, 60, 80]}
    tree.delete(handles[30])
    keys = [k for k, _ in tree.iter_inorder()]
    assert keys == [20, 40, 50, 60, 70, 80]


# ---------------------------------------------------------------------------
# Structural RB invariants (CLRS chapter 13)
# ---------------------------------------------------------------------------


def test_invariants_hold_after_sequential_insertion():
    tree = RBTree()
    for k in range(1, 32):  # forces a long right-leaning chain that triggers rotations
        tree.insert(k, k)
        tree._assert_rb_invariants()


def test_invariants_hold_after_reverse_insertion():
    tree = RBTree()
    for k in range(31, 0, -1):
        tree.insert(k, k)
        tree._assert_rb_invariants()


def test_invariants_hold_under_random_insert_and_delete():
    rng = random.Random(20260517)
    tree = RBTree()
    handles: list = []
    for _ in range(500):
        if handles and rng.random() < 0.4:
            idx = rng.randrange(len(handles))
            tree.delete(handles.pop(idx))
        else:
            handles.append(tree.insert(rng.randrange(10_000), None))
        tree._assert_rb_invariants()
    assert len(tree) == len(handles)


# ---------------------------------------------------------------------------
# Fuzz test against sortedcontainers (validation criterion #3 from CLAUDE.md)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fuzz_matches_sortedcontainers_10k_ops():
    """CLAUDE.md validation #3: 10k random insert/delete, final inorder
    sequence must match :class:`sortedcontainers.SortedList` exactly.
    """
    rng = random.Random(424242)
    tree = RBTree()
    oracle: SortedList = SortedList()
    handles: list[tuple[int, object]] = []  # (key, node_handle) for delete by index

    for _ in range(10_000):
        if handles and rng.random() < 0.45:
            idx = rng.randrange(len(handles))
            key, node = handles.pop(idx)
            tree.delete(node)
            oracle.remove(key)
        else:
            key = rng.randrange(100_000)
            node = tree.insert(key, None)
            oracle.add(key)
            handles.append((key, node))

    tree._assert_rb_invariants()
    tree_keys = [k for k, _ in tree.iter_inorder()]
    assert tree_keys == list(oracle)
    assert len(tree) == len(oracle)
    if oracle:
        assert tree.leftmost()[0] == oracle[0]
        assert tree.rightmost()[0] == oracle[-1]


# ---------------------------------------------------------------------------
# CFS use-case smoke test: re-insertion after vruntime advance
# ---------------------------------------------------------------------------


def test_reinsert_after_key_change_simulates_vruntime_advance():
    """CFS pattern: pick leftmost, run, vruntime += delta, re-insert."""
    tree = RBTree()
    handles = {pid: tree.insert(0, pid) for pid in range(5)}
    schedule_log: list[int] = []
    for _ in range(100):
        key, pid = tree.leftmost()
        schedule_log.append(pid)
        tree.delete(handles[pid])
        handles[pid] = tree.insert(key + 7, pid)
        tree._assert_rb_invariants()
    # Each of the 5 tasks should have run exactly 20 times (perfect round-robin
    # under equal weights — the whole point of CFS).
    counts = {pid: schedule_log.count(pid) for pid in range(5)}
    assert all(c == 20 for c in counts.values()), counts
