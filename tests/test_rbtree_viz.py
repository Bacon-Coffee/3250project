"""Tests for the red-black tree Graphviz exporter.

These are intentionally narrow: they pin the DOT string output (which
needs no external binary) and skip actual PNG rendering unless the
``dot`` executable is on PATH.
"""

from __future__ import annotations

import shutil

import pytest

from src.core.rbtree import RBTree
from src.visualization.rbtree_viz import (
    render_insertion_sequence,
    render_tree,
    to_dot,
)


def test_empty_tree_dot_marks_empty():
    dot = to_dot(RBTree())
    assert dot.startswith("digraph RBTree")
    assert "(empty)" in dot


def test_single_node_dot_shows_key_and_black_color():
    tree = RBTree()
    tree.insert(42, "answer")
    dot = to_dot(tree)
    assert 'label="42"' in dot
    # A single inserted node becomes the BLACK root after fixup.
    assert "fillcolor=black" in dot
    # Both children of the root are NIL leaves.
    assert dot.count("NIL") == 2


def test_two_node_dot_emits_parent_to_child_edge():
    tree = RBTree()
    tree.insert(10, None)
    tree.insert(5, None)
    dot = to_dot(tree)
    assert 'label="10"' in dot
    assert 'label="5"' in dot
    # The smaller key is a RED child of the BLACK root.
    assert "firebrick2" in dot
    # Exactly one inter-node edge plus NIL-leaf edges; check at least one "n... -> n..." pair.
    inter_node_edges = [
        line for line in dot.splitlines() if "->" in line and "nil_" not in line
    ]
    assert len(inter_node_edges) == 1, dot


def test_title_is_embedded_when_provided():
    dot = to_dot(RBTree(), title="step 0")
    assert 'label="step 0"' in dot


# ---- render to PNG only if the system has the graphviz binary ----


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` binary not on PATH")
def test_render_tree_writes_file(tmp_path):
    tree = RBTree()
    for k in [10, 20, 5, 15, 25]:
        tree.insert(k, k)
    out = tmp_path / "tree.png"
    written = render_tree(tree, out, title="five-node tree")
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` binary not on PATH")
def test_render_insertion_sequence_writes_one_file_per_step(tmp_path):
    keys = [10, 20, 30, 15, 5]
    paths = render_insertion_sequence(keys, tmp_path)
    assert len(paths) == len(keys)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
