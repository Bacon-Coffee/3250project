"""Graphviz visualizer for the hand-written red-black tree.

Produces the figures used in the paper's Methodology chapter to walk
through CFS's runqueue invariants. Three entry points:

* :func:`to_dot`                   — convert an :class:`RBTree` to a DOT string.
* :func:`render_tree`              — write a single PNG snapshot.
* :func:`render_insertion_sequence` — render N PNGs, one per step,
  showing rotations / recolorings as keys are inserted.

The renderer reaches into private RBTree internals (``_root``, ``_NIL``,
node ``color`` / ``parent`` / ``left`` / ``right``). This module lives in
``src/visualization/`` precisely because that coupling is intentional —
the viewer is part of the same logical unit as the tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.core.rbtree import RED, RBTree, _Node


def _node_id(node: _Node) -> str:
    """Stable graphviz ID derived from Python object identity."""
    return f"n{id(node)}"


def to_dot(tree: RBTree, *, title: str | None = None) -> str:
    """Serialize ``tree`` to a Graphviz DOT string.

    RED nodes are drawn red, BLACK nodes black, and NIL leaves as small
    grey rectangles (matching the textbook style in CLRS).
    """
    nil = tree._NIL
    lines: list[str] = ["digraph RBTree {", "  graph [ranksep=0.45, nodesep=0.30];"]
    if title:
        safe = title.replace('"', r"\"")
        lines.append(f'  labelloc="t"; label="{safe}";')
    lines.append('  node [shape=circle, style=filled, fontname="Helvetica", fontcolor=white];')

    if tree._root is nil:
        lines.append(
            '  empty [label="(empty)", shape=plaintext, fontcolor=black, fillcolor=white];'
        )
        lines.append("}")
        return "\n".join(lines)

    def emit(node: _Node) -> None:
        color = "firebrick2" if node.color == RED else "black"
        label = str(node.key)
        lines.append(f'  {_node_id(node)} [label="{label}", fillcolor={color}];')
        for child, side in ((node.left, "L"), (node.right, "R")):
            if child is nil:
                leaf_id = f"nil_{id(node)}_{side}"
                lines.append(
                    f'  {leaf_id} [label="NIL", shape=box, fillcolor=gray45, '
                    f"width=0.30, height=0.25, fontsize=8];"
                )
                lines.append(f"  {_node_id(node)} -> {leaf_id};")
            else:
                lines.append(f"  {_node_id(node)} -> {_node_id(child)};")
                emit(child)

    emit(tree._root)
    lines.append("}")
    return "\n".join(lines)


def render_tree(
    tree: RBTree,
    out_path: str | Path,
    *,
    title: str | None = None,
    fmt: str = "png",
) -> Path:
    """Write a single rendered image. Requires the ``graphviz`` binary on PATH."""
    import graphviz  # local import keeps the binary dependency optional for tests.

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_src = to_dot(tree, title=title)
    src = graphviz.Source(dot_src, format=fmt)
    rendered = Path(
        src.render(filename=out_path.stem, directory=str(out_path.parent), cleanup=True)
    )
    if rendered != out_path:
        rendered.replace(out_path)
    return out_path


def render_insertion_sequence(
    keys: Iterable,
    out_dir: str | Path,
    *,
    prefix: str = "step",
    fmt: str = "png",
) -> list[Path]:
    """Render one image per insertion step (for the Methodology figure).

    Returns the list of generated file paths in insertion order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = RBTree()
    paths: list[Path] = []
    for i, key in enumerate(keys):
        tree.insert(key, key)
        out = out_dir / f"{prefix}_{i:02d}_insert_{key}.{fmt}"
        render_tree(tree, out, title=f"step {i}: insert {key}", fmt=fmt)
        paths.append(out)
    return paths
