"""Hand-written red-black tree for CFS / EEVDF runqueues.

CLAUDE.md invariant #1: this module MUST NOT delegate to
``sortedcontainers``. Reading and rebuilding the kernel's runqueue data
structure is the methodological centerpiece of the paper.

Algorithm
---------
Classic CLRS (3rd ed., chapter 13) red-black tree with a single
sentinel ``NIL`` leaf, mirroring the intrusive style used in
Linux's ``lib/rbtree.c`` and consumed by ``kernel/sched/fair.c``. Each
:meth:`insert` returns the freshly created node so callers can perform
O(log n) :meth:`delete` later — CFS needs this on every preemption when
``vruntime`` advances and the task must be re-inserted.

Cached extremes
---------------
* ``leftmost``  — O(1); mirrors ``cfs_rq->rb_leftmost`` (see the cached
  variant ``struct rb_root_cached`` introduced in commit ``cd9e6112``).
* ``rightmost`` — O(1); used by :class:`src.core.cpu.LoadBalancer` to
  pick the cheapest steal candidate (CLAUDE.md invariant #3).

Duplicate keys are permitted (two tasks may transiently share
``vruntime``); ties go right so insertion order is preserved among
duplicates during in-order traversal.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

RED = True
BLACK = False


class _Node:
    """A red-black tree node. Public attributes are stable handles returned
    by :meth:`RBTree.insert` and consumed by :meth:`RBTree.delete`.

    Callers MUST NOT reuse a handle after passing it to ``delete`` — the
    underlying node is physically unlinked by the CLRS deletion routine.
    """

    __slots__ = ("color", "key", "left", "parent", "right", "value")

    def __init__(
        self,
        key: Any,
        value: Any,
        color: bool = RED,
        parent: _Node | None = None,
        left: _Node | None = None,
        right: _Node | None = None,
    ) -> None:
        self.key = key
        self.value = value
        self.color = color
        self.parent = parent  # type: ignore[assignment]
        self.left = left  # type: ignore[assignment]
        self.right = right  # type: ignore[assignment]


class RBTree:
    """Order-statistic-free red-black tree keyed on arbitrary comparable keys."""

    def __init__(self) -> None:
        # Single sentinel: all "missing" pointers (NIL leaves AND the root's
        # parent) refer to this one BLACK node. CLRS uses this trick to avoid
        # special-casing NIL throughout the fixup routines.
        nil = _Node(key=None, value=None, color=BLACK)
        nil.parent = nil
        nil.left = nil
        nil.right = nil
        self._NIL = nil
        self._root: _Node = nil
        self._size = 0
        self._leftmost: _Node = nil
        self._rightmost: _Node = nil

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def leftmost(self) -> tuple | None:
        if self._leftmost is self._NIL:
            return None
        return (self._leftmost.key, self._leftmost.value)

    def rightmost(self) -> tuple | None:
        if self._rightmost is self._NIL:
            return None
        return (self._rightmost.key, self._rightmost.value)

    def pop_leftmost(self) -> tuple | None:
        if self._leftmost is self._NIL:
            return None
        node = self._leftmost
        out = (node.key, node.value)
        self.delete(node)
        return out

    def pop_rightmost(self) -> tuple | None:
        if self._rightmost is self._NIL:
            return None
        node = self._rightmost
        out = (node.key, node.value)
        self.delete(node)
        return out

    def iter_inorder(self) -> Iterator[tuple]:
        """Yield (key, value) in non-decreasing key order. O(n) total, O(log n) per step."""
        node = self._leftmost
        nil = self._NIL
        while node is not nil:
            yield (node.key, node.value)
            node = self._successor(node)

    def insert(self, key: Any, value: Any) -> _Node:
        """Insert and return a node handle usable by :meth:`delete`."""
        nil = self._NIL
        z = _Node(key=key, value=value, color=RED, parent=nil, left=nil, right=nil)

        y = nil
        x = self._root
        while x is not nil:
            y = x
            # Ties go RIGHT — preserves insertion order among equal keys when
            # iterating, and lets the leftmost cache use strict less-than.
            x = x.left if key < x.key else x.right
        z.parent = y
        if y is nil:
            self._root = z
        elif key < y.key:
            y.left = z
        else:
            y.right = z

        self._size += 1
        if self._size == 1:
            self._leftmost = z
            self._rightmost = z
        else:
            if key < self._leftmost.key:
                self._leftmost = z
            if key >= self._rightmost.key:
                # Equal keys go right per the BST rule above, so a tied
                # insertion becomes the new rightmost.
                self._rightmost = z

        self._insert_fixup(z)
        return z

    def delete(self, node: _Node) -> None:
        """Remove ``node`` (a handle returned by :meth:`insert`) in O(log n)."""
        nil = self._NIL
        # Refresh cached extremes BEFORE structural changes confuse traversal.
        if node is self._leftmost:
            self._leftmost = self._successor(node)
        if node is self._rightmost:
            self._rightmost = self._predecessor(node)

        z = node
        y = z
        y_original_color = y.color

        if z.left is nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                # x may be nil; we transiently set sentinel.parent so the
                # fixup can walk back up correctly.
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        self._size -= 1
        if self._size == 0:
            self._leftmost = nil
            self._rightmost = nil

        if y_original_color == BLACK:
            self._delete_fixup(x)

    # ------------------------------------------------------------------
    # Internal — rotations & fixups (CLRS pseudo-code, kept close to print)
    # ------------------------------------------------------------------

    def _left_rotate(self, x: _Node) -> None:
        nil = self._NIL
        y = x.right
        x.right = y.left
        if y.left is not nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x: _Node) -> None:
        nil = self._NIL
        y = x.left
        x.left = y.right
        if y.right is not nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is nil:
            self._root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _insert_fixup(self, z: _Node) -> None:
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right  # uncle
                if y.color == RED:
                    # Case 1: recolor and recurse upward.
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        # Case 2: left rotate to reduce to case 3.
                        z = z.parent
                        self._left_rotate(z)
                    # Case 3.
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left  # uncle (mirror)
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self._root.color = BLACK

    def _transplant(self, u: _Node, v: _Node) -> None:
        nil = self._NIL
        if u.parent is nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_fixup(self, x: _Node) -> None:
        while x is not self._root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right  # sibling
                if w.color == RED:
                    # Case 1: rotate so the sibling is BLACK.
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    # Case 2: recolor and shift the deficit upward.
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        # Case 3: rotate sibling to reach case 4.
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    # Case 4: rotate parent, recolor, terminate.
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self._root
            else:
                w = x.parent.left  # mirror
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self._root
        x.color = BLACK

    # ------------------------------------------------------------------
    # Tree traversal helpers
    # ------------------------------------------------------------------

    def _tree_minimum(self, node: _Node) -> _Node:
        nil = self._NIL
        while node.left is not nil:
            node = node.left
        return node

    def _tree_maximum(self, node: _Node) -> _Node:
        nil = self._NIL
        while node.right is not nil:
            node = node.right
        return node

    def _successor(self, node: _Node) -> _Node:
        nil = self._NIL
        if node.right is not nil:
            return self._tree_minimum(node.right)
        parent = node.parent
        while parent is not nil and node is parent.right:
            node = parent
            parent = parent.parent
        return parent

    def _predecessor(self, node: _Node) -> _Node:
        nil = self._NIL
        if node.left is not nil:
            return self._tree_maximum(node.left)
        parent = node.parent
        while parent is not nil and node is parent.left:
            node = parent
            parent = parent.parent
        return parent

    # ------------------------------------------------------------------
    # Debug helper used by tests (and the Methodology figure generator).
    # ------------------------------------------------------------------

    def _assert_rb_invariants(self) -> None:
        """Verify every CLRS invariant. Raises AssertionError if violated.

        Invariants checked:
          1. The sentinel NIL is BLACK.
          2. The root is BLACK.
          3. No RED node has a RED child.
          4. Every root-to-NIL path has the same black-height.
          5. BST ordering holds (left.key <= node.key <= right.key).
          6. Parent pointers are consistent with the actual structure.
          7. Cached leftmost / rightmost match the true tree extremes.
          8. ``len(self)`` matches the number of nodes reachable in-order.
        """
        nil = self._NIL
        assert nil.color == BLACK, "sentinel NIL must be BLACK"

        if self._root is nil:
            assert self._size == 0
            assert self._leftmost is nil and self._rightmost is nil
            return

        assert self._root.color == BLACK, "root must be BLACK"
        assert self._root.parent is nil, "root.parent must be the sentinel"

        def black_height(node: _Node) -> int:
            if node is nil:
                return 1
            if node.color == RED:
                assert node.left.color == BLACK, (
                    f"RED node key={node.key} has RED left child"
                )
                assert node.right.color == BLACK, (
                    f"RED node key={node.key} has RED right child"
                )
            if node.left is not nil:
                assert node.left.parent is node, "left.parent broken"
                assert node.left.key <= node.key, "BST property broken on left"
            if node.right is not nil:
                assert node.right.parent is node, "right.parent broken"
                assert node.right.key >= node.key, "BST property broken on right"
            lh = black_height(node.left)
            rh = black_height(node.right)
            assert lh == rh, (
                f"black-height mismatch at key={node.key}: left={lh} right={rh}"
            )
            return lh + (1 if node.color == BLACK else 0)

        black_height(self._root)

        actual_left = self._tree_minimum(self._root)
        actual_right = self._tree_maximum(self._root)
        assert self._leftmost is actual_left, "leftmost cache stale"
        assert self._rightmost is actual_right, "rightmost cache stale"

        count = sum(1 for _ in self.iter_inorder())
        assert count == self._size, f"size {self._size} != inorder count {count}"
