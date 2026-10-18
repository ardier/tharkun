"""Fixture for tests/contracts/test_adapters_avengers.py ([internal ref] step 2).

A minimal class whose only boundary call (``pop``) is backed directly by
``heapq.heappop`` -- the ticket's own worked example -- so constructing it
with the default empty-list baseline can never satisfy a goal that wants
``pop()`` to return normally.
"""

from __future__ import annotations

import heapq


class MinHeap:
    def __init__(self, data: list):
        self._data = data
        heapq.heapify(self._data)

    def pop(self):
        return heapq.heappop(self._data)
