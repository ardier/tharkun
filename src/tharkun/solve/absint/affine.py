"""Private exact affine equalities, alongside the integer difference domain.

Rows are rational coefficient vectors for equations equal to zero. ZERO is
the constant coordinate (one), never a pivot of a consistent equality state.
Projection eliminates a variable; join intersects row spaces, so it is the
affine hull of both input states. No concrete integers are enumerated.
"""
from __future__ import annotations

from fractions import Fraction

from .errors import Unsupported
from .zone import ZERO


def _key(name):
    return name == ZERO, name


class Affine:
    __slots__ = ('rows', 'clock', 'deadline', 'inconsistent')

    def __init__(self, clock, deadline):
        self.rows = {}
        self.clock = clock
        self.deadline = deadline
        self.inconsistent = False

    def _tick(self):
        if self.clock() > self.deadline:
            raise Unsupported('affine equality time budget exhausted')

    def copy(self):
        result = Affine(self.clock, self.deadline)
        result.rows = {pivot: dict(row) for pivot, row in self.rows.items()}
        result.inconsistent = self.inconsistent
        return result

    def _subtract(self, row, other, factor):
        self._tick()
        for name, coefficient in other.items():
            value = row.get(name, 0) - factor * coefficient
            if value:
                row[name] = value
            else:
                row.pop(name, None)

    def _reduce(self, row, basis):
        row = dict(row)
        for pivot in sorted(basis, key=_key):
            if pivot in row:
                self._subtract(row, basis[pivot], row[pivot])
        return row

    def _insert(self, row, basis):
        row = self._reduce(row, basis)
        if not row:
            return
        pivot = min(row, key=_key)
        divisor = row[pivot]
        row = {name: value / divisor for name, value in row.items()}
        for old in basis.values():
            if pivot in old:
                self._subtract(old, row, old[pivot])
        basis[pivot] = row

    def _rebuild(self, rows):
        self.rows = {}
        for row in rows:
            self._tick()
            self._insert(row, self.rows)
        self.inconsistent = ZERO in self.rows

    def remember(self, terms, constant):
        self._tick()
        row = {name: Fraction(value) for name, value in terms if value and name != ZERO}
        if constant:
            row[ZERO] = Fraction(constant)
        self._insert(row, self.rows)
        self.inconsistent = ZERO in self.rows

    def reduce(self, terms, constant):
        self._tick()
        row = {name: Fraction(value) for name, value in terms if value and name != ZERO}
        if constant:
            row[ZERO] = Fraction(constant)
        row = self._reduce(row, self.rows)
        if any(value.denominator != 1 for value in row.values()):
            return None
        return ({name: int(value) for name, value in row.items() if name != ZERO}, int(row.get(ZERO, 0)))

    def forget(self, name):
        self._tick()
        eliminating = next((row for row in self.rows.values() if name in row), None)
        if eliminating is None:
            return
        rows = []
        for original in self.rows.values():
            if original is eliminating:
                continue
            row = dict(original)
            if name in row:
                self._subtract(row, eliminating, row[name] / eliminating[name])
            rows.append(row)
        self._rebuild(rows)

    def shift(self, name, amount):
        rows = []
        for original in self.rows.values():
            self._tick()
            row = dict(original)
            constant = row.get(ZERO, 0) - row.get(name, 0) * amount
            if constant:
                row[ZERO] = constant
            else:
                row.pop(ZERO, None)
            rows.append(row)
        self._rebuild(rows)

    def rename(self, old, new):
        self.forget(new)
        rows = []
        for original in self.rows.values():
            row = dict(original)
            if old in row:
                row[new] = row.pop(old)
            rows.append(row)
        self._rebuild(rows)

    def leq(self, other):
        return all(not self._reduce(row, self.rows) for row in other.rows.values())

    def join(self, other):
        # Carry the left-space component through elimination of the union.
        # Each dependent right row yields one vector in both row spaces.
        basis, labels, common = {}, {}, []
        for side, rows in ((True, self.rows), (False, other.rows)):
            for original in rows.values():
                self._tick()
                row = dict(original)
                label = dict(original) if side else {}
                for pivot in sorted(basis, key=_key):
                    if pivot in row:
                        factor = row[pivot]
                        self._subtract(row, basis[pivot], factor)
                        self._subtract(label, labels[pivot], factor)
                if not row:
                    if label:
                        common.append(label)
                    continue
                pivot = min(row, key=_key)
                divisor = row[pivot]
                basis[pivot] = {name: value / divisor for name, value in row.items()}
                labels[pivot] = {name: value / divisor for name, value in label.items()}
        result = Affine(self.clock, self.deadline)
        result._rebuild(common)
        return result
