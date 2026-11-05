"""A zone (difference-bound matrix) over named integer variables.

``m[(a, b)] = c`` means ``a - b <= c``.  The variable ``ZERO`` is the constant 0,
so ``x - ZERO <= c`` is ``x <= c`` and ``ZERO - x <= c`` is ``x >= -c``.  An
absent entry is ``+inf``.  The matrix is kept **closed** (every entry is the
tightest bound derivable by shortest paths) after every operation except
``widen``, whose result the caller closes explicitly.

Integer semantics: every bound is an ``int``; a strict ``a < b`` is ``a - b <= -1``.

Variables are plain strings.  The absint interpreter names them
``<frame>:<local>`` for locals, ``L.<site>`` for a list length, ``$<n>`` for a
statement temporary and ``R.<site>.lo`` / ``R.<site>.hi`` for range bounds.
"""

from __future__ import annotations

ZERO = "0"


class Zone:
    __slots__ = ("m", "vars", "bottom", "affine")

    def __init__(self) -> None:
        self.m: dict[tuple[str, str], int] = {}
        self.vars: set[str] = {ZERO}
        self.bottom = False
        self.affine = None

    def enable_affine(self, clock, deadline):
        from .affine import Affine
        self.affine = Affine(clock, deadline)

    def remember_affine(self, terms, constant):
        if self.affine is not None and not self.bottom:
            self.vars.update(name for name, coefficient in terms if coefficient)
            self.affine.remember(terms, constant)
            if self.affine.inconsistent:
                self.bottom = True
                self.m.clear()

    def _affine_basis(self):
        if self.affine is None:
            return None
        basis = self.affine.copy()
        names = {name for row in basis.rows.values() for name in row if name != ZERO}
        constants = {name for name in self.vars
                     if self.bounds(name)[0] is not None and self.bounds(name)[0] == self.bounds(name)[1]}
        for name in names:
            lo, hi = self.bounds(name)
            if lo is not None and lo == hi and basis.rows.get(name) != {name: 1, ZERO: -lo}:
                basis.remember([(name, 1)], -lo)
            if name not in constants:
                for other in sorted(self.vars - constants - {name}):
                    delta = self.get(name, other)
                    if delta is not None and self.get(other, name) == -delta:
                        basis.remember([(name, 1), (other, -1)], -delta)
                        break
        return basis

    # ------------------------------------------------------------------ basics
    def copy(self) -> "Zone":
        z = Zone.__new__(Zone)
        z.m = dict(self.m)
        z.vars = set(self.vars)
        z.bottom = self.bottom
        z.affine = None if self.affine is None else self.affine.copy()
        return z

    def get(self, a: str, b: str) -> int | None:
        if a == b:
            return 0
        return self.m.get((a, b))

    def bounds(self, x: str) -> tuple[int | None, int | None]:
        """``(lo, hi)`` of ``x``; ``None`` is an infinite bound."""
        hi = self.get(x, ZERO)
        lo = self.get(ZERO, x)
        return (None if lo is None else -lo, hi)

    def entails(self, a: str, b: str, c: int) -> bool:
        """Does every state satisfy ``a - b <= c``?"""
        if self.bottom:
            return True
        v = self.get(a, b)
        return v is not None and v <= c

    def add_var(self, x: str) -> None:
        self.vars.add(x)

    # -------------------------------------------------------------- constraint
    def add(self, a: str, b: str, c: int) -> None:
        """Meet with ``a - b <= c`` and restore closure (O(n^2))."""
        if self.bottom:
            return
        if a == b:
            if c < 0:
                self.bottom = True
            return
        self.vars.add(a)
        self.vars.add(b)
        cur = self.m.get((a, b))
        if cur is not None and cur <= c:
            return
        back = self.m.get((b, a))
        if back is not None and back + c < 0:
            self.bottom = True
            self.m.clear()
            return
        m = self.m
        vs = list(self.vars)
        # i -> a -> b -> j
        into_a = [(i, (0 if i == a else m.get((i, a)))) for i in vs]
        into_a = [(i, d) for i, d in into_a if d is not None]
        from_b = [(j, (0 if j == b else m.get((b, j)))) for j in vs]
        from_b = [(j, d) for j, d in from_b if d is not None]
        for i, dia in into_a:
            base = dia + c
            for j, dbj in from_b:
                nv = base + dbj
                if i == j:
                    if nv < 0:
                        self.bottom = True
                        self.m.clear()
                        return
                    continue
                old = m.get((i, j))
                if old is None or nv < old:
                    m[(i, j)] = nv

    def add_upper(self, x: str, c: int) -> None:
        self.add(x, ZERO, c)

    def add_lower(self, x: str, c: int) -> None:
        self.add(ZERO, x, -c)

    # ------------------------------------------------------------ assignments
    def forget(self, x: str) -> None:
        if x == ZERO:
            raise ValueError("cannot forget the zero variable")
        if x not in self.vars:
            return
        if self.affine is not None:
            self.affine = self._affine_basis()
            self.affine.forget(x)
        self.vars.discard(x)
        self.m = {k: v for k, v in self.m.items() if k[0] != x and k[1] != x}

    def assign_var(self, x: str, y: str, c: int) -> None:
        """``x := y + c``."""
        if self.bottom:
            return
        if x == y:
            if c == 0:
                return
            if self.affine is not None:
                self.affine.shift(x, c)
            nm = {}
            for (a, b), v in self.m.items():
                if a == x:
                    nm[(a, b)] = v + c
                elif b == x:
                    nm[(a, b)] = v - c
                else:
                    nm[(a, b)] = v
            self.m = nm
            return
        self.forget(x)
        self.vars.add(x)
        if y not in self.vars:
            self.vars.add(y)
        self.add(x, y, c)
        self.add(y, x, -c)
        terms = [(x, 1)] if y == ZERO else [(x, 1), (y, -1)]
        self.remember_affine(terms, -c)

    def assign_interval(self, x: str, lo: int | None, hi: int | None) -> None:
        self.forget(x)
        self.vars.add(x)
        if hi is not None:
            self.add(x, ZERO, hi)
        if lo is not None:
            self.add(ZERO, x, -lo)
        if lo is not None and lo == hi:
            self.remember_affine([(x, 1)], -lo)

    def rename(self, old: str, new: str) -> None:
        if old == new or old not in self.vars:
            return
        self.forget(new)
        if self.affine is not None:
            self.affine.rename(old, new)
        self.vars.discard(old)
        self.vars.add(new)
        nm = {}
        for (a, b), v in self.m.items():
            nm[(new if a == old else a, new if b == old else b)] = v
        self.m = nm

    # ---------------------------------------------------------------- lattice
    def close(self) -> None:
        """Full Floyd-Warshall (after a widening)."""
        if self.bottom:
            return
        vs = list(self.vars)
        m = self.m
        # Row indexing avoids tuple-key lookups for every candidate path.
        rows = {i: {i: 0} for i in vs}
        for (i, j), bound in m.items():
            if i in rows and j in rows and i != j:
                rows[i][j] = bound
        for k in vs:
            # A zero diagonal leaves row k unchanged unless a negative cycle ends closure.
            outgoing = tuple(rows[k].items())
            for i in vs:
                row = rows[i]
                dik = row.get(k)
                if dik is None:
                    continue
                for j, dkj in outgoing:
                    nv = dik + dkj
                    if i == j:
                        if nv < 0:
                            self.bottom = True
                            self.m.clear()
                            return
                        continue
                    old = row.get(j)
                    if old is None or nv < old:
                        row[j] = nv
                        m[(i, j)] = nv

    def leq(self, other: "Zone") -> bool:
        """``self`` is included in ``other`` (both closed)."""
        if self.bottom:
            return True
        if other.bottom:
            return False
        for k, c in other.m.items():
            mine = self.m.get(k)
            if mine is None or mine > c:
                return False
        if other.affine is not None and other.affine.rows:
            if self.affine is None or not self._affine_basis().leq(other.affine):
                return False
        return True

    def join(self, other: "Zone") -> "Zone":
        if self.bottom:
            return other.copy()
        if other.bottom:
            return self.copy()
        z = Zone()
        z.vars = self.vars | other.vars
        om = other.m
        z.m = {k: (v if v >= om[k] else om[k]) for k, v in self.m.items() if k in om}
        if self.affine is not None and other.affine is not None:
            z.affine = self._affine_basis().join(other._affine_basis())
        return z

    def widen(self, new: "Zone") -> "Zone":
        """Standard DBM widening: keep the old bounds that ``new`` respects."""
        if self.bottom:
            return new.copy()
        if new.bottom:
            return self.copy()
        z = Zone()
        z.vars = self.vars | new.vars
        nm = new.m
        z.m = {k: v for k, v in self.m.items() if k in nm and nm[k] <= v}
        if self.affine is not None and new.affine is not None:
            z.affine = self._affine_basis().join(new._affine_basis())
        return z

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.bottom:
            return "Zone(bottom)"
        parts = []
        for (a, b), c in sorted(self.m.items()):
            if b == ZERO:
                parts.append(f"{a}<={c}")
            elif a == ZERO:
                parts.append(f"{b}>={-c}")
            else:
                parts.append(f"{a}-{b}<={c}")
        return "Zone(" + ", ".join(parts) + ")"
