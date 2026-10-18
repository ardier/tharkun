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
    __slots__ = ("m", "vars", "bottom")

    def __init__(self) -> None:
        self.m: dict[tuple[str, str], int] = {}
        self.vars: set[str] = {ZERO}
        self.bottom = False

    # ------------------------------------------------------------------ basics
    def copy(self) -> "Zone":
        z = Zone.__new__(Zone)
        z.m = dict(self.m)
        z.vars = set(self.vars)
        z.bottom = self.bottom
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
        self.vars.discard(x)
        self.m = {k: v for k, v in self.m.items() if k[0] != x and k[1] != x}

    def assign_var(self, x: str, y: str, c: int) -> None:
        """``x := y + c``."""
        if self.bottom:
            return
        if x == y:
            if c == 0:
                return
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

    def assign_interval(self, x: str, lo: int | None, hi: int | None) -> None:
        self.forget(x)
        self.vars.add(x)
        if hi is not None:
            self.add(x, ZERO, hi)
        if lo is not None:
            self.add(ZERO, x, -lo)

    def rename(self, old: str, new: str) -> None:
        if old == new or old not in self.vars:
            return
        self.forget(new)
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
        for k in vs:
            for i in vs:
                dik = 0 if i == k else m.get((i, k))
                if dik is None:
                    continue
                for j in vs:
                    dkj = 0 if k == j else m.get((k, j))
                    if dkj is None:
                        continue
                    nv = dik + dkj
                    if i == j:
                        if nv < 0:
                            self.bottom = True
                            self.m.clear()
                            return
                        continue
                    old = m.get((i, j))
                    if old is None or nv < old:
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
