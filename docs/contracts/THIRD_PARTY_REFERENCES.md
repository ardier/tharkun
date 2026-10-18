# Third-party references consulted ([internal ref])

Before writing a contract, its author cross-checks it against existing Python semantic
models rather than starting from a blank page. **Nothing in `src/tharkun/solve/contracts/`
or `tests/contracts/` is copied from any of these projects.** Each is read for *what
condition to test*, not *what code to write*; the contract's own text and every test are
written independently and verified by running CPython 3.14.4 (`docs/contracts/FORMAT.md`).
No task-specific invariant (a check tied to one benchmark or one property file) is taken
from any of them -- only general stdlib semantics that hold for every caller.

A citation naming one of these tools in a `Contract.source` tuple means "this claim was
cross-checked against it", not "this text originates there" -- the CPython 3.14 docs URL
or CPython source `file::function` next to it is the actual evidence for the claim; the
tool citation is corroboration.

| Project | Role | License | What was read | Commit/tag |
|---|---|---|---|---|
| [CPython](https://github.com/python/cpython) | ground truth | PSF License | `Lib/test/test_{builtin,int,float,long,math,random,list,dict,set,tuple,bytes,str}.py`, `Lib/test/{string,seq,list}_tests.py` -- mined for `assertRaises` cases as extra edge-case inputs, and as a check that this batch's own tests don't miss a documented case | tag `v3.14.4` |
| [Nagini](https://github.com/marcoeilers/nagini) | cross-check | MPL-2.0 | `src/nagini_translation/resources/{list,set_dict,str,bytes,tuple}.sil` -- Viper preconditions on `list.__getitem__`/`__setitem__`/`insert`/`append`/`extend`, `dict.__getitem__`/`keys`/`values` (index-bounds and key-presence conditions) | `44a1a04fbc69dcce90027a3a334c7644c8db701e` |
| [ESBMC](https://github.com/esbmc/esbmc) | cross-check | Apache-2.0 (core; see `COPYING` for third-party solver exceptions -- not implicated here) | `src/python-frontend/models/{builtins,math,random,exceptions,collections,queue,sys}.py` -- operational models' own `raise` sites (e.g. `random.choice`/`sample` on an empty sequence raising `IndexError`, `min`/`max` of `[]` raising `ValueError`) | `b3d49c401d004baa93fc6f4c1eca89a91d32cc21` |
| [Mopsa](https://gitlab.com/mopsa/mopsa-analyzer) | reference only | LGPL-3.0 (the stubs; the analyzer's semantics live in OCaml, also LGPL-3.0) | `share/mopsa/stubs/python/{builtins,math,random,string,stdlib}.py` -- declaration-only stubs (no bodies for the list/dict/set container methods, which Mopsa models natively in OCaml, not in these files); read for which argument-type checks the stub layer itself performs (e.g. `math.sin`/`cos`'s explicit `TypeError` on a non-numeric argument) | `5019c58b6fc84236b338dc64654509b5d8ec53f8` |

## Why these four

`docs/contracts/FORMAT.md`'s own citation rule (a 3.14 docs URL, typeshed, or CPython
source) is necessary but not sufficient evidence that a *raise condition* is complete --
the docs are often prose ("raises IndexError if the index is out of range") that a
contract still has to turn into an exact boundary, and it is easy to miss a class the
docs mention only in passing. Four independent existing models of the same semantics,
built for unrelated verification systems (Viper/Silicon, an SMT-based bounded model
checker, and an abstract-interpretation framework), converging on the same boundary is
strong evidence the boundary is right; a divergence is a reason to re-test on 3.14.4
before trusting either side.

## What was *not* found there

- Nagini's `set_dict.sil` has no `dict.pop`/`dict.setdefault`/`set.discard` methods
  modelled at all (only `__getitem__`, `__setitem__`, `keys`, `values`) -- silence, not
  a "never raises" claim; those contracts here are tested against CPython directly.
- Mopsa's Python stubs contain no `list`/`dict`/`set` file (LGPL OCaml has that
  semantics natively) -- per the ticket's instruction, not fetched or read.
