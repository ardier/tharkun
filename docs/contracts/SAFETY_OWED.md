# Safety templates still owed

- `collections.defaultdict` (`TypeError`, `AttributeError`, `IndexError`, `KeyError`, `ZeroDivisionError`, `FloatingPointError`, `AssertionError`): M1 terms cannot yet represent exact `default_factory` class identity or initializer callback effects.
- `protocol.call` (`AttributeError`, `AssertionError`, `FloatingPointError`, `IndexError`, `KeyError`, `ZeroDivisionError`): Terms cannot express callable-target `tp_call` slots, distinguish class/metaclass call routing, or state constructor callback effects. The binding also cannot describe `*args`/`**kwargs` expansion. `TypeError` cases cover only exact non-callable scalar targets with source-stable receiver/arguments.
- `_sv_verifier.nondet_list` / `_sv_verifier.nondet_dict` (all seven property classes): M1 terms cannot yet represent fresh nondeterministic sequence/map domains or callback-derived element/key/value summaries.
- `_sv_verifier.check_type` (all seven property classes): M1 terms cannot express the exact builtin class identity supplied as the hint.
