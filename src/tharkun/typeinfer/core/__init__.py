"""The type-inference core (docs/TYPEINFER.md): stdlib only, SOLVE layer (docs/LAYERS.md), never runs the target.

``kinds`` is the kind vocabulary (annotation or value to a ``find_inputs`` kind, a kind to a type name or to
``tharkun.strategies`` constraints); ``params`` is ``infer_params`` and its numbered sources; ``usage`` is the
usage rules (source 3), which ``solve/condition.py::infer_var_type`` also runs; ``shape`` recovers the structure a
parameter's uses imply.
"""
