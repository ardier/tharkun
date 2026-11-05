"""Optional native samples at original source points; none of this is proof evidence."""
import sys

# Optional hints share one conservative resident/transport allowance per native run.
SAMPLE_STORAGE_BYTES = 512 * 1024


def watch(program, points):
    if not points:
        return lambda: []
    monitoring, getframe = sys.monitoring, sys._getframe
    value_type, length, identity, integer_hex = type, len, id, hex
    scalar = {id(int): "int", id(bool): "bool", id(float): "float", id(str): "str"}
    sized = {id(t): t.__name__ for t in (str, bytes, bytearray, list, tuple, dict, set, frozenset)}
    by_line, samples, codes = {}, {}, set()
    stored = 0
    for point in points:
        by_line.setdefault(point["line"], []).append(point)
    functions = {point["function"] for point in points}
    recursion_limit = sys.getrecursionlimit
    owner, tool = "tharkun.invariant.samples", None

    def stop():
        monitoring.set_events(tool, 0)
        for code in codes:
            monitoring.set_local_events(tool, code, 0)

    def visited(code, line):
        nonlocal stored
        if code.co_filename != program:
            return
        frame = getframe(1)
        depth, ancestor = 0, frame
        while ancestor is not None:
            depth += 1
            ancestor = ancestor.f_back
        # Sampling must stop before its own Python frames consume the target's recursion budget.
        if depth + 8 >= recursion_limit():
            stop()
            return
        if line not in by_line:
            return
        local, global_ = frame.f_locals, frame.f_globals
        values, storage = {}, 256
        requested = {name for point in by_line[line] for name in point["names"]}
        for name in requested:
            if name not in local and name not in global_:
                continue
            value = local.get(name, global_.get(name))
            # Identity avoids invoking a custom metaclass's __hash__ or __eq__.
            kind = identity(value_type(value))
            scalar_kind, sized_kind = scalar.get(kind), sized.get(kind)
            if scalar_kind is None and sized_kind is None and value is not None:
                continue
            # Charge before copying/encoding a value, including very large ints/strings.
            storage += 384 + 12 * length(name)
            if scalar_kind == "int":
                storage += value.bit_length()
            elif scalar_kind == "str":
                storage += 12 * length(value)
            if stored + storage > SAMPLE_STORAGE_BYTES:
                stop()
                return
            item = {"kind": scalar_kind or sized_kind or "NoneType"}
            if scalar_kind is not None:
                item["value"] = (value.hex() if scalar_kind == "float" else
                                 integer_hex(value) if scalar_kind == "int" and value.bit_length() > 63 else value)
            if sized_kind is not None:
                item["length"] = length(value)
            values[name] = item
        for point in by_line[line]:
            if point["function"] == code.co_qualname.replace(".<locals>.", "."):
                if stored + storage > SAMPLE_STORAGE_BYTES:
                    stop()
                    return
                key = (point["line"], point["column"], point["function"], point["type"])
                samples.setdefault(key, {"site": point, "samples": []})["samples"].append(values)
                stored += storage

    def started(code, offset):
        if code.co_filename == program:
            # Arm the compiled source once; no entry callback belongs on unrelated recursive calls.
            monitoring.set_events(tool, 0)
            pending, code_type = [code], value_type(code)
            while pending:
                current = pending.pop()
                if current.co_qualname.replace(".<locals>.", ".") in functions:
                    codes.add(current)
                    monitoring.set_local_events(tool, current, monitoring.events.LINE)
                pending.extend(value for value in current.co_consts if value_type(value) is code_type)

    for candidate in range(5, -1, -1):
        try:
            monitoring.use_tool_id(candidate, owner)
            tool = candidate
            break
        except ValueError:
            continue
    if tool is None:
        return lambda: []
    monitoring.register_callback(tool, monitoring.events.LINE, visited)
    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    monitoring.set_events(tool, monitoring.events.PY_START)

    def finish():
        stop()
        monitoring.register_callback(tool, monitoring.events.LINE, None)
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)
        return list(samples.values())

    return finish
