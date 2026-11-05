# SV task environment model ([internal ref], [internal ref])

## Program-created in-memory streams ([internal ref])

The 2026-09-28 ruling says: “A FALSE witness may therefore rely ONLY on
`_sv_verifier` draws and `argv` -- never on stdin or file CONTENT.” The same model
admits “bytes created earlier in that run”. `io.StringIO()` and `io.BytesIO()`
created and written by the program are internal state: no environmental content
is drawn or replaced, so native replay keeps the ordinary SV witness format.
External world streams and stdin keep their mocked provenance and restrictions.

`solve/world.py::StreamObject` pairs native buffer bytes, cursor, newline and
closed state with exact `FactState` values and `Origin` provenance. A write starts
a new state with new facts; it does not meet the old buffer equality into the new
buffer. `add` refuses facts inconsistent with the program's bytes. Concrete
projections use CPython's native IO methods, including overwrite and sparse seek.
The symbolic executor stores text expressions and cursors in its path environment;
aliases and inlined helpers observe the same stream keys. Read/getvalue results
are expressions derived from program inputs, never new input draws.


The existing `io.BytesIO` factory contract retains its unknown raise obligations;
precise binary buffer facts do not invent a proof for those claims. Opaque bytes
remain top outside the explicitly supported stream, equality and length operations.

## File world objects ([internal ref], owner 2026-10-25)

`solve/world.py::WorldObject` pairs current `concrete` bytes and their line view with
`lattice.py::FactState` and each clause's `Origin`. `WorldObject.add` meets facts,
refuses bottom or an unmaterializable conjunction with `WorldRefused`, and changes
neither bytes nor facts on refusal. Satisfying bytes stay unchanged; otherwise
`materialize` first tries local parsed-field edits from `solve/arith.py::solve_int_guard`,
then uses Z3 strings and minimizes length difference, then changed positions.
Regex membership reuses `solve/condition.py::_generate_from_regex`; its bounded
producer may refuse without claiming a proof of contradiction. Numeric fields use
`int(content[slice])` clauses. Unsupported predicate shapes are refused explicitly.
The materializer's search subset uses UTF-8 text, nonnegative slices/line indices
and decimal integers; each proposal is checked with Python before acceptance.

`WorldStore` is one per run, recorded as an uncertified `world_fact` in its
KnowledgeBase. `environment.py::install_files` and `::confined` accept that store
or its concrete request mapping and serve scratch copies. `WorldRecord.touch`
records content and provenance-bearing facts for world reads, retaining `absent`
for ordinary absent box paths. Host files are never overwritten.
Metadata probes of a world file also record its mock provenance and share the
staged-byte refresh with `model_open`, so a fact refinement changes the size
observed by the next stat as well as the next read. Target writes and descriptor
opens on world files remain explicitly unmodelled, so their
outcomes cannot be promoted to confirmed input rows. Worker snapshots meet the
existing facts transactionally and update the same instances held by KB readers.

This extends the mocked search world, not the 2026-09-28 witness ruling: file
content is still not an SV FALSE witness channel. Mock-dependent input answers
retain [internal ref]'s `obtained via mocking` and `review_required` labels. Recipes,
symbolic stdin and broader object kinds remain later stages.

### Symbolic world reads ([internal ref])

Owner 2026-10-25: "if allowed, we can use concolic stuff in this mode too";
the symbolic side reuses FactState clauses and Z3 strings. In search only,
`sv/_symexec.py::solve_targets` accepts the run's WorldStore and an explicit
per-site snapshot receiver. `sv/_symexec_world.py::WorldReads` binds read-only
`open` to that object's symbolic string through the tested `open.read` contract.
Independent opens share contents; aliases share a cursor. Cursors and closed
flags live in the executor's path environment and merge with branches.
`read`, `readline`, optional concrete sizes, and bounded file iteration project
onto the same string. With exits close handles, including early returns.

Supported Python path predicates become Origin-bearing clauses through
`WorldObject.add`; prefix facts are materialized before parsed-integer facts
using the existing producers. The solver rechecks those produced bytes and
updates the draw model, so a later consumer can refine the file further while
retaining the earlier guards. Only when produced bytes cannot satisfy an
unprojectable relationship does the candidate retain its model's content
equality. This fallback restricts that candidate's future refinements;
another site's proposal starts from the original store. Contradictions and
producer refusals return UNKNOWN without changing the live world. Snapshots
meet into the existing objects only after native fresh-child confirmation.

The search subset is UTF-8 text without carriage returns, default universal
newlines, strict decoding, nonnegative constant slices and decimal parsed
integers. Binary/update/descriptor opens, other encodings, seek/tell, symbolic
read sizes and nested returning with scopes decline explicitly. Iteration uses
the executor's existing LOOP_BOUND. An unknown symbolic path can select the
sole declared world object; multiple objects require a concrete path.
These are search limits, never proof assumptions. `run_exact` receives no world,
and absint's declared/arbitrary/OSError interpretation is unchanged. Concolic
world candidates carry [internal ref] labels and cannot close a Solved site set; ordinary
SV verification never passes a world to this search. File content remains
outside the 2026-09-28 FALSE-witness channel.

## Boxed hybrid search ([internal ref])

`sv/hybrid.py` passes the run's world snapshot only when `Context.box` is true.
`sv/_hybrid_worker.py::_boxed_input` gives each input and refinement replay a fresh
`environment.py::confined` context: target writes, cursors and flags cannot leak
from an earlier input. Undeclared opens discover file objects whose bytes are drawn
from the fuzz buffer. Supplied task files keep their declared bytes. Existing
`driver.py::_world_proposals` projects blocked paths onto read/readline subjects,
accumulates successful integer conversions and guards, and rematerializes with
`WorldObject.add`. Unsupported projections decline; a changed world must run again.
At most eight discoveries/refinements run per input, and 64 seeds are retained.

`sv/_hybrid_draws.py::pack_world` frames a snapshot (content, facts, origins and
binary bytes) beside ordinary nondet bytes in the corpus. The worker's finding is
`{vector, world}`; its identity includes both. Search and deepening carry the
snapshot through `Candidate.detail` into the scheduler's fresh or forked discovery
and independent fresh confirmation. `engine.py::_witness` uses the distinct
`tharkun-mocked-witness/1` format and [internal ref] labels for a touched world; this is an
internal search finding, not an SV witness. `engine.py::replay` refuses that finding
unless `box=True`; boxed findings never certify an obligation or prove absence.
The unboxed request, corpus layout and `sv-py-witness/1` output stay unchanged.


`tharkun.solve.environment` is the shared, stdlib-only contract. `TaskEnvironment.from_task(path)` reads `input_files` and `required_files`, normalizes their real paths, and exposes `files` for child execution. `read_effect(path)` answers `fixed` for a declared file, `generated` after a run writes it, or `unknown` otherwise. For proofs, `unknown` means both arbitrary bytes and any `OSError`; a proof may close a site only if it is safe for every such result. For witnesses, only the fixed bytes of declared files and bytes created earlier in that run are admissible. An undeclared read has no admissible initial bytes and raises `FileNotFoundError`. Writes stay in the child's scratch filesystem, preserve parent-directory failures and expose the requested file name. A later read or metadata check can observe a generated file. A file open blocks execution-only TRUE certification because its path and errors are environmental effects. Directory enumeration and low-level `os.open` remain unmodelled; runs using them cannot produce FALSE witnesses. Since [internal ref] the metadata primitives follow `os.stat`'s model: `environment.py::install_files` installs `model_lstat`/`model_readlink`/`model_access`/`model_statvfs`, one shared `environment.py::stat_like` with `model_stat` (a declared file, and the model's own scratch bookkeeping, answers for real; an undeclared, non-infrastructure path records `file_unmodelled` BEFORE the host call, so neither its success nor its `OSError` can pass as modelled), and patches `posix.open` with the same `model_os_open` (the import machinery's own `.pyc` write exempted, `environment.py::_import_machinery_read`). The harness's OWN path resolution -- `posixpath.realpath`/`pathlib` driven by a frame whose file is under `environment.py::_HARNESS_SOURCE_ROOT` (a worker resolving the program's path, the mutation confinement's containment check) -- answers for real, unflagged: it is mechanical, never target observation (`environment.py::_resolution_probe`). `solve/extern.py::native_import_reason` refuses `import posix`/`import nt`, whose raw primitives bypass the model.

**The box ([internal ref], the hybrid search's mocked world).** `environment.py::install_files(..., box=True)` -- and `environment.py::confined(..., box=True)`, and a request `"box": true` read by `sv/_hybrid_worker.py::_install_environment` -- makes every undeclared path a MOCKED object: `environment.py::stat_like` (installed as `model_stat`/`model_lstat`/`model_readlink`/`model_access`/`model_statvfs`; `os.access` answers `False` where the metadata calls raise `FileNotFoundError`), the box's `os.listdir`/`os.scandir` (`environment.py::box_enumerate`) and `environment.py::model_open`/`model_os_open` answer "absent" with NO host call and NO `file_unmodelled`, and each touch is recorded in `environment.py::WorldRecord` (`environment.py::world_record`; HOME-relative, in touch order; `sv/engine.py::Outcome.world` carries it back from a child, and `sv/_hybrid_worker.py::TestOneInput` resets it per input). The import roots -- the program's own directory, `sys.prefix`/`sys.base_prefix` and every `sys.path` entry, never `/` or an ancestor of HOME or the scratch -- are readable, statable and listable unflagged, and a stat-like probe on an ANCESTOR of a root answers for real (resolving a root walks its ancestors, `posixpath.realpath`). Writes never leave the scratch, boxed or not (`environment.py::model_os_open` still refuses a write outside it, and `environment.py::model_open` still stages it under the scratch); `install_files` points `tempfile.tempdir` into the scratch (`confined` restores it, like every other attribute), and a child started for a boxed run gets the fixed, absent `environment.py::BOX_HOME` as HOME and the XDG dirs (`sv/engine.py::run_once`, `sv/_hybrid_worker.py::_install_environment`), so object state carries no run-varying path. `box=False` -- every SV-COMP run -- is the model above, unchanged.

**Aliases bound before the model are rebound ([internal ref], `environment.py::_rebind_aliases`).** The installers replace each primitive on the module that owns it (`builtins.open`, `io.open`, `os.stat`, `os.remove`, the spawn functions, ...), but a module imported EARLIER that bound the same object under its own name kept the real one: `tokenize`, `bz2` and `tarfile` do `from builtins import open as ...`, and a worker imports `tokenize` (through `inspect`/`linecache`) and `bz2` (through `shutil`) before it installs the model, so `tokenize.open` and `bz2.open` read an undeclared host file unflagged. `install_files` therefore ends by rebinding every module-level global that IS one of the replaced objects to its replacement, in the interpreter's own modules: the stdlib and installed packages under `sys.prefix` / `sys.base_prefix`, where a target's libraries live. The sweep leaves out `posix`/`nt` and the import machinery, which reads through `posix` itself (`environment.py::_import_machinery_read`). It also leaves out the harness (`__main__` and Tharkun's own modules: `sv/_child.py::_RESULT_FILE_IO` keeps the real `FileIO` on purpose to write the result file) and every module elsewhere, such as a test or the program hosting an in-process `confined()`, which keeps what it bound. Not swept: a package installed in editable mode outside the prefixes. Identity only: no target `__eq__`/`__hash__` runs, and a lazy module is not loaded. Not swept: a primitive captured in a class attribute, a default argument or a closure (the stdlib has none); a program that re-imports `posix` and calls it directly is, as below, outside what this protects against. Sockets are not part of this model at all (`ssl.create_connection` is a `socket` alias, not a file one).

**Mutating calls are confined to the scratch (`environment.py::install_mutation_confinement`).** `open` is shadowed (a write outside the scratch lands in a scratch-side shadow file), but the other file-system mutators used to reach the host: a replay, or the Atheris worker that runs the program for real, could `os.remove`, rename, `mkdir`/`rmdir`, `chmod`/`chown`/`utime`/`truncate` or link a file outside the scratch. Now every such `os` call -- `remove unlink rmdir mkdir mkfifo mknod chmod lchmod chown lchown chflags lchflags truncate utime setxattr removexattr chroot rename replace link symlink`, on `os` and on `posix` -- checks each path argument (symbolic links resolved): all under the scratch, it runs for real; otherwise it raises `OSError(EROFS)` like a read-only file system, never a silent success, and sets `file_effect` and `file_unmodelled` (so the run is never a violation or a TRUE). A `dir_fd` argument is refused (a path relative to a descriptor the model cannot see), as is a symbolic or hard link that would lead out of the scratch (a write through it would escape). A descriptor the program holds (opened through the model) passes. Everything built on these -- `os.makedirs`/`removedirs`/`renames`, `shutil.rmtree`/`move`/`copy*`/`copytree`/`chown`, `pathlib.Path.unlink`/`rename`/`mkdir`/`rmdir`/`touch`/`chmod`/`symlink_to`/`hardlink_to`, `tempfile` -- reaches the disk through them and is confined with them (CPython's import machinery expects `EROFS` when it cannot write a `.pyc` and ignores it). Process spawning (all `os` and `posix` primitives: `system`, `popen`, `fork`, `forkpty`, `posix_spawn*`, `exec*`, `spawn*`, `startfile`, plus `subprocess.Popen._execute_child`, `_posixsubprocess.fork_exec`, and `_winapi.CreateProcess`, which transitively cover `multiprocessing`, `concurrent.futures`, `pty`, `asyncio` and `os.popen`) is also confined: every call raises `OSError(EPERM)` and sets the same flags (`environment.py::install_spawn_confinement`). Both confinements are protection against ACCIDENTAL damage by a benchmark program, not a sandbox against a hostile one: a program that removes `posix` or `_posixsubprocess` from `sys.modules` and imports it again gets the original functions back (an SV run never admits that case: `solve/extern.py::native_import_reason` refuses `import posix`/`nt`, [internal ref]), and `ctypes` calls libc directly (measured on this change with a sentinel directory: `sys.modules.pop('posix')` then `posix.system(...)`, `sys.modules.pop('_posixsubprocess')` then a fresh `import subprocess`, and `ctypes.CDLL(None).system(...)` all still run; `importlib.reload` of `os`, `subprocess` or `_posixsubprocess` does not undo the refusal).  `tests/test_sv_fs_confinement.py` has one test per call family against victims outside the scratch, in the child and in the hybrid worker (10 of its 11 tests fail on sv_2027).
**In-process replays: `environment.py::confined` (#1132b).** `install_files` is installed once for the whole life of a child or a worker process, which is what an engine child and the Atheris worker are. A tool that must run the program INSIDE its own process (the Eregion inverter's forward replay: `invert_`/`solve_` load the real project module and call the original function to check every answer) needs the same model for a bounded window and its removal afterwards. `confined(files=(), scratch=None, flags=None)` is that: a context manager that snapshots every attribute `install_files` rewrites (`builtins.open`, `io.open`, `_io.open`, `io.FileIO`, `_io.FileIO`, `os.stat`, `os.lstat`, `os.readlink`, `os.access`, `os.statvfs`, `os.open` AND `posix.open`, `os.listdir`, `os.scandir` and, for every mutator and `symlink`, `os.<name>` AND `posix.<name>`; `environment.py::_patched_attributes` is the list, and a test fails when `install_files` rewrites one it does not name), installs the unchanged model, changes the working directory to the scratch, and puts every attribute and the working directory back in a `finally` BEFORE it removes a scratch it created. Contexts nest by a depth counter: an inner `with` shares the outermost scratch and flags without re-snapshotting, and the last exit restores. It yields a `Confinement(scratch, flags, files)`; `flags` (`file_effect`, `file_unmodelled`) say the confined code touched the environment.

It is PROCESS-GLOBAL by construction, because `builtins`, `os` and the working directory are: while any context is active, every thread in the process is confined and sees the scratch as its cwd. The contract with the caller is therefore that the replay runs with NO concurrent Tharkun work in the process, and that Tharkun's own writers (a witness, a log, an output file) use absolute paths captured BEFORE entry, never a path relative to the cwd. `tests/test_sv_confined.py` holds both halves: a relative write by the caller after the exit goes to the original cwd, and a write from a second thread during the window is confined (a victim file outside the scratch survives, a relative write lands in the scratch). Process spawning is confined with the rest (`environment.py::install_spawn_confinement`): its attribute list `SPAWN_CONFINEMENT_TARGETS` is part of `_patched_attributes` (`environment.py::_spawn_attributes` resolves it, importing `posix`, `_posixsubprocess` and `_winapi` first so the snapshot names them), so `confined()` snapshots and restores it with the rest, and the `is` check in `tests/test_sv_confined.py` covers the new names the same way. `subprocess` binds `_posixsubprocess.fork_exec` as its own `_fork_exec` when it is first imported: a first import INSIDE a confined block would keep the refusing function after the exit, so the last exit puts back every module-level alias of what was installed, not only the patched attributes (`environment.py::_restore`; a module that bound `open` inside the window gets the real one back the same way). The one other user is `solve/loader.py::load_project_module`, which enters it around the real import of a project module and around every function call made through the returned facade.

**Output.** `environment.py::open_effect(call)` classifies one `open(...)` call by its mode alone: `output` for a constant write-only mode (`OUTPUT_MODES`: exactly one of `w`, `a`, `x`, with an optional `b` or `t` on either side), `read` for a constant read mode or no mode (`READ_MODES`), and `unknown` for everything else: an update mode (`r+`, `w+`, `a+`, `x+`: it reads), a mode that is not a constant string, an invalid mode, a file descriptor as the path, `opener=` or `closefd=`, a starred or `**` argument, an argument given twice. What an output open writes can reach the program again only through a later read, and every read is classified on its own, so a write cannot change what the program computes. `proof_blocker` therefore does not stop at an output open and still stops at every read, update mode, descriptor, rebound `open` (`from io import open`, `def open`, a parameter named `open`, ...) and every attribute `open` (`Path.open` takes its mode first, `os.open` takes flags, `io.open` and `builtins.open` are not recognised): a program that writes a file and reads it back is stopped at the read. Two engines consume it beyond the gate ([internal ref], `contracts/rules_fileio.py`): `solve/absint` models `open(..., <output mode>)`, `with` on its handle and the write side of the handle (`write`, `writelines`, `flush`, `close`, `__enter__`, `__exit__`), and `sv/_symexec` runs the program past them; both read the SAME contracts (`open.output`, `io.TextIOWrapper.*`, `io.BufferedWriter.*`), so a raise claim is stated once. Reads stay unknown. `tell`/`seek` are not modelled. This is the one classification for all engines: `determinism.py::check_open` (the census) uses it too, and the stationary proof's own output model (`output_effects.py::output_only`, which also decides whether the handle can be read back, caught or escape) accepts the subset of `OUTPUT_MODES` its probe replays (`output_effects.py::MODES`).

An output open still fails sometimes, and the model keeps that: `environment.py::OUTPUT_OPEN_RAISES` is `OSError` (the environment: a missing parent directory, a directory, a permission; not a property class), `TypeError` (a path that is not `str`, `bytes` or path-like: the `dynamic-typing` class) and `ValueError` (an embedded NUL, an invalid mode; its subclass `UnicodeEncodeError` for a path that does not encode). The gate does not know a path's type and discharges none of these. An analysis that accepts the call must model them or stay open at it: `absint` has no model of `open` or of `with`, so it aborts there (`unsupported: call of an unknown value`) and a program that opens a file gets no proof from it; the census has no table for `TypeError`. Passing the gate is therefore never evidence of safety, only the end of the environment's objection. Measured on the Real-world `_nondet` tasks (48 pairs): `bm_raytrace_nondet` x4 and `bm_chaos_nondet` x4 were stopped by `environment: unknown file read at line 213 / 141` and are now stopped further in: raytrace by `math.tan` (line 252; the math group lands with C2's contracts) and chaos, since [internal ref] models `nondet_float_range` and [internal ref] the iterable-reducing builtins, by "allocation would invalidate an in-flight heap reference" (the borrow design: pin a call's operands, not built); behind them are try/finally and recursion (raytrace) and `%`-formatting and bytes (chaos); no verdict changes.

`argv` is one unbounded list-of-strings draw, used when the source references `argv`, and recorded in the normal witness vector. Other programs start with the admissible real list `[program]`; computed access remains replayable but is not searched over other lists. Its search candidates are samples, never a complete proof domain. `stdin` has no constraint in the SV-Benchmarks Python README. Proofs therefore assume arbitrary input; an execution that observes stdin cannot certify TRUE from an empty stream. FALSE witnesses run against an empty stream, matching the runner's `DEVNULL` convention. `input()` reaches EOF in that witness mode. This asymmetry prevents an invented stream from becoming a FALSE.

**2026-09-28 owner ruling (interim, pending [internal ref]'s admin answer).** Basis: the track README (`sv-benchmarks python/README.md`@33972db) names only `_sv_verifier.nondet_*()` and `sys.argv` as inputs; it has no stdin or file-content channel at all. A FALSE witness may therefore rely ONLY on `_sv_verifier` draws and `argv` -- never on stdin or file CONTENT. Two things changed here as a result, both found by REV's soundness review of PR #38 (repro: `[host]:[local path]`):

* **No stand-in for `sys.stdin`, ever.** A prior dynamic wrapper (`_child._Stdin`) wrapped the real stdin to flag when a program touched it. The wrapper had no `.mode`, was never `sys.__stdin__`, and had the wrong `type()` -- REV's tasks a04/b01/b02/b06 (checking any of those) got a wrong FALSE, since real execution never raises on them. Fixed by removing the wrapper: `sys.stdin` is the real object, always (empty under the forkserver's own `DEVNULL`-inherited fd, matching the FALSE-witness convention above; genuinely arbitrary for a canary/proof run, same as before). "Stdin touched" -- the same TRUE-disqualifying signal as `time`/`unseeded_random_unmodelled` in `flags` -- is now a static, conservative source scan, `environment.references_stdin(source)`: True if the file references `sys.stdin`/`sys.__stdin__` or calls `input()` anywhere, reachable or not. Over-approximating here costs only precision (an avoidable UNKNOWN), never soundness.
* **`sys.orig_argv` is never a witness channel** ([internal ref]; owner, 2026-10-29 audit: "do the thing the competition expects"). The competition's model (sv-benchmarks `python/CONVENTIONS.md`) is "`sys.argv` is assumed to be any list of strings", set IN-PROCESS -- the grader's replayer does exactly that -- and it does not define `sys.orig_argv`. The child keeps the process's real value, which is the VERIFIER's launch line (`[python, .../_child.py, .../req.json]`; a forked worker's differs from a fresh child's). So a FALSE that depends on it cannot be confirmed: code that reads it by ANY spelling -- even `len(sys.orig_argv)` or `orig_argv[1:]` -- sets the `orig_argv` flag (never a witness, `engine.py::NEVER_A_WITNESS_FLAGS`), and a static TRUE is refused (`environment.py::proof_blocker`). Both come from `environment.py::orig_argv_reason`, which reads the CODE only (`ast`; a string counts only where it runs or is used as a name, `environment.py::_live_strings`): it replaced a raw `b"orig_argv" in source` test that a COMMENT tripped and a computed name evaded. Reflection that reaches attributes without naming them is judged by OBJECT where the child can see it (`_child.py::_mapping_observer`, right before the read): `name.__dict__`, `vars(name)`, `getattr(name, "__dict__")` with a module receiver (`sys` included) and a computed `getattr(name, k)` with `sys` itself as receiver set `import_state_unmodelled`; a class's mapping keeps its FALSE (R1), and a `getattr` the program rebound is not the builtin. What it cannot see -- a non-name receiver (`vars(os.sys)`), a resolver used as a value or renamed, `inspect.getmembers`, `gc.get_objects`, `pydoc`, `eval`/`exec`/`compile` of a non-literal -- counts statically.
* **Declared/required file content is never a witness channel either**, on the validator (`py-inv/eregion-sv`) side: a file's real bytes come only from disk next to the task, staged unconditionally for every replay from the task's own `.yml`, never from anything a witness carries (closing REV's f1/f2: an invented undeclared file, and a declared file's content overridden). This was also why a06's witness (a genuine FALSE for a program reading a declared relative file) failed to replay: the validator used to require a witness-carried `files` draw to know what to stage at all, which Tharkun's own witnesses never emitted (Tharkun's exported `vector` was always just its own consumed `_sv_verifier`/`argv` draws -- see `engine._witness`). Making the validator stage declared files unconditionally, the same way Tharkun's own `install_files` already does internally, fixes the replay with no change to Tharkun's witness format at all.

Hooks:

* `_child` executes declared files and generated writes in a scratch directory (`install_files`, driven purely by the task's own `TaskEnvironment`, never by a witness); `engine.Candidates` samples argv lists. `run_once`, forkserver requests and replay carry the task manifest. A program that observes import state is executed by a child whose `sys.modules` is the native startup set (`_child.py::evict_import_cache`, [internal ref]: every pure-Python module not in it and not model-held is dropped before `run_as_main`, so the task's own stdlib shadow is reproduced exactly as `python program.py` sees it). `_probe` (the stationarity proof's observing child) mirrors `_child`'s clock/random/stdin handling via `import _child as C` -- both call `C._patch_clock()` (clock only; no stdin patch) and both static-pre-scan their own source with `references_stdin`/`C.references_stdin` before executing it.
* Static proof returns UNKNOWN on stdin and file-read effects its domain cannot handle. Since [internal ref], absint models read-only world text through `open.read`: exact declared files or arbitrary undeclared text, with unsupported exception paths left UNKNOWN. An output open is classified, not blocked: see Output above. The determinism census still rejects reads.
* `sv/sites` lifts argv access under the main guard into a list parameter. The engine's fresh replay checks every FALSE witness.

The external witness validator (`py-inv/eregion-sv`) supports the `argv` vector entry (PR #26); it defaults to `[task]` when a vector has no argv draw. It never supports `stdin`/`files` vector entries: `eregion_sv.replay.validate` REJECTS any witness whose vector names either (`FORBIDDEN_KINDS`), before running anything, and `eregion_sv.replay.run` always stages the task's declared files (`eregion_sv.tasks.Task.declared_files`, from the `.yml` sitting next to the `.py`) and always replays with `stdin=subprocess.DEVNULL`, never inherited from the validator's own invocation.

## Files and how they relate

- `src/tharkun/solve/environment.py` is the shared, stdlib-only contract (`_child.py` imports it as a top-level module, so it imports nothing from the package): `TaskEnvironment`, `read_effect`, `install_files` (the witness-mode sandbox), `references_stdin`, `references_argv`, `orig_argv_reason`, `proof_blocker`, and the output classification `open_effect` with `OUTPUT_MODES`, `READ_MODES` and `OUTPUT_OPEN_RAISES`.
- `src/tharkun/solve/environment.py::confined` wraps `install_files` (and, through it, `install_mutation_confinement`) in a reversible, nestable context for in-process replays; `src/tharkun/solve/loader.py::load_project_module` and `import_project_isolated` (the project import, #1132b) are its callers, `tests/test_sv_confined.py` and `tests/test_solve_project_loader.py` pin them.
- `src/tharkun/solve/environment.py::preload_infrastructure` ([internal ref], `INFRASTRUCTURE_MODULES`: `sysconfig`, `platform`) imports, before confinement, the one standard-library module whose import-time probe `model_stat` flags (`sysconfig`'s `is_python_build()` stat; `ctypes`, `pydoc`, `trace` and `zoneinfo` flag only through it); it also finishes `sysconfig`'s LAZY preparation -- `preload_infrastructure` calls `sysconfig.get_config_vars()` -- so a target's first `get_config_var` (``zoneinfo``'s module init asks for `TZPATH`) no longer runs `sysconfig._init_config_vars` inside the window, whose `sysconfig._safe_realpath` walk probes the interpreter prefix's ancestors with the now-modelled `os.lstat` ([internal ref]). The Avengers worker calls it; no SV child does, because none has been shown to meet the flag. It exempts no path: `model_stat` is as strict as before (`tests/test_environment_preload.py`).
- `src/tharkun/solve/absint/__init__.py::prove` and `src/tharkun/sv/engine.py` (the static proof stage) consult `proof_blocker` before any proof runs.
- `src/tharkun/solve/determinism.py::check_open` calls `open_effect` (lazily) for the mode check of the determinism census.
- `src/tharkun/solve/output_effects.py::output_only` is the stationary proof's output model; its `MODES` is a subset of `OUTPUT_MODES`, checked by `tests/test_environment_output_open.py`.
- `tests/test_environment_output_open.py` pins the classification, the gate and the consumers; `tests/test_sv_environment.py` pins the rest of the model.

## Avengers and Pynguin boxes ([internal ref])

`avengers/executor.py::IsolatedExecutor` starts confined workers with the fixed
`environment.py::boxed_home` and the box enabled. Class description and argument
generation execute module code too, so their default workers declare their source
and use the same boundary. The existing explicit `unconfined=True` and recorded
per-case replay opt-outs retain their meaning. Supplied `files` remain declared.
`ReplayOutcome.world` carries ordered mock touches through the bounded wire
protocol; `provenance` and `review_required` expose [internal ref]'s label whenever a replay
touched a mock. A successful replay with no mock touch carries neither label.

`avengers/backends.py::PynguinBackendPass` defaults to a boxed worker even with no
explicit files list, declaring the goal source. `box=False` is the explicit legacy
unconfined mode; a portfolio's explicit unconfined executor selects it. Boxed
Pynguin retains the existing refusal of secondary execution and creates generated
outputs in its scratch. No Pynguin code enters the SV default portfolio.

## Stream world objects ([internal ref], manager clarification 2026-10-26)

`world` accepts `{"stdin": {"kind": "stream", "content": "text\n"}}`; other names
of the same kind represent `io` streams. `WorldStore.open_stream(name, binary=)`
materializes a genuine `io.StringIO` or `io.BytesIO` with a fresh cursor, preserving
the world's current bytes, facts and origins. These names are not file paths and
are not staged as files. A supplied concrete stream uses its supplied bytes;
facts refine the same object through the existing `add`/`materialize` contract.
An already-open io cursor retains its current concrete input; open a new cursor
after a refinement. Reconcile live cursor state in stage 1439.

Boxed `engine.py::run_once` feeds the stdin object's bytes to the child's real fd 0.
The forkserver and hybrid worker use `environment.py::feed_stdin` to redirect fd 0
in their disposable executions and clear buffered input between trials. Neither
`sys.stdin` nor `sys.__stdin__` is replaced: native type, identity and `.mode` stay
intact. `sv/_child.py` records conservative source references and fd consumption
by imported functions in the mock touch record, including empty streams named by
the source. Boxed Atheris draws stdin bytes from its buffer and reuses the driver's
unsized `sys.stdin.read`/`readline` path projection; explicit raise sites remain
search hints even when the census refuses stdin. `input()` can consume drawn or
supplied bytes; its newline-stripping projection is not yet modelled.

This follows [internal ref]'s object-identity rule. Ordinary SV still has real stdin fed
from DEVNULL; no stdin or file-content entry enters its witness vocabulary.
A boxed stream finding retains the separate mocked format and [internal ref] review label.
`engine.py::replay` cannot admit it without `box=True`. Mocked world search cannot
certify an obligation or absence. Confirmed snapshots meet back into the run's
existing world objects, preserving identity for the later shared-refinement loop.

Target reads of undeclared data files do not inherit import roots' read exemption:
actual import machinery and interpreter infrastructure still read as before.
Declared task files and supplied world files keep their existing paths. Metadata
and module discovery retain C6's import-root behavior.

Boxed generation also permits `open(fd)` for descriptors opened by the model inside
its scratch, checking the live device/inode before reuse. Host descriptors and
world-file descriptors remain unmodelled. This lets Hypothesis use its own scratch
files without widening the file boundary. Its storage is rooted in the worker's
scratch; trusted tool modules preload before confinement. The Pynguin worker
declares the trusted `pynguin.__main__` source because runpy loads it outside an
ordinary import, while target execution remains boxed.
The Pynguin entry lookup uses a top-level module spec and PathFinder for a package's
main module, so neither module-form nor package-form tool initialization executes
before confinement. The bootstrap refusal test covers the module form as well as
the separate package-form fake tool regression.

Program stream facts use native line boundaries, including binary data that is not
UTF-8. `StreamObject._holds` checks the same facts as `materialize`; a byte buffer
is never validated through the file world's text decoder. Native IO class identities
are captured before target initialization, so rebinding a public `io` constructor
cannot cause target code to run as a trusted projection. Constructor lookup reads
the exact native module dictionary without invoking a target module attribute hook.

When transferring search-only world-read projections, `sv/_symexec_world.py::WorldReads.snapshot`
first keeps the existing materializer's preference for satisfying prior bytes. If
that producer refuses (including a solver timeout), it may reuse the path solver's
concrete model only after `WorldObject.add` validates every prior and incoming
fact natively. A failed validation still refuses, and the caller rechecks the
complete path with the snapshot bytes. This is mocked search state, never a new
SV witness channel. Sequential reads retain their shared cursor.


## Task-owned Python source imports ([internal ref])

When task context declares only the primary program, determinism certification
uses its directory as a program-source root, behind explicit static import roots. `determinism_census`
passes that context to the existing `pyresolve.Resolver`; ambient interpreter
paths are not searched. Flat and package imports, aliases, relative imports
inside packages and literal star exports resolve through the same source cache.
Every reached module and package initializer is recursively censused, including
initializers containing only imports. Imported functions remain live. An import
is not an allow-by-name determinism certificate.

Source module attributes are public, read-only module bindings. Import-location
metadata use or rebinding, observed/mutated or nonliteral __all__, source-root escapes,
compiled competitors, stdlib shadows, cycles and graphs beyond 64 modules refuse.
The harness-provided `_sv_verifier` API retains its existing meaning when a task
contains a local stand-in; it is not a stdlib shadow.
Explicit multi-file manifests remain closed; ordinary resolver clients keep
their previous source boundary. Reading
fixed task source to analyze it does not supply file CONTENT to a FALSE witness;
the existing SV IO ruling and native replay channels remain unchanged.
