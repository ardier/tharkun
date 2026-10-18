"""Fresh-process replay for constructor and boundary-call recipes."""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Sequence

from tharkun.avengers._wire import MAX_RESPONSE_BYTES, decode_response
from tharkun.avengers.model import ConstructionRecipe, ObjectGoal, ReplayOutcome


@dataclass(frozen=True)
class IsolatedExecutor:
    """Replay each proposal in a child interpreter.

    The parent owns search and evidence. A native crash, hard exit, or hung
    target therefore becomes an outcome and cannot take the portfolio down.
    """

    timeout_s: float = 3.0
    trace: bool = False
    # Which Python runs the child.  ``None`` means this one.  Naming another
    # lets the parent analyse a target whose dependencies only that
    # interpreter has -- one benchmark subject per environment -- and that
    # interpreter must therefore have Tharkun installed itself.
    interpreter: str | None = None
    files: tuple[str, ...] | None = None

    def confine(self, flags=None):
        """The same declared-file boundary for target work in the parent."""
        from contextlib import nullcontext
        from tharkun.solve.environment import confined
        from tharkun.solve.extern import native_import_reason

        if self.files:
            with open(self.files[0], "rb") as stream:
                reason = native_import_reason(stream.read(), self.files[0], reversible=True)
            if reason:
                raise ValueError(reason)
        return nullcontext() if self.files is None else confined(self.files, flags=flags, threads=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def replay(self, goal: ObjectGoal, recipe: ConstructionRecipe) -> ReplayOutcome:
        response = self._invoke("replay", goal, recipe, {"trace": self.trace})
        return response if isinstance(response, ReplayOutcome) else ReplayOutcome(
            "worker-error", phase="worker", detail="invalid replay response"
        )

    def replay_many(
        self, items: Sequence[tuple[ObjectGoal, ConstructionRecipe]]
    ) -> tuple[ReplayOutcome, ...]:
        """Replay a whole batch in ONE child, falling back on any failure.

        This trades isolation BETWEEN the recipes in a batch for the process
        spawn they would otherwise each pay: they share one interpreter, so
        module-level and class-level state one recipe writes is visible to the
        next.  Isolation from the PARENT is untouched -- a crash still costs
        only the child.

        Because a crash or a hang takes the whole batch down with it, an
        abnormal exit is never reported as "they all crashed": the batch is
        re-run one recipe per child so the failure lands on the recipe that
        caused it and the others get their real outcome.
        """

        items = tuple(items)
        if not items:
            return ()
        if len(items) == 1:
            return (self.replay(*items[0]),)
        response = self._invoke(
            "batch",
            items,
            None,
            options={"trace": self.trace},
            timeout_s=self.timeout_s * len(items),
        )
        if isinstance(response, tuple) and len(response) == len(items):
            return response
        return tuple(self.replay(goal, recipe) for goal, recipe in items)

    def mine_template(self, goal: ObjectGoal, recipe: ConstructionRecipe):
        """Return a ``FlatObjectTemplate`` or a replay-style failure outcome."""

        from tharkun.avengers.flatting import FlatObjectTemplate

        response = self._invoke("flatting", goal, recipe, {})
        if isinstance(response, FlatObjectTemplate):
            try:
                response.recipe()
            except Exception as error:
                return ReplayOutcome(
                    "worker-error",
                    phase="flatting",
                    detail=f"invalid template response: {error}",
                    exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
                )
            return response
        if isinstance(response, ReplayOutcome):
            return response
        return ReplayOutcome("worker-error", phase="flatting", detail="invalid template response")

    def generate_source(self, goal: ObjectGoal, tool: str = "ghostwriter"):
        """Ask a child to write test source for the target module."""

        from tharkun.avengers._wire import GeneratedSource

        response = self._invoke(tool, goal, ConstructionRecipe(goal.key), {})
        if isinstance(response, (GeneratedSource, ReplayOutcome)):
            return response
        return ReplayOutcome("worker-error", phase=tool, detail="invalid source response")

    def search_constructors(self, goal: ObjectGoal, *, max_examples: int = 50):
        """Ask a child to draw constructor arguments with ``st.from_type``."""

        from tharkun.avengers._wire import ConstructorCandidates

        response = self._invoke(
            "hypothesis-arguments",
            goal,
            ConstructionRecipe(goal.key),
            {"max_examples": max_examples},
        )
        if isinstance(response, (ConstructorCandidates, ReplayOutcome)):
            return response
        return ReplayOutcome(
            "worker-error", phase="hypothesis", detail="invalid candidate response"
        )

    def _invoke(
        self,
        action: str,
        goal,
        recipe,
        options: dict | None = None,
        *,
        timeout_s: float | None = None,
    ):
        options = dict(options or {})
        if self.files is not None:
            options["files"] = self.files
        payload = (
            (action, goal, options or {})
            if action == "batch"
            else (action, goal, recipe, options or {})
        )
        deadline = self.timeout_s if timeout_s is None else timeout_s
        scratch = tempfile.TemporaryDirectory(prefix="tharkun-avengers-") if self.files is not None else None
        if scratch is not None:
            options["scratch"] = scratch.name
        request = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        fd, response_path = tempfile.mkstemp(prefix="response-", suffix=".json",
                                            dir=scratch.name if scratch is not None else None)
        os.close(fd)
        try:
            environment = os.environ.copy()
            if self.interpreter is None:
                # A source checkout or embedding application may have made
                # Tharkun and the target importable through ``sys.path`` rather
                # than an installed distribution. Preserve that import context
                # for the disposable interpreter without sharing any live
                # module state.
                paths: list[str] = []
                for entry in sys.path:
                    resolved = os.path.abspath(entry or os.getcwd())
                    if os.path.isdir(resolved) and resolved not in paths:
                        paths.append(resolved)
                inherited = environment.get("PYTHONPATH")
                if inherited:
                    paths.extend(
                        entry for entry in inherited.split(os.pathsep) if entry not in paths
                    )
                environment["PYTHONPATH"] = os.pathsep.join(paths)
            else:
                # A DIFFERENT interpreter was named, so this one's sys.path is
                # the wrong answer: handing over the parent's site-packages
                # would let the child import a version of the target that the
                # named environment does not have. It brings its own.
                environment.pop("PYTHONPATH", None)
            proc = subprocess.Popen(
                [self.interpreter or sys.executable, "-m", "tharkun.avengers._worker", response_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
            try:
                proc.communicate(request, timeout=deadline)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (AttributeError, ProcessLookupError, PermissionError):
                    proc.kill()
                proc.communicate()
                return ReplayOutcome("timeout", phase="worker")

            # A target can see the response path in the worker's argv. Never
            # let bytes it wrote before a hard exit conceal the process crash.
            if proc.returncode != 0:
                return self._abnormal(proc.returncode)
            response = self._read_response(response_path)
            if response is not None:
                return response
            return self._abnormal(proc.returncode)
        finally:
            try:
                os.unlink(response_path)
            except FileNotFoundError:
                pass
            if scratch is not None:
                scratch.cleanup()

    @staticmethod
    def _read_response(path: str):
        try:
            size = os.path.getsize(path)
            if size == 0 or size > MAX_RESPONSE_BYTES:
                return None
            with open(path, "rb") as stream:
                value = decode_response(stream.read(MAX_RESPONSE_BYTES + 1))
        except (OSError, ValueError):
            return None
        return value

    @staticmethod
    def _abnormal(returncode: int | None) -> ReplayOutcome:
        if returncode is None:
            return ReplayOutcome("worker-error", phase="worker", detail="missing return code")
        if returncode < 0:
            signum = -returncode
            kind = "killed" if signum in {signal.SIGKILL, signal.SIGTERM} else "crashed"
            return ReplayOutcome(kind, phase="worker", signal=signum)
        if returncode >= 128:
            signum = returncode - 128
            return ReplayOutcome("crashed", phase="worker", signal=signum)
        return ReplayOutcome(
            "worker-error",
            phase="worker",
            detail=f"child exited with status {returncode}",
        )
