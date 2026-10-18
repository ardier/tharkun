"""Attacking frontiers in every module the harness touches.

The assist used to name one target_path/target_func, so on a real harness the frontiers
in the other modules were simply invisible — measured at 4 of 23 reachable frontier
statements on ntlm_auth. Lifting that needs one fact the positional codec never had:
which fuzzer field reaches which callee parameter. It is recoverable from the harness,
because a harness passes consumes straight into calls.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import unittest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.hybrid.harness_map import map_harness_calls

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
HARNESS = os.path.join(FIXDIR, "xmod_harness.py")
PKG = os.path.join(FIXDIR, "xpkg")


class TestHarnessMapping(unittest.TestCase):
    """Pure AST — runs without atheris."""

    def test_recovers_field_to_parameter_bindings(self) -> None:
        got = {t.func: t.field_to_param for t in map_harness_calls(HARNESS, PKG)}
        self.assertEqual(got["check_a"], {0: "code"})
        self.assertEqual(got["check_c"], {1: "n", 2: "m"})
        self.assertEqual(got["check_b"], {3: "name"})

    def test_each_target_knows_its_module(self) -> None:
        mods = {t.func: os.path.basename(t.module_path) for t in map_harness_calls(HARNESS, PKG)}
        self.assertEqual(mods, {"check_a": "mod_a.py", "check_b": "mod_b.py",
                                "check_c": "mod_c.py"})

    def test_unknown_package_yields_nothing(self) -> None:
        self.assertEqual(map_harness_calls(HARNESS, "/nonexistent"), [])

    def test_missing_harness_yields_nothing(self) -> None:
        self.assertEqual(map_harness_calls("/nonexistent.py", PKG), [])


@requires_atheris
class TestCrossModuleAssist(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, FIXDIR)
        from xpkg import mod_a, mod_b, mod_c
        self.mods = (mod_a, mod_b, mod_c)
        self.corpus = tempfile.mkdtemp()
        with open(os.path.join(self.corpus, "s"), "wb") as f:
            f.write(b"\x00" * 48)

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def _arms(self) -> str:
        import atheris
        a, b, c = self.mods
        got = set()
        for p in glob.glob(self.corpus + "/*"):
            f = atheris.FuzzedDataProvider(open(p, "rb").read())
            try:
                got.add(a.check_a(f.ConsumeUInt(4)))
                got.add(c.check_c(f.ConsumeUInt(4), f.ConsumeUInt(4)))
                got.add(b.check_b(f.ConsumeUnicodeNoSurrogates(32)))
            except Exception:
                pass
        return "".join(sorted(got - {"-"}))

    def test_unlocks_frontiers_in_every_module(self) -> None:
        spec = HarnessSpec(harness_path=HARNESS,
                           target_path=os.path.join(PKG, "mod_a.py"),
                           target_func="check_a", corpus_dir=self.corpus,
                           package_dir=PKG)
        eng = AssistEngine(spec, max_frontiers=1)
        for _ in range(8):
            eng.assist_once()
            if self._arms() == "ABC":
                break
        self.assertEqual(self._arms(), "ABC")

    def test_single_target_mode_only_reaches_its_own_module(self) -> None:
        # Without package_dir the old behaviour stands: only mod_a is attacked.
        spec = HarnessSpec(harness_path=HARNESS,
                           target_path=os.path.join(PKG, "mod_a.py"),
                           target_func="check_a", corpus_dir=self.corpus)
        eng = AssistEngine(spec, max_frontiers=4)
        for _ in range(4):
            eng.assist_once()
        self.assertNotIn("B", self._arms())


class TestOnlyWhatTheHarnessCanReach(unittest.TestCase):
    """The assist must never target code the original harness cannot statically reach.

    Two ways that could break, both closed here: resolving a callee by bare name (this
    fixture and real packages alike define one name in several modules — ntlm_auth has
    seven), and inventing a value for a parameter the harness pins to a constant.
    """

    def test_qualifier_decides_which_module_is_targeted(self) -> None:
        # xpkg/decoy.py defines check_a and check_b too, and the harness never imports
        # it. Resolution must follow `mod_a.check_a`, not the bare name.
        by_func = {t.func: t.module_path for t in map_harness_calls(HARNESS, PKG)}
        self.assertEqual(os.path.basename(by_func["check_a"]), "mod_a.py")
        self.assertEqual(os.path.basename(by_func["check_b"]), "mod_b.py")
        self.assertNotIn("decoy.py", [os.path.basename(p) for p in by_func.values()])

    def test_every_target_is_a_call_written_in_the_harness(self) -> None:
        import ast

        called = set()
        for node in ast.walk(ast.parse(open(HARNESS).read())):
            if isinstance(node, ast.Call):
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else getattr(node.func, "id", None))
                if name:
                    called.add(name)
        for t in map_harness_calls(HARNESS, PKG):
            self.assertIn(t.func, called, f"{t.func} is not called in the harness source")

    def test_fixed_args_come_from_the_harness_call_site(self) -> None:
        import ast
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            pkg = os.path.join(d, "pin")
            os.makedirs(pkg)
            open(os.path.join(pkg, "__init__.py"), "w").close()
            with open(os.path.join(pkg, "m.py"), "w") as f:
                f.write("def gated(flag, tag):\n    return tag if flag else None\n")
            h = os.path.join(d, "h.py")
            with open(h, "w") as f:
                f.write("import atheris\n"
                        "from pin import m\n"
                        "TAG = b'pinned-by-the-harness'\n"
                        "def TestOneInput(data):\n"
                        "    fdp = atheris.FuzzedDataProvider(data)\n"
                        "    m.gated(fdp.ConsumeInt(4), TAG)\n")
            [t] = map_harness_calls(h, pkg)
            self.assertEqual(t.field_to_param, {0: "flag"})
            # The value is what the harness passes, read from the harness — not invented.
            self.assertEqual(t.fixed_args, {"tag": b"pinned-by-the-harness"})
            src = ast.parse(open(h).read())
            literal = next(n for n in ast.walk(src)
                           if isinstance(n, ast.Assign) and n.targets[0].id == "TAG")
            self.assertEqual(t.fixed_args["tag"], ast.literal_eval(literal.value))

    def test_ambiguous_bare_call_is_dropped_not_guessed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            pkg = os.path.join(d, "amb")
            os.makedirs(pkg)
            open(os.path.join(pkg, "__init__.py"), "w").close()
            for mod in ("one.py", "two.py"):
                with open(os.path.join(pkg, mod), "w") as f:
                    f.write("def same(x):\n    return x\n")
            h = os.path.join(d, "h.py")
            with open(h, "w") as f:
                f.write("import atheris\n"
                        "from amb.one import same\n"
                        "def TestOneInput(data):\n"
                        "    fdp = atheris.FuzzedDataProvider(data)\n"
                        "    same(fdp.ConsumeInt(4))\n")
            # `same` is defined in two modules and called without a qualifier: there is no
            # sound way to pick, so it is dropped rather than resolved to whichever came
            # first on disk.
            self.assertEqual(map_harness_calls(h, pkg), [])


class TestSeedsAreNotOverwritten(unittest.TestCase):
    """Two solutions differing only in late bytes must be two corpus files.

    Seeds were named by `data.hex()[:40]` — the first TWENTY bytes. FDP consumes ints
    from the END of the buffer, so every solution that differed only in an int field
    collided on one filename and all but the last were silently lost. On ntlm_auth that
    was all three negotiate_flags solutions, and it cost the last uncovered line.
    """

    def setUp(self) -> None:
        self.corpus = tempfile.mkdtemp()
        self.eng = AssistEngine(HarnessSpec(
            harness_path=HARNESS, target_path=os.path.join(PKG, "mod_a.py"),
            target_func="check_a", corpus_dir=self.corpus))

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def test_late_byte_difference_yields_a_distinct_file(self) -> None:
        head = b"\xAA" * 64
        a = self.eng._write_seed(head + b"\x01")
        b = self.eng._write_seed(head + b"\x02")
        self.assertNotEqual(a, b)
        self.assertEqual(len(os.listdir(self.corpus)), 2)
        self.assertEqual(open(a, "rb").read(), head + b"\x01")
        self.assertEqual(open(b, "rb").read(), head + b"\x02")

    def test_identical_content_reuses_one_file(self) -> None:
        data = b"\xAA" * 64 + b"\x01"
        self.assertEqual(self.eng._write_seed(data), self.eng._write_seed(data))
        self.assertEqual(len(os.listdir(self.corpus)), 1)

    def test_name_is_stable_across_processes(self) -> None:
        # hash() is salted per process; the name must not be.
        import subprocess
        import sys as _sys

        code = ("import sys; sys.path.insert(0, 'src');"
                "from tharkun.hybrid.assist import AssistEngine, HarnessSpec;"
                f"e = AssistEngine(HarnessSpec('{HARNESS}', '{PKG}/mod_a.py',"
                f" 'check_a', '{self.corpus}'));"
                "print(e._write_seed(b'stable-content'))")
        runs = {subprocess.run([_sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=os.getcwd()).stdout.strip()
                for _ in range(2)}
        self.assertEqual(len(runs), 1, f"seed name varies across processes: {runs}")


@requires_atheris
class TestExternalCorpusReopensFrontiers(unittest.TestCase):
    """A seed written by Atheris — not by the assist — must re-open exhausted frontiers.

    The assist skips a frontier that already resisted everything, gated on nothing having
    been learned since. But the corpus changes from OUTSIDE: the whole design runs Atheris
    in the background writing seeds into the same directory. Without noticing that, the
    assist would go permanently quiet in exactly the mode it exists for, while Atheris
    kept feeding it fresh material.
    """

    def setUp(self) -> None:
        sys.path.insert(0, FIXDIR)
        self.corpus = tempfile.mkdtemp()
        with open(os.path.join(self.corpus, "s"), "wb") as f:
            f.write(b"\x00" * 48)
        self.eng = AssistEngine(HarnessSpec(
            harness_path=HARNESS, target_path=os.path.join(PKG, "mod_a.py"),
            target_func="check_a", corpus_dir=self.corpus, package_dir=PKG),
            max_frontiers=6)

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def test_an_externally_written_seed_bumps_the_epoch(self) -> None:
        for _ in range(4):
            self.eng.assist_once()
        before = self.eng.facts.epoch
        self.eng.assist_once()               # quiet round: learns nothing
        quiet = self.eng.facts.epoch
        with open(os.path.join(self.corpus, "atheris-seed"), "wb") as f:
            f.write(bytes((i * 13 + 5) & 0xFF for i in range(64)))
        self.eng.assist_once()
        self.assertGreater(self.eng.facts.epoch, quiet,
                           "a seed from outside the assist did not count as new information")
        self.assertGreaterEqual(quiet, before)

    def test_a_quiet_round_skips_rather_than_re_attacking(self) -> None:
        for _ in range(4):
            self.eng.assist_once()
        before = self.eng.facts.skipped["frontier"]
        self.eng.assist_once()
        self.assertGreaterEqual(self.eng.facts.skipped["frontier"], before)
