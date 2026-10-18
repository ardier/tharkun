"""Run Pynguin in the SV environment; secondary execution is refused."""
from __future__ import annotations

import json
import os
import runpy
import sys

from tharkun.solve.environment import install_files, refuse_threads


def main():
    with open(sys.argv[1]) as stream:
        request = json.load(stream)
    sys.argv = ["pynguin", *request["argv"]]
    flags = set()
    install_files(request["files"], request["scratch"], flags)
    refuse_threads(flags)
    os.chdir(request["scratch"])
    # Keep the model installed through finalizers and atexit callbacks.
    runpy.run_module("pynguin", run_name="__main__")


if __name__ == "__main__":
    main()
