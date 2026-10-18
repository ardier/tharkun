"""Targets used to verify Avengers in a disposable child interpreter."""

from __future__ import annotations

import os
import signal
import sys
import time


class Box:
    def __init__(self, lower: int, upper: int):
        if lower >= upper:
            raise ValueError("lower must be smaller than upper")
        self.lower = lower
        self.upper = upper

    def width(self) -> int:
        return self.upper - self.lower


def classify_box(box: Box, minimum: int = 0) -> str:
    return "wide" if box.width() > minimum else "narrow"


def start_timer(timer: "TimerLike") -> None:
    timer.start()


class TimerLike:
    def __init__(self, logger: object = None):
        self.logger = logger
        self.running = False

    def start(self) -> None:
        if self.running:
            raise RuntimeError("already running")
        self.running = True

    def stop(self) -> float:
        if not self.running:
            raise RuntimeError("not running")
        self.running = False
        return 0.0

    def hard_crash(self) -> None:
        os.kill(os.getpid(), signal.SIGSEGV)

    def spoof_then_crash(self) -> None:
        # The target can see the response path. A valid-looking response must
        # never take precedence over the worker's abnormal exit status.
        response = (
            b'{"protocol":"tharkun-avengers-response/1","type":"replay","value":'
            b'{"kind":"returned","phase":"boundary","detail":null,'
            b'"exception_type":null,"signal":null,"return_types":[],'
            b'"state_digest":null}}'
        )
        with open(sys.argv[1], "wb") as stream:
            stream.write(response)
        os.kill(os.getpid(), signal.SIGSEGV)

    def hang(self) -> None:
        time.sleep(60)


class Noisy:
    def __init__(self):
        print("constructor noise")

    def speak(self) -> None:
        print("method noise")


class Impossible:
    def __init__(self, value: int):
        raise ValueError(value)


class Shipment:
    """A class that cannot be built from literals: it needs a real Box."""

    def __init__(self, box: Box, label: str = ""):
        if not isinstance(box, Box):
            raise TypeError("box must be a Box")
        self.box = box
        self.label = label
        self.sealed = False

    def seal(self) -> None:
        if self.sealed:
            raise RuntimeError("already sealed")
        self.sealed = True

    def volume(self) -> int:
        return self.box.width()
