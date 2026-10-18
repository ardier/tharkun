"""Linux exploration worker. Exec once, then fork without ever loading a target here.

The socket is private to the verifier/server; draw children inherit no IPC descriptor.
Results are advisory: the engine replays violations and never proves from fork runs.
"""
import ctypes
import json
import math
import os
import resource
import signal
import site
import socket
import sys
import tempfile
import threading
import time

# PR_SET_CHILD_SUBREAPER = 36: reparent orphaned grandchildren here so they cannot escape reap.
try:
    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
except Exception:
    pass

# Install before preloading anything else: even a joined Python thread forbids fork.
_threads_started = False
_start = threading.Thread.start


def _thread_start(self, *args, **kwargs):
    global _threads_started
    _threads_started = True
    return _start(self, *args, **kwargs)


threading.Thread.start = _thread_start
import _child  # noqa: E402 -- stdlib/model only; never import the target in the server


def _single_threaded():
    return not _threads_started and len(os.listdir('/proc/self/task')) == 1


def _child_setup(tmp, memory_bytes, timeout, dispositions):
    os.setsid()
    # No server socket, request stream, or incidental preload FD reaches the target.
    for name in os.listdir('/proc/self/fd'):
        fd = int(name)
        if fd > 2:
            try:
                os.close(fd)
            except OSError:
                pass
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    if null > 2:
        os.close(null)
    for sig, handler in dispositions.items():
        signal.signal(sig, handler)
    for timer in (signal.ITIMER_REAL, signal.ITIMER_VIRTUAL, signal.ITIMER_PROF):
        signal.setitimer(timer, 0)
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    os.chdir(tmp)
    os.environ.update(TMPDIR=tmp, TMP=tmp, TEMP=tmp)
    tempfile.tempdir = tmp
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if memory_bytes:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = min(memory_bytes, hard) if hard != resource.RLIM_INFINITY else memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limit = max(1, math.ceil(timeout))
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    # Clear parent atexit handlers so server preloads do not run hooks in the child.
    import atexit
    atexit._clear()
    # The server starts with -S: .pth/sitecustomize hooks can start threads, so run
    # them only in disposable children, after isolation and resource limits apply.
    site.main()
    _child._random.seed(None)  # discard the preloaded generator's state before modelling


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _cleanup_descendants(child_pid):
    _kill_group(child_pid)
    try:
        os.kill(child_pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.wait4(child_pid, 0)
    except OSError:
        pass
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        return

    server_pid = os.getpid()
    for _ in range(50):
        descendants = []
        for entry in os.listdir('/proc'):
            if entry.isdigit():
                try:
                    with open(f'/proc/{entry}/status') as f:
                        for line in f:
                            if line.startswith('PPid:'):
                                if int(line.split()[1]) in (child_pid, server_pid):
                                    descendants.append(int(entry))
                                break
                except OSError:
                    pass
        for p in descendants:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                wpid, _ = os.waitpid(-1, os.WNOHANG)
                if wpid <= 0:
                    break
            except ChildProcessError:
                break
        if not descendants:
            break
        time.sleep(0.01)


def _execute(req, dispositions):
    if not _single_threaded():
        return {'status': 'crashed', 'msg': 'fork server is not single-threaded'}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCHLD])
    pid = os.fork()
    if pid == 0:
        try:
            _child_setup(req['tmp'], req['memory_bytes'], req['timeout'], dispositions)
            _child.main(req['path'])
        except BaseException:
            os._exit(1)
        # Run atexit functions for target cleanup; verdict is already written to req['out'].
        # Nothing a verdict depends on may rely on exit handlers.
        try:
            import atexit
            atexit._run_exitfuncs()
        except BaseException:
            pass
        os._exit(0)
    timed_out = False
    deadline = time.monotonic() + req['timeout']
    status = 0
    try:
        # Blocking SIGCHLD before fork prevents a lost wakeup; no polling delay.
        while True:
            waited, status, _ = os.wait4(pid, os.WNOHANG)
            if waited:
                break
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                _cleanup_descendants(pid)
                try:
                    _, status, _ = os.wait4(pid, 0)
                except ChildProcessError:
                    pass
                break
            signal.sigtimedwait({signal.SIGCHLD}, left)
    finally:
        _cleanup_descendants(pid)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    return {'status': 'timeout' if timed_out else ('ok' if status == 0 else 'crashed')}


def main(fd):
    dispositions = {sig: signal.getsignal(sig) for sig in signal.valid_signals()
                    if sig not in (signal.SIGKILL, signal.SIGSTOP)
                    and signal.getsignal(sig) is not None}
    with socket.socket(fileno=fd) as sock, sock.makefile('rwb') as stream:
        own = resource.getrusage(resource.RUSAGE_SELF)
        stream.write(json.dumps({"ready": True, "cpu": own.ru_utime + own.ru_stime}).encode() + b'\n')
        stream.flush()
        for line in stream:
            try:
                result = _execute(json.loads(line), dispositions)
            except Exception as exc:
                result = {'status': 'crashed', 'msg': str(exc)[:300]}
            own = resource.getrusage(resource.RUSAGE_SELF)
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            result['cpu'] = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
            stream.write(json.dumps(result).encode() + b'\n')
            stream.flush()


if __name__ == '__main__':
    main(int(sys.argv[1]))
