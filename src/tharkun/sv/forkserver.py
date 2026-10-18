"""An exec-started, single-threaded server for advisory draw-vector exploration."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time


class ForkServer:
    def __init__(self, python):
        self.python = python
        self.proc = None
        self.cpu = 0.0
        self.lock = threading.Lock()

    def _start(self):
        parent, child = socket.socketpair()
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONSTARTUP')}
        # Hashing is fixed at exec, not by changing the child's environment after fork.
        env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
        try:
            self.proc = subprocess.Popen(
                [self.python, '-S', os.path.join(os.path.dirname(__file__), '_forkserver.py'), str(child.fileno())],
                pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.sock = parent
        self.stream = parent.makefile('rwb')

    def close(self):
        if self.proc is not None:
            self.stream.close()
            self.sock.close()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            self.cpu = 0.0  # os.times now includes the reaped server and its children

    def run(self, python, program, vector, *, timeout, hashseed=0, memory_bytes=None, files=None):
        from tharkun.sv.engine import Outcome, run_once
        if sys.platform != 'linux' or hashseed != 0:
            return run_once(python, program, vector, timeout=timeout, hashseed=hashseed,
                            memory_bytes=memory_bytes, files=files)
        # ponytail: one in-flight vector per server; callers use one server per batch slot.
        with self.lock, tempfile.TemporaryDirectory(prefix='tharkun-sv-fork-') as tmp:
            t0 = time.monotonic()
            req, out = os.path.join(tmp, 'req.json'), os.path.join(tmp, 'out.json')
            sent = False
            try:
                with open(req, 'w') as fh:
                    json.dump({'program': os.path.abspath(program), 'vector': vector, 'out': out, **({'files': files} if files is not None else {})}, fh)
                if self.proc is None:
                    if timeout <= time.monotonic() - t0:
                        return Outcome('timeout', seconds=time.monotonic() - t0)
                    self._start()
                    left = timeout - (time.monotonic() - t0)
                    if left <= 0:
                        self.proc.kill()
                        self.close()
                        return Outcome('timeout', seconds=time.monotonic() - t0)
                    self.sock.settimeout(left)
                    ready = json.loads(self.stream.readline())
                    if ready.get('ready') is not True:
                        raise ValueError('fork server did not finish startup')
                    self.cpu = ready['cpu']
                left = timeout - (time.monotonic() - t0)
                if left <= 0:
                    return Outcome('timeout', seconds=time.monotonic() - t0)
                # The response grace permits cleanup; the target gets only the remaining share.
                self.sock.settimeout(left + 5)
                # A partial write can start a child, so its server must still drain it.
                sent = True
                self.stream.write(json.dumps({'path': req, 'tmp': tmp, 'timeout': left,
                                              'memory_bytes': memory_bytes}).encode() + b'\n')
                self.stream.flush()
                result = json.loads(self.stream.readline())
                self.cpu = result['cpu']
                if result['status'] != 'ok':
                    return Outcome(result['status'], msg=result.get('msg', ''),
                                   seconds=time.monotonic() - t0)
                with open(out) as fh:
                    record = json.load(fh)
                return Outcome(record['status'], record.get('exc'), record.get('mro', []),
                               record.get('msg', ''), record.get('site'), record.get('draws', []),
                               record.get('flags', []) + ['forked'], time.monotonic() - t0,
                               invalid_draw=record.get('invalid_draw', False),
                               ranges=record.get('ranges', []), absent_imports=record.get('absent_imports', []))
            except (OSError, ValueError, KeyError) as error:
                if not sent and self.proc is not None:
                    self.proc.kill()
                self.close()
                return Outcome('timeout' if isinstance(error, TimeoutError) else 'crashed',
                               msg='fork server failed or left an unreadable record',
                               seconds=time.monotonic() - t0)
