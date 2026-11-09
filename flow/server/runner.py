"""Subprocess runner for the flow/ MVP (Task 6).

One global GPU job slot (webui single-job semantics, mirrored): a job is
a run_custom.py subprocess launched with the repo's venv python on a
generated config yaml. A second start while one is live is a typed
ValueError - never a queue.

Stop contract (webui U11, mirrored verbatim - canonical mechanism in
src/stop.py and webui/app.py::request_stop): the runner NEVER kills the
process. request_stop() drops a STOP flag file (UTC timestamp payload,
same as webui) inside the job's training output_dir (the
"train.output_dir" from the config, resolved like train.py/webui do:
relative to the repo root). The trainer sees the flag only at the next
checkpoint save and exits cleanly with a valid checkpoint on disk; the
exact same zero-flag command relaunches to resume.

stdout is drained by a dedicated reader thread that iterates
proc.stdout line by line (blocking reads happen THERE - they never
block the caller, which only reads a deque snapshot under the lock).
Lines land in a deque(maxlen=TAIL_MAXLEN); the thread also waits for
the process so exit_code is captured exactly once, then clears the
global job slot. The last finished status is kept until the next start.

Tests inject script_path and interpreter; never launch real jobs.
"""

import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Repo conventions (Task 6): venv python + scripts/run_custom.py.
DEFAULT_INTERPRETER = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "run_custom.py"

# webui U11 flag name: src/stop.py::stop_flag_path -> <output_dir>/STOP.
STOP_FLAG_NAME = "STOP"

TAIL_MAXLEN = 200

# The generated configs carry exactly one "train.output_dir" (webui
# build_config contract; Task 5 config_gen copies it verbatim).
_OUTPUT_DIR_RE = re.compile(r"^\s+output_dir:\s*(.+?)\s*$", re.MULTILINE)


class JobRunningError(ValueError):
    """A GPU job is already live; start is rejected (single-job lock)."""


class NoJobRunningError(RuntimeError):
    """request_stop() with no live job (webui: 'no live job to stop')."""


class _State:
    """One global, lock-guarded job slot + last-finished snapshot."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None          # subprocess.Popen while running
        self.script_path = None
        self.config_path = None
        self.started_at = None
        self.stop_dir = None      # dir whose STOP file the job watches
        self.tail = deque(maxlen=TAIL_MAXLEN)
        self.exit_code = None
        self.exit_at = None

    def running(self):
        return self.proc is not None

    def status(self):
        return {
            "running": self.running(),
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "exit_at": self.exit_at,
            "tail": list(self.tail),
        }


_state = _State()


def _stop_dir_for(config_path, explicit=None):
    """Resolve the job's output_dir (where the U11 STOP flag lives).

    Explicit injection wins (tests); otherwise the first "output_dir:"
    line in the config is resolved against the repo root (train.py /
    webui convention for relative runs/<name> dirs).
    """
    if explicit is not None:
        return Path(explicit)
    text = Path(config_path).read_text(encoding="utf-8")
    match = _OUTPUT_DIR_RE.search(text)
    if match is None:
        raise ValueError(
            "config %s has no 'output_dir:' key - cannot locate the "
            "trainer output_dir for the U11 STOP flag" % (config_path,))
    value = match.group(1).strip().strip('"').strip("'")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _clear_state_exit(code):
    """Record completion and clear the slot (caller holds the lock)."""
    _state.exit_code = code
    _state.exit_at = time.time()
    _state.proc = None


def _reader_thread_body(job_proc, reader_stdout):
    """Drain stdout without blocking the caller.

    A dedicated daemon thread performs the blocking line reads from the
    child's stdout pipe so the caller and status() never block; lines
    land in the shared deque (thread-safe appends) which status() only
    ever copies under the state lock. After EOF it waits for the
    process, records exit code/time and clears the global job slot.
    """
    for line in reader_stdout:
        _state.tail.append(line.rstrip("\r\n"))
    code = job_proc.wait()
    with _state.lock:
        _clear_state_exit(code)


def start(config_path, *, script_path=None, interpreter=None,
          stop_dir=None, env=None):
    """Start a run_custom.py job on config_path; return the started pid.

    Raises a typed JobRunningError ('GPU job already running...') while
    a job is live and FileNotFoundError when the config is missing.
    """
    config_path = str(config_path)
    if script_path is None:
        script_path = str(DEFAULT_SCRIPT)
    if interpreter is None:
        interpreter = str(DEFAULT_INTERPRETER)
    env_pass = None
    if env is not None:
        env_pass = dict(env)

    if not Path(config_path).is_file():
        raise FileNotFoundError(
            "config not found: %s - nothing was started" % (config_path,))

    with _state.lock:
        if _state.running():
            raise JobRunningError(
                "GPU job already running (pid %s on config %s) - single "
                "global job slot, one GPU" % (_state.proc.pid,
                                              _state.config_path))
        stop_target = _stop_dir_for(config_path, stop_dir)
        proc = subprocess.Popen(
            [str(interpreter), str(script_path), "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # tail shows tracebacks too
            text=True,
            cwd=str(REPO_ROOT),
            env=env_pass,
        )
        _state.proc = proc
        _state.script_path = script_path
        _state.config_path = config_path
        _state.started_at = time.time()
        _state.stop_dir = stop_target
        _state.tail.clear()
        _state.exit_code = None
        _state.exit_at = None
    reader = threading.Thread(
        target=_reader_thread_body, args=(proc, proc.stdout), daemon=True)
    reader.start()
    return proc.pid


def wait_finished(timeout=30):
    """Helper for tests/teardown: wait until the slot is free.

    Does not force anything; a live job keeps running past the timeout
    and the current status is returned either way.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not status()["running"]:
            break
        time.sleep(0.05)
    return status()


def status():
    """Snapshot {running, exit_code, started_at, exit_at, tail}.

    Process none-safe: with no job ever started, running is False,
    exit_code/started_at are None and tail is an empty list. The tail
    is returned as a list copy.
    """
    with _state.lock:
        return _state.status()


def request_stop():
    """Write the U11 STOP flag file into the job's output_dir.

    NEVER terminates/kills the process: the trainer sees the flag at
    the next checkpoint save and exits cleanly (src/stop.py contract).
    Returns the flag path. webui-consistent no-live-job behaviour: a
    NoJobRunningError is raised when nothing is running.
    """
    with _state.lock:
        if not _state.running():
            raise NoJobRunningError("no live GPU job to stop")
        stop_dir = _state.stop_dir
    stop_dir.mkdir(parents=True, exist_ok=True)
    flag = stop_dir / STOP_FLAG_NAME
    payload = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(flag, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload + "\n")
        f.flush()
    return str(flag)
