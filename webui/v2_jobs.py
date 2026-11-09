"""Launch, resume and inspect V2 runs on behalf of the Train and Monitor tabs.

Everything here shells out to the V2 command shell (pipeline_v2/run.py). The
V2 engine is the authority on stages, the lease and the terminal label; this
module only starts processes, records what it started and renders what the
commands printed. It never reimplements a V2 decision and never writes inside
runs/v2, which is the engine's evidence tree.

Three rules come straight from the V2 contract and are not negotiable here:

* Resume is the original command with zero added flags. resume() replays the
  recorded argv verbatim. It does not append --resume, --seed or --device,
  because any change to the command is a NEW run identity in V2, not a resume.
* No kill button. A V2 run is not stopped by killing a pid. The UI offers no
  destructive stop, and says why, rather than offering a button that would
  break the exact-resume guarantee.
* preflight --gpu is the only device probe. It runs only when the operator
  ticks the GPU box, and the V2 lease is the authoritative busy-device guard.
  The UI check below is a courtesy pre-check that reports what it saw; it
  never overrides the lease.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from v2_reader import DASH, ROOT, TICK

PIPELINE = ROOT / "pipeline_v2"
RUN_PY = PIPELINE / "run.py"
EXPERIMENTS = PIPELINE / "configs" / "experiments"
JOB_STATE = ROOT / "webui" / "v2_job_state.json"
# The UI writes its own logs here rather than into runs/v2, which belongs to
# the engine and is hash-chained evidence.
JOB_LOGS = ROOT / "webui" / "v2_job_logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CLI_TIMEOUT = 120


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def code(value) -> str:
    """Render a value as markdown inline code."""
    return TICK + str(value) + TICK


def _emit_ok(payload) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    status = payload.get("status") or payload.get("decision") or "-"
    reason = payload.get("reason") or payload.get("error") or ""
    return "**" + str(status) + "**" + (
        (" " + DASH + " " + str(reason)) if reason else "")


def run_cli(arguments: list, timeout: int = CLI_TIMEOUT):
    """Run one V2 command and return (exit_code, payload).

    The command is CPU-only by construction: every command this module issues
    without the --gpu flag is documented never to touch a device. A caller
    that wants the device asks for it explicitly.
    """
    argv = [str(PYTHON), str(RUN_PY)] + list(arguments)
    try:
        done = subprocess.run(argv, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        return 124, {"status": "BLOCKED",
                     "reason": "the command did not finish within "
                               + str(timeout) + "s"}
    except OSError as error:
        return 127, {"status": "BLOCKED", "reason": str(error)}
    raw = (done.stdout or "").strip()
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None
    if payload is None:
        detail = (done.stderr or "").strip() or raw
        payload = {"status": "FAIL", "reason": detail[:800]}
    return done.returncode, payload


def experiment_configs() -> list:
    """The shipped V2 experiment configs, repo-relative, sorted.

    V2 states that YAML is the adaptation surface and that an ordinary
    experiment needs no Python edit, so the Train tab launches a declared
    config rather than synthesising one from form fields. Writing a config
    from a form would also fight the schema: in V2 an unknown key is a
    validation error, not a silently ignored field.
    """
    if not EXPERIMENTS.is_dir():
        return []
    return sorted(p.relative_to(ROOT).as_posix()
                  for p in EXPERIMENTS.glob("*.yaml"))

# ------------------------------------------------------------ read-only views

def validate_report(config_rel: str) -> str:
    """Validate a config. CPU-only and writes nothing."""
    if not config_rel:
        return "Pick a config first."
    _, payload = run_cli(["validate", "--config", config_rel])
    return "### validate " + code(config_rel) + "\n\n" + _emit_ok(payload)


def plan_report(config_rel: str) -> str:
    """The stage graph and its file effects. Writes nothing, takes no device."""
    if not config_rel:
        return "Pick a config first."
    _, payload = run_cli(["plan", "--config", config_rel])
    if not isinstance(payload, dict) or not payload.get("stages"):
        return "### plan " + code(config_rel) + "\n\n" + _emit_ok(payload)
    lines = ["### plan " + code(config_rel), "",
             str(len(payload["stages"])) + " stages. Writes nothing and "
             "acquires no device.", "",
             "| # | stage | requires | completion marker |",
             "|---|---|---|---|"]
    for index, stage in enumerate(payload["stages"], start=1):
        requires = ", ".join(stage.get("requires") or []) or "-"
        lines.append("| " + str(index) + " | " + str(stage.get("name"))
                     + " | " + requires + " | "
                     + code(stage.get("completion_marker", "-")) + " |")
    return "\n".join(lines)


def explain_report(config_rel: str) -> str:
    """The resolved fields, capabilities and semantic hash for a config."""
    if not config_rel:
        return "Pick a config first."
    _, payload = run_cli(["explain", "--config", config_rel])
    if not isinstance(payload, dict):
        return "### explain\n\n" + _emit_ok(payload)
    lines = ["### explain " + code(config_rel), ""]
    for section in ("run", "model", "runtime", "evaluation"):
        body = payload.get(section)
        if isinstance(body, dict):
            lines.append("**" + section + "**")
            for key, value in body.items():
                lines.append("- " + key + ": " + code(value))
            lines.append("")
    if payload.get("semantic_hash"):
        lines.append("semantic_hash: " + code(payload["semantic_hash"]))
    return "\n".join(lines)


def preflight_report(config_rel: str, gpu: bool, device: str = "cuda:0") -> str:
    """Run the V2 preflight. The --gpu form only when the operator asked.

    The GPU form is the one V2 command that probes a device, so it is never
    issued implicitly. The rendered result names which of the two forms ran,
    so the panel never implies a device check that did not happen.
    """
    if not config_rel:
        return "Pick a config first."
    arguments = ["preflight", "--config", config_rel]
    if gpu:
        arguments += ["--gpu", "--device", device]
    _, payload = run_cli(arguments, timeout=180)
    body = _emit_ok(payload)
    note = ""
    if gpu and isinstance(payload, dict):
        observation = payload.get("observation") or {}
        free = observation.get("free_memory_bytes")
        total = observation.get("total_memory_bytes")
        if observation.get("name") and isinstance(free, int) and isinstance(total, int):
            note = ("\n\n" + str(observation.get("name")) + " "
                    + code(observation.get("device", "")) + ": "
                    + format(free / 2**30, ".1f") + " GiB free of "
                    + format(total / 2**30, ".1f"))
    return ("### preflight (" + ("GPU" if gpu else "CPU") + ") "
            + code(config_rel) + "\n\n" + body + note)


# ------------------------------------------------------------------ GPU guard

def gpu_compute_pids() -> list:
    """PIDs holding a CUDA context right now; [] when nvidia-smi is absent.

    Unavailable is reported as an empty list and the caller must treat that as
    unknown rather than as idle. The V2 lease is the authoritative guard; this
    is a courtesy pre-check that puts the state on screen before a launch.
    """
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True)
    except Exception:  # noqa: BLE001 - no nvidia-smi, or it refused
        return []
    pids = []
    for line in (done.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids

# ------------------------------------------------------------------ job state

def job():
    """The recorded job, or None when this UI never launched one."""
    return load_json(JOB_STATE)


def job_alive(record) -> bool:
    """Whether the recorded process is still running. Never raises.

    When the liveness of a pid cannot be established the answer is False,
    because claiming a job is live when it is not would hide a launch that
    already ended. Resume re-reads the recorded argv either way.
    """
    if not record:
        return False
    try:
        pid = int(record.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:  # noqa: BLE001 - cannot tell, so do not claim it is live
        return False


def _write_job(payload: dict) -> None:
    JOB_STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def argv_for(config_rel: str) -> list:
    """The exact command for a config. This is what resume replays."""
    return [str(PYTHON), str(RUN_PY), "run", "--config", str(config_rel)]


# --------------------------------------------------------------------- launch

def launch(config_rel: str, config_ok: bool) -> str:
    """Start the V2 run as a detached subprocess.

    The argv recorded here is the argv that resume() replays. V2 treats any
    command change as a new run identity, so this record is the resume
    contract: it is stored verbatim and never normalised or re-ordered.
    """
    if not config_rel:
        return "Pick a config before launching."
    if not config_ok:
        return ("**BLOCKED** " + DASH + " run validate first. Launching an "
                "unvalidated config produces a run identity nobody checked.")
    if not PYTHON.is_file():
        return ("**BLOCKED** " + DASH + " the repository venv is missing: "
                + code(PYTHON))

    current = job()
    if job_alive(current):
        return ("**BLOCKED** " + DASH + " a V2 job is already live (pid "
                + str(current.get("pid"))
                + "). One protected job at a time.")

    busy = gpu_compute_pids()
    if busy:
        return ("**BLOCKED** " + DASH + " the device already holds CUDA "
                "contexts from pids " + ", ".join(str(p) for p in busy)
                + ". The V2 lease would refuse this too. No foreign process is "
                  "ever killed.")

    argv = argv_for(config_rel)
    JOB_LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_path = JOB_LOGS / ("run-" + stamp + ".out.log")
    err_path = JOB_LOGS / ("run-" + stamp + ".err.log")
    try:
        handle = out_path.open("w", encoding="utf-8")
        process = subprocess.Popen(argv, cwd=str(ROOT), stdout=handle,
                                   stderr=err_path.open("w", encoding="utf-8"))
    except OSError as error:
        return "**BLOCKED** " + DASH + " could not start the run: " + str(error)

    _write_job({
        "argv": argv,
        "config": config_rel,
        "pid": process.pid,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stdout_log": out_path.relative_to(ROOT).as_posix(),
        "stderr_log": err_path.relative_to(ROOT).as_posix(),
    })
    return ("**Launched** " + code(config_rel) + " (pid " + str(process.pid)
            + ").\n\nThe exact command was recorded and will be replayed "
              "verbatim by Resume. Watch it in the Monitor tab.")


def resume() -> str:
    """Replay the recorded argv verbatim: the original command, zero flags.

    This is the point of recording it. V2 refuses a resume that changed the
    command, and this module has no --resume flag to offer, so the UI cannot
    manufacture a non-resume and label it a resume.
    """
    current = job()
    if not current:
        return "**BLOCKED** " + DASH + " no V2 job has been recorded."
    if job_alive(current):
        return ("**BLOCKED** " + DASH + " the recorded job is still live (pid "
                + str(current.get("pid")) + ").")
    argv = current.get("argv")
    if not isinstance(argv, list) or not argv:
        return ("**BLOCKED** " + DASH + " the recorded job has no argv, so the "
                "exact command cannot be replayed.")
    try:
        process = subprocess.Popen(argv, cwd=str(ROOT))
    except OSError as error:
        return ("**BLOCKED** " + DASH + " could not replay the command: "
                + str(error))
    updated = dict(current)
    updated["pid"] = process.pid
    updated["resumed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_job(updated)
    return ("**Resumed** pid " + str(process.pid)
            + " by replaying the exact recorded command, with no flags "
              "added.\n\n" + code(" ".join(str(a) for a in argv)))


def job_banner() -> str:
    """The one-line job state shown on every tab."""
    current = job()
    if not current:
        return "No V2 job has been launched from this UI."
    state = "LIVE" if job_alive(current) else "not running"
    return ("V2 job: **" + state + "** " + DASH + " "
            + code(str(current.get("config"))) + " (pid "
            + str(current.get("pid")) + ", started "
            + str(current.get("started_utc", "?")) + ")\n\n"
            + code(" ".join(str(a) for a in (current.get("argv") or []))))


def stop_unavailable_markdown() -> str:
    """Why the Monitor stop control is disabled in V2 mode.

    The V1 stop control writes a cooperative stop file that the V1 trainer
    honours at the next checkpoint. V2 defines no equivalent protocol, and
    terminating the process would leave a partial write that the exact-command
    resume rule then refuses to continue. The honest control is absent with
    the reason shown, not a button that silently degrades the guarantee.
    """
    return (
        "**No stop control in V2 mode.** " + DASH + " V1 stops a run by "
        "writing a cooperative stop file the trainer honours at the next "
        "checkpoint. V2 defines no equivalent protocol, and terminating the "
        "process would leave a partial write that the exact-command resume "
        "rule then refuses to continue. A V2 run ends at its terminal label, "
        "or is ended deliberately by the operator at the shell."
        "\n\nUse **Resume** once a V2 job has ended: it replays the recorded "
        "command with zero flags.")


