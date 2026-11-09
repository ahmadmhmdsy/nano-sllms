"""Read-only, on-demand hardware state for the local machine.

The command reports CPU, RAM, disk, network, GPU, dedicated VRAM, shared GPU
memory, and per-process GPU state as JSON on stdout. It is a local measurement
only:

- it never writes repository state or any other file;
- it never starts, stops, or signals a process;
- it never opens a network connection or resolves a name;
- it never calls a shell, so no argument can be interpreted as shell syntax.

A section that cannot be measured is reported as "unavailable" with a bounded
diagnostic. A measurement that exists but could not be read is reported as
null, never as zero. Dedicated VRAM (memory attached to the GPU) and shared GPU
memory (system memory used by GPU processes) stay separate at device and
process level and are never summed or relabelled.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import ntpath
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_BYTES_PER_MIB = 1024 * 1024
_MAX_ITEMS = 64
_MAX_TEXT = 160
_PROCESS_SECRET = re.compile(
    r"(?i)(?:AKIA|ASIA)[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|password|secret|token|api[_-]?key"
)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one read-only command; never carries a traceback."""

    returncode: int
    stdout: str
    stderr: str = ""
    error: str | None = None


def _number(
    value: str, *, integer: bool = False, scale: int = 1
) -> int | float | None:
    """Return a validated non-negative number, or None when unavailable."""
    text = value.strip()
    if not text or text.lower() in {
        "n/a",
        "na",
        "[n/a]",
        "[unknown]",
        "[not supported]",
        "not supported",
    }:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError("invalid numeric hardware field") from exc
    if number < 0:
        raise ValueError("negative hardware field")
    if integer:
        return int(number * scale)
    return number


def _bounded_process_name(value: str) -> str:
    """Return a safe process basename, redacting secret-shaped or odd names."""
    name = ntpath.basename(value.replace("/", "\\")).strip()
    if (
        not name
        or _PROCESS_SECRET.search(name)
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        return "<redacted-process-name>"
    return name[:_MAX_TEXT]


def _clean(text: str) -> str:
    """Return bounded, control-free, ASCII-safe text for the JSON document."""
    cleaned = "".join(
        char if 32 <= ord(char) < 127 else " " for char in str(text)
    ).strip()
    return cleaned[:_MAX_TEXT]


def parse_cuda_version(text: str) -> str | None:
    """Return the CUDA version from an nvidia-smi header, or None."""
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", text or "")
    return match.group(1) if match is not None else None


def parse_nvidia_gpu_csv(text: str) -> list[dict[str, Any]]:
    """Parse the read-only nvidia-smi device query into device records."""
    result: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text or "")):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 10:
            raise ValueError("invalid NVIDIA GPU row")
        result.append(
            {
                "index": _number(row[0], integer=True),
                "uuid": _clean(row[1]),
                "name": _clean(row[2]),
                "driver_version": _clean(row[3]),
                "dedicated_vram_total_bytes": _number(
                    row[4], integer=True, scale=_BYTES_PER_MIB
                ),
                "dedicated_vram_used_bytes": _number(
                    row[5], integer=True, scale=_BYTES_PER_MIB
                ),
                "dedicated_vram_available_bytes": _number(
                    row[6], integer=True, scale=_BYTES_PER_MIB
                ),
                "dedicated_vram_source": "nvidia-smi",
                "utilization_percent": _number(row[7]),
                "temperature_c": _number(row[8]),
                "power_w": _number(row[9]),
                "cuda_version": None,
                # Shared GPU memory is a separate measurement; it is never
                # derived from a dedicated VRAM value.
                "shared_memory_bytes": None,
                "shared_memory_source": None,
            }
        )
        if len(result) > _MAX_ITEMS:
            break
    return result


def parse_nvidia_process_csv(text: str) -> list[dict[str, Any]]:
    """Parse the read-only nvidia-smi compute-process query."""
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in csv.reader(io.StringIO(text or "")):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 3:
            raise ValueError("invalid NVIDIA process row")
        pid = _number(row[0], integer=True)
        if not pid or pid <= 0 or pid in seen:
            raise ValueError("invalid NVIDIA process pid")
        seen.add(pid)
        result.append(
            {
                "pid": pid,
                "process_name": _bounded_process_name(row[1]),
                "dedicated_vram_bytes": _number(
                    row[2], integer=True, scale=_BYTES_PER_MIB
                ),
                "dedicated_vram_source": "nvidia-smi",
                "shared_memory_bytes": None,
                "shared_memory_source": None,
                "gpu_utilization_percent": None,
            }
        )
        if len(result) > _MAX_ITEMS:
            break
    return result


def parse_shared_process_csv(text: str) -> dict[int, int]:
    """Parse the read-only Windows shared GPU memory counter into pid->bytes."""
    result: dict[int, int] = {}
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid shared-memory row")
        try:
            pid = int(fields[0].strip())
            value = int(fields[1].strip())
        except ValueError as exc:
            raise ValueError("invalid shared-memory row") from exc
        if pid <= 0 or value < 0 or pid in result:
            raise ValueError("invalid shared-memory value")
        result[pid] = value
    return result


def merge_gpu_processes(
    dedicated: list[dict[str, Any]],
    shared: dict[int, int],
) -> list[dict[str, Any]]:
    """Join dedicated VRAM and shared memory by pid, keeping both distinct."""
    by_pid = {row["pid"]: dict(row) for row in dedicated}
    for pid, value in shared.items():
        row = by_pid.get(pid)
        if row is None:
            by_pid[pid] = {
                "pid": pid,
                "process_name": "<unavailable>",
                "dedicated_vram_bytes": None,
                "dedicated_vram_source": None,
                "shared_memory_bytes": value,
                "shared_memory_source": "GPU Process Memory Shared Usage",
                "gpu_utilization_percent": None,
            }
        else:
            row["shared_memory_bytes"] = value
            row["shared_memory_source"] = "GPU Process Memory Shared Usage"
    return [by_pid[pid] for pid in sorted(by_pid)]


CommandRunner = Callable[[Sequence[str]], CommandResult]

_POWERSHELL = ("powershell", "-NoProfile", "-Command")


def run_command(command: Sequence[str]) -> CommandResult:
    """Run one read-only command with no shell and a bounded timeout."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(127, "", error="command unavailable")
    except subprocess.TimeoutExpired:
        return CommandResult(124, "", error="command timed out")
    except OSError:
        return CommandResult(1, "", error="command failed")
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _section(
    data: Any, status: str, diagnostic: str | None = None
) -> dict[str, Any]:
    """Return one section envelope; a missing measurement is never zero."""
    result: dict[str, Any] = {"status": status, "data": data}
    if diagnostic is not None:
        result["diagnostics"] = [diagnostic]
    return result


def _requires_windows() -> bool:
    return os.name == "nt"


def _power(script: str) -> list[str]:
    return [*_POWERSHELL, script]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_cpu_counter(text: str) -> dict[str, Any]:
    """Return a bounded, one-shot CPU sample with explicit nulls."""
    utilization: float | None = None
    for line in (text or "").splitlines():
        if line.strip():
            try:
                utilization = float(line.strip())
            except ValueError:
                utilization = None
            break
    return {
        "model": _clean(platform.processor()) or None,
        "logical_cores": os.cpu_count(),
        "utilization_percent": utilization,
        "sampled_at": _timestamp(),
        "source": "Get-Counter Processor(_Total)",
    }


def collect_cpu(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect one bounded CPU utilization sample."""
    if not _requires_windows():
        return _section({}, "unavailable", "cpu source unsupported")
    result = runner(
        _power(
            "(Get-Counter '\\Processor(_Total)\\% Processor Time')"
            ".CounterSamples.CookedValue"
        )
    )
    if result.returncode != 0:
        return _section({}, "unavailable", "cpu source unavailable")
    return _section(parse_cpu_counter(result.stdout), "available")


def parse_memory_json(text: str) -> dict[str, Any]:
    """Parse the read-only operating-system memory payload."""
    payload = json.loads(text or "{}")
    if not isinstance(payload, dict):
        raise ValueError("memory source is not a mapping")
    total_kb = _number(str(payload.get("TotalVisibleMemorySize", "")))
    free_kb = _number(str(payload.get("FreePhysicalMemory", "")))
    total = None if total_kb is None else int(total_kb) * 1024
    available = None if free_kb is None else int(free_kb) * 1024
    used = None if total is None or available is None else total - available
    return {
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "used_percent": (
            None
            if not total or used is None
            else round(100.0 * used / total, 2)
        ),
        "source": "Get-CimInstance Win32_OperatingSystem",
    }


def collect_memory(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect physical memory totals from the read-only CIM query."""
    if not _requires_windows():
        return _section({}, "unavailable", "memory source unsupported")
    result = runner(
        _power(
            "Get-CimInstance Win32_OperatingSystem | Select-Object"
            " TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"
        )
    )
    if result.returncode != 0:
        return _section({}, "unavailable", "memory source unavailable")
    try:
        data = parse_memory_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section({}, "unavailable", "memory data invalid")
    if data["total_bytes"] is None:
        return _section(data, "unavailable", "memory data empty")
    return _section(data, "available")


def collect_gpu(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect per-device dedicated VRAM, utilization, and optional CUDA."""
    result = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,"
            "memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return _section([], "unavailable", "gpu source unavailable")
    try:
        data = parse_nvidia_gpu_csv(result.stdout)
    except ValueError:
        return _section([], "unavailable", "gpu data invalid")
    header = runner(["nvidia-smi"])
    cuda_version = (
        parse_cuda_version(header.stdout) if header.returncode == 0 else None
    )
    for row in data:
        row["cuda_version"] = cuda_version
    if not data:
        return _section(data, "unavailable", "gpu data empty")
    return _section(data, "available")


def collect_gpu_processes(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect per-process dedicated VRAM and, on Windows, shared memory."""
    result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return _section([], "unavailable", "gpu process source unavailable")
    try:
        dedicated = parse_nvidia_process_csv(result.stdout)
    except ValueError:
        return _section([], "unavailable", "gpu process data invalid")
    shared: dict[int, int] = {}
    shared_status: str | None = None
    if _requires_windows():
        shared_result = runner(
            _power(
                "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage'"
                " -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object"
                " { $v=$_.InstanceName; $processId=[int]($v -replace 'pid_',''"
                " -replace '_.*$',''); '{0}{1}{2}' -f $processId, [char]9,"
                " [uint64]$_.CookedValue }"
            )
        )
        if shared_result.returncode != 0:
            shared_status = "shared memory source unavailable"
        else:
            try:
                shared = parse_shared_process_csv(shared_result.stdout)
            except ValueError:
                shared_status = "shared memory data invalid"
    data = merge_gpu_processes(dedicated, shared)
    return _section(
        data,
        "partial" if shared_status is not None else "available",
        shared_status,
    )


def parse_disk_json(text: str) -> list[dict[str, Any]]:
    """Parse the read-only logical-disk payload into per-volume records."""
    payload = json.loads(text or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("disk source is not a list")
    result: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("invalid disk record")
        mount = item.get("DeviceID", item.get("mount_point"))
        total = _number(str(item.get("Size", "")))
        available = _number(str(item.get("FreeSpace", "")))
        if not isinstance(mount, str) or total is None or available is None:
            raise ValueError("invalid disk values")
        result.append(
            {
                "mount_point": _clean(mount),
                "total_bytes": int(total),
                "used_bytes": int(total) - int(available),
                "available_bytes": int(available),
                "source": "Get-CimInstance Win32_LogicalDisk",
            }
        )
        if len(result) > _MAX_ITEMS:
            break
    return result


def collect_disk(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect free local fixed-disk volumes from the read-only CIM query."""
    if not _requires_windows():
        return _section([], "unavailable", "disk source unsupported")
    result = runner(
        _power(
            "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |"
            " Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json -Compress"
        )
    )
    if result.returncode != 0:
        return _section([], "unavailable", "disk source unavailable")
    try:
        data = parse_disk_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section([], "unavailable", "disk data invalid")
    if not data:
        return _section(data, "unavailable", "disk data empty")
    return _section(data, "available")


def parse_network_json(text: str) -> list[dict[str, Any]]:
    """Parse interface names, local addresses, and link state only."""
    payload = json.loads(text or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("network source is not a list")
    result: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("invalid network record")
        raw_addresses = item.get("addresses", [])
        if isinstance(raw_addresses, str):
            raw_addresses = [raw_addresses]
        if not isinstance(raw_addresses, list):
            raise ValueError("invalid network addresses")
        addresses = [
            _clean(address)
            for address in raw_addresses
            if isinstance(address, str) and address
        ]
        result.append(
            {
                "name": _clean(item["name"]),
                "addresses": addresses[:_MAX_ITEMS],
                "link_state": _clean(str(item.get("status", "unknown")))[:40],
            }
        )
        if len(result) > _MAX_ITEMS:
            break
    return result


def collect_network(runner: CommandRunner = run_command) -> dict[str, Any]:
    """Collect local interface configuration; never resolves or connects."""
    if not _requires_windows():
        return _section([], "unavailable", "network source unsupported")
    result = runner(
        _power(
            "Get-NetIPConfiguration -ErrorAction SilentlyContinue | ForEach-Object"
            " { [pscustomobject]@{ name=$_.InterfaceAlias;"
            " status=$_.NetAdapter.Status; addresses=@($_.IPv4Address |"
            " ForEach-Object { $_.IPAddress }) + @($_.IPv6Address |"
            " ForEach-Object { $_.IPAddress }) } } | ConvertTo-Json -Compress"
        )
    )
    if result.returncode != 0:
        return _section([], "unavailable", "network source unavailable")
    try:
        data = parse_network_json(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return _section([], "unavailable", "network data invalid")
    if not data:
        return _section(data, "unavailable", "network interfaces unavailable")
    return _section(data, "available")


SECTION_NAMES = (
    "overview",
    "cpu",
    "memory",
    "gpu",
    "gpu-processes",
    "disk",
    "network",
)

def _overview_status(parts: dict[str, dict[str, Any]]) -> str:
    statuses = [part["status"] for part in parts.values()]
    if all(status == "available" for status in statuses):
        return "available"
    if any(status == "available" for status in statuses):
        return "partial"
    return "unavailable"


def collect_hardware_state(
    sections: list[str],
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Collect only the requested sections; the default is the overview."""
    selected = set(sections)
    if not selected:
        selected = {"overview"}
    if selected - set(SECTION_NAMES) - {"all"}:
        raise ValueError("unknown hardware section")
    if "all" in selected:
        selected = set(SECTION_NAMES)
    collectors: dict[str, Callable[[CommandRunner], dict[str, Any]]] = {
        "cpu": collect_cpu,
        "memory": collect_memory,
        "gpu": collect_gpu,
        "gpu-processes": collect_gpu_processes,
        "disk": collect_disk,
        "network": collect_network,
    }

    result: dict[str, Any] = {}
    for name in SECTION_NAMES:
        if name not in selected:
            continue
        if name == "overview":
            # The overview is a low-cost summary: it never inspects processes.
            parts = {
                "cpu": collect_cpu(runner),
                "memory": collect_memory(runner),
                "gpu": collect_gpu(runner),
                "disk": collect_disk(runner),
            }
            result[name] = {
                "status": _overview_status(parts),
                "data": parts,
            }
        else:
            result[name] = collectors[name](runner)
    return result


def _requested_statuses(sections: dict[str, Any]) -> list[str]:
    return [
        section["status"]
        for section in sections.values()
        if isinstance(section, dict) and "status" in section
    ]


def build_payload(
    sections: list[str],
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Return the JSON envelope for the requested sections."""
    return {
        "schema_version": 1,
        "generated_at": _timestamp(),
        "sections": collect_hardware_state(sections, runner),
    }


def _write_json(payload: dict[str, Any]) -> bool:
    """Write one ASCII JSON document to stdout; never to a file."""
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except (OSError, UnicodeError):
        return False
    return True


class _CliParser(argparse.ArgumentParser):
    """Argument parser whose diagnostics never echo raw argument values."""

    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")


def main(
    argv: list[str] | None = None,
    *,
    runner: CommandRunner = run_command,
) -> int:
    """Return 0 when all requested sections are available, 2 otherwise."""
    parser = _CliParser(description="Read-only hardware state")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--section",
        action="append",
        choices=[*SECTION_NAMES, "all"],
        dest="sections",
    )
    selection.add_argument("--all", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    except ValueError:
        return 1
    requested = ["all"] if args.all else (args.sections or ["overview"])
    try:
        payload = build_payload(requested, runner)
        if not _write_json(payload):
            return 1
    except (OSError, ValueError, TypeError, RuntimeError):
        return 1
    statuses = _requested_statuses(payload["sections"])
    return 2 if any(status in {"unavailable", "partial"} for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
