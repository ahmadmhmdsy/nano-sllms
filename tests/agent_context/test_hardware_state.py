"""Parser contract for the read-only hardware-state command.

These fixtures are fixed strings. No test in this module may call a live
command, read /proc or /sys, query WMI, or open a socket.
"""

import io
import json
import re
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.hardware_state import (
    CommandResult,
    build_payload,
    collect_cpu,
    collect_disk,
    collect_gpu,
    collect_gpu_processes,
    collect_hardware_state,
    collect_memory,
    collect_network,
    main,
    merge_gpu_processes,
    parse_cuda_version,
    parse_disk_json,
    parse_memory_json,
    parse_nvidia_gpu_csv,
    parse_nvidia_process_csv,
    parse_network_json,
    parse_shared_process_csv,
    run_command,
)


MIB = 1024 * 1024

GPU_CSV = (
    "0, GPU-0000, NVIDIA RTX TEST, 999.9, 8192, 2048, 6144, 37, 52, 80.5\n"
    "1, GPU-0001, NVIDIA RTX TEST TWO, 999.9, 16384, [N/A], 16384, 0, 40, 20.0\n"
)

PROCESS_CSV = (
    "1234, C:\\Python\\python.exe, 512\n"
    "5678, C:\\Python\\worker.exe, 256\n"
)

SHARED_CSV = "1234\t268435456\n5678\t134217728\n"


class HardwareParserTests(unittest.TestCase):
    def test_gpu_csv_keeps_dedicated_vram_separate(self):
        gpus = parse_nvidia_gpu_csv(GPU_CSV)
        self.assertEqual(2, len(gpus))
        self.assertEqual(8192 * MIB, gpus[0]["dedicated_vram_total_bytes"])
        self.assertEqual(2048 * MIB, gpus[0]["dedicated_vram_used_bytes"])
        self.assertEqual(6144 * MIB, gpus[0]["dedicated_vram_available_bytes"])
        # Shared GPU memory is never derived from, or folded into, dedicated
        # VRAM: the device level has no shared measurement here.
        self.assertIsNone(gpus[0]["shared_memory_bytes"])
        self.assertIsNone(gpus[0]["shared_memory_source"])
        self.assertEqual("nvidia-smi", gpus[0]["dedicated_vram_source"])
        self.assertEqual(37, gpus[0]["utilization_percent"])
        self.assertEqual(52, gpus[0]["temperature_c"])
        self.assertEqual(80.5, gpus[0]["power_w"])

    def test_gpu_csv_preserves_unavailable_values_as_null(self):
        gpus = parse_nvidia_gpu_csv(GPU_CSV)
        self.assertIsNone(gpus[1]["dedicated_vram_used_bytes"])
        self.assertEqual(16384 * MIB, gpus[1]["dedicated_vram_total_bytes"])

    def test_gpu_csv_rejects_short_row(self):
        with self.assertRaises(ValueError):
            parse_nvidia_gpu_csv("0, GPU-0000, NAME\n")

    def test_gpu_csv_rejects_negative_value(self):
        with self.assertRaises(ValueError):
            parse_nvidia_gpu_csv(
                "0, GPU-0000, NAME, 1.0, 8192, -5, 6144, 0, 40, 20.0\n"
            )

    def test_gpu_csv_rejects_non_numeric_value(self):
        with self.assertRaises(ValueError):
            parse_nvidia_gpu_csv(
                "0, GPU-0000, NAME, 1.0, 8192, abc, 6144, 0, 40, 20.0\n"
            )

    def test_gpu_csv_ignores_blank_lines(self):
        self.assertEqual(2, len(parse_nvidia_gpu_csv(GPU_CSV + "\n   \n")))

    def test_process_csv_and_shared_counter_join_by_pid(self):
        dedicated = parse_nvidia_process_csv(PROCESS_CSV)
        shared = parse_shared_process_csv(SHARED_CSV)
        processes = merge_gpu_processes(dedicated, shared)
        by_pid = {row["pid"]: row for row in processes}
        self.assertEqual(512 * MIB, by_pid[1234]["dedicated_vram_bytes"])
        self.assertEqual(268435456, by_pid[1234]["shared_memory_bytes"])
        self.assertNotEqual(
            by_pid[1234]["dedicated_vram_bytes"],
            by_pid[1234]["shared_memory_bytes"],
        )
        self.assertEqual("nvidia-smi", by_pid[1234]["dedicated_vram_source"])
        self.assertEqual(
            "GPU Process Memory Shared Usage",
            by_pid[1234]["shared_memory_source"],
        )

    def test_process_names_are_reduced_to_basenames(self):
        rows = parse_nvidia_process_csv(PROCESS_CSV)
        self.assertEqual("python.exe", rows[0]["process_name"])
        self.assertEqual("worker.exe", rows[1]["process_name"])

    def test_process_csv_rejects_duplicate_pid(self):
        with self.assertRaises(ValueError):
            parse_nvidia_process_csv("1234, a.exe, 1\n1234, b.exe, 2\n")

    def test_process_csv_rejects_non_positive_pid(self):
        with self.assertRaises(ValueError):
            parse_nvidia_process_csv("0, a.exe, 1\n")

    def test_process_csv_rejects_short_row(self):
        with self.assertRaises(ValueError):
            parse_nvidia_process_csv("1234, a.exe\n")

    def test_process_csv_rejects_negative_memory(self):
        with self.assertRaises(ValueError):
            parse_nvidia_process_csv("1234, a.exe, -7\n")

    def test_process_name_is_bounded_and_redacted_when_secret_shaped(self):
        long_name = "x" * 400 + ".exe"
        rows = parse_nvidia_process_csv("1234, " + long_name + ", 8\n")
        self.assertEqual(160, len(rows[0]["process_name"]))
        secret = parse_nvidia_process_csv("1234, api_key=abcd1234efgh5678.exe, 8\n")
        self.assertEqual("<redacted-process-name>", secret[0]["process_name"])
        control = parse_nvidia_process_csv("1234, a\u0007b.exe, 8\n")
        self.assertEqual("<redacted-process-name>", control[0]["process_name"])

    def test_shared_csv_parses_tab_separated_bytes(self):
        shared = parse_shared_process_csv(SHARED_CSV)
        self.assertEqual({1234: 268435456, 5678: 134217728}, shared)

    def test_shared_csv_rejects_duplicate_pid(self):
        with self.assertRaises(ValueError):
            parse_shared_process_csv("1234\t10\n1234\t20\n")

    def test_shared_csv_rejects_non_positive_pid(self):
        with self.assertRaises(ValueError):
            parse_shared_process_csv("0\t10\n")

    def test_shared_csv_rejects_negative_bytes(self):
        with self.assertRaises(ValueError):
            parse_shared_process_csv("1234\t-1\n")

    def test_shared_csv_rejects_malformed_row(self):
        with self.assertRaises(ValueError):
            parse_shared_process_csv("1234 10\n")

    def test_unmatched_pid_keeps_missing_side_null(self):
        processes = merge_gpu_processes(
            parse_nvidia_process_csv(PROCESS_CSV),
            {1234: 268435456},
        )
        by_pid = {row["pid"]: row for row in processes}
        self.assertEqual(268435456, by_pid[1234]["shared_memory_bytes"])
        self.assertIsNone(by_pid[5678]["shared_memory_bytes"])
        self.assertIsNone(by_pid[5678]["shared_memory_source"])

    def test_shared_only_pid_is_reported_without_dedicated_value(self):
        processes = merge_gpu_processes([], {99: 4096})
        self.assertEqual(1, len(processes))
        self.assertEqual(99, processes[0]["pid"])
        self.assertIsNone(processes[0]["dedicated_vram_bytes"])
        self.assertIsNone(processes[0]["dedicated_vram_source"])
        self.assertEqual(4096, processes[0]["shared_memory_bytes"])
        self.assertEqual("<unavailable>", processes[0]["process_name"])

    def test_merged_processes_are_sorted_by_pid(self):
        processes = merge_gpu_processes(
            parse_nvidia_process_csv(PROCESS_CSV), {99: 1}
        )
        self.assertEqual([99, 1234, 5678], [row["pid"] for row in processes])

    def test_empty_inputs_return_empty_results(self):
        self.assertEqual([], parse_nvidia_gpu_csv(""))
        self.assertEqual([], parse_nvidia_process_csv(""))
        self.assertEqual({}, parse_shared_process_csv(""))

    def test_cuda_version_is_parsed_from_the_header(self):
        header = (
            "Fri Sep 25 10:00:00 2026\n"
            "+----------------------------------------------------------------+\n"
            "| NVIDIA-SMI 560.00   Driver Version: 560.00   CUDA Version: 12.6 |\n"
        )
        self.assertEqual("12.6", parse_cuda_version(header))

    def test_cuda_version_is_null_when_absent(self):
        self.assertIsNone(parse_cuda_version(""))
        self.assertIsNone(parse_cuda_version("no cuda here"))




MEMORY_JSON = '{"TotalVisibleMemorySize": 33554432, "FreePhysicalMemory": 16777216}'
DISK_JSON = (
    '[{"DeviceID": "C:", "Size": 1000, "FreeSpace": 400},'
    ' {"DeviceID": "D:", "Size": 2000, "FreeSpace": 1500}]'
)
NETWORK_JSON = (
    '[{"name": "Ethernet", "status": "Up",'
    ' "addresses": ["10.0.0.5", "fe80::1"]},'
    ' {"name": "Wi-Fi", "status": "Disconnected", "addresses": []}]'
)
SHARED_COUNTER = "1234\t268435456\n5678\t134217728\n"
CUDA_HEADER = "NVIDIA-SMI 560.00   Driver Version: 560.00   CUDA Version: 12.6"

# Each entry pins one read-only source marker. A collector must query
# exactly the source named here, so the marker is the contract under test.
CPU_SOURCE = "Processor(_Total)"
MEMORY_SOURCE = "Win32_OperatingSystem"
DISK_SOURCE = "Win32_LogicalDisk"
NETWORK_SOURCE = "Get-NetIPConfiguration"
SHARED_SOURCE = "GPU Process Memory"
GPU_SOURCE = "--query-gpu="
GPU_PROCESS_SOURCE = "--query-compute-apps="

ALL_SOURCES = (
    CPU_SOURCE,
    MEMORY_SOURCE,
    DISK_SOURCE,
    NETWORK_SOURCE,
    SHARED_SOURCE,
    GPU_SOURCE,
    GPU_PROCESS_SOURCE,
)

# Values returned by the fake runner for each source marker.
SOURCE_OUTPUTS = {
    CPU_SOURCE: CommandResult(0, "12.5\n"),
    MEMORY_SOURCE: CommandResult(0, MEMORY_JSON),
    DISK_SOURCE: CommandResult(0, DISK_JSON),
    NETWORK_SOURCE: CommandResult(0, NETWORK_JSON),
    SHARED_SOURCE: CommandResult(0, SHARED_COUNTER),
    GPU_SOURCE: CommandResult(0, GPU_CSV),
    GPU_PROCESS_SOURCE: CommandResult(0, PROCESS_CSV),
}


class RecordingRunner:
    """Fake read-only command runner keyed by a source marker.

    A command matches when any argument contains the marker. The bare
    "nvidia-smi" header call is matched by the sentinel below.
    """

    HEADER_MARKER = "<nvidia-smi-header>"

    def __init__(self, sources=None, **overrides):
        self.responses = dict(sources or {})
        self.responses.update(overrides)
        self.commands = []

    def __call__(self, command):
        self.commands.append(tuple(command))
        joined = " ".join(command)
        if len(command) == 1:
            return self.responses.get(
                self.HEADER_MARKER, CommandResult(1, "", "unavailable")
            )
        for marker, result in self.responses.items():
            if marker in joined:
                return result
        return CommandResult(1, "", "unavailable")

    def markers_used(self):
        return [
            marker
            for marker in ALL_SOURCES
            if any(marker in " ".join(command) for command in self.commands)
        ]


def _all_sources(**overrides):
    """Return a runner serving every read-only source, with overrides."""
    responses = dict(SOURCE_OUTPUTS)
    responses[RecordingRunner.HEADER_MARKER] = CommandResult(0, CUDA_HEADER)
    responses.update(overrides)
    return RecordingRunner(responses)


def _sources(*markers, **overrides):
    """Return a runner serving only the named sources, with overrides."""
    responses = {
        marker: SOURCE_OUTPUTS[marker]
        for marker in markers
        if marker in SOURCE_OUTPUTS
    }
    if any(marker == GPU_SOURCE for marker in markers):
        responses[RecordingRunner.HEADER_MARKER] = CommandResult(0, CUDA_HEADER)
    responses.update(overrides)
    return RecordingRunner(responses)



class HardwareCollectorTests(unittest.TestCase):
    """Every collector is exercised through an injected fake runner only."""

    def test_run_command_never_uses_a_shell(self):
        with patch("scripts.hardware_state.subprocess.run") as run:
            run.return_value = CommandResult(0, "out", "err")
            result = run_command(["nvidia-smi", "--query-gpu=index"])
        self.assertIsNot(run.return_value, result)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(0, result.returncode)
        self.assertEqual(["nvidia-smi", "--query-gpu=index"], run.call_args.args[0])

    def test_run_command_reports_a_missing_command_without_stderr(self):
        with patch(
            "scripts.hardware_state.subprocess.run",
            side_effect=FileNotFoundError("missing"),
        ):
            result = run_command(["nvidia-smi"])
        self.assertEqual(127, result.returncode)
        self.assertEqual("command unavailable", result.error)
        self.assertEqual("", result.stdout)

    def test_run_command_reports_a_timeout_without_stderr(self):
        with patch(
            "scripts.hardware_state.subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 30),
        ):
            result = run_command(["nvidia-smi"])
        self.assertEqual(124, result.returncode)
        self.assertEqual("command timed out", result.error)
        self.assertEqual("", result.stdout)

    def test_run_command_reports_an_os_error_without_stderr(self):
        with patch(
            "scripts.hardware_state.subprocess.run",
            side_effect=OSError("denied"),
        ):
            result = run_command(["nvidia-smi"])
        self.assertEqual(1, result.returncode)
        self.assertEqual("command failed", result.error)
        self.assertEqual("", result.stdout)

    def test_collect_cpu_reports_utilization(self):
        runner = _sources(CPU_SOURCE)
        section = collect_cpu(runner)
        self.assertEqual("available", section["status"])
        self.assertEqual(12.5, section["data"]["utilization_percent"])
        self.assertIsNotNone(section["data"]["logical_cores"])
        self.assertTrue(section["data"]["sampled_at"].endswith("Z"))
        self.assertEqual(1, len(runner.commands))

    def test_collect_cpu_keeps_unreadable_output_null(self):
        runner = _sources(CPU_SOURCE, **{CPU_SOURCE: CommandResult(0, "not-a-number")})
        section = collect_cpu(runner)
        self.assertEqual("available", section["status"])
        self.assertIsNone(section["data"]["utilization_percent"])

    def test_collect_cpu_is_unavailable_when_the_source_fails(self):
        section = collect_cpu(_sources())
        self.assertEqual("unavailable", section["status"])
        self.assertEqual({}, section["data"])
        self.assertEqual(["cpu source unavailable"], section["diagnostics"])

    def test_collect_cpu_uses_the_total_processor_counter(self):
        runner = _sources(CPU_SOURCE)
        collect_cpu(runner)
        self.assertIn("Processor(_Total)", " ".join(runner.commands[0]))

    def test_collect_memory_reports_bytes_and_percent(self):
        section = collect_memory(_sources(MEMORY_SOURCE))
        self.assertEqual("available", section["status"])
        data = section["data"]
        self.assertEqual(33554432 * 1024, data["total_bytes"])
        self.assertEqual(16777216 * 1024, data["available_bytes"])
        self.assertEqual(16777216 * 1024, data["used_bytes"])
        self.assertEqual(50.0, data["used_percent"])

    def test_collect_memory_keeps_missing_fields_null(self):
        section = collect_memory(
            _sources(
                MEMORY_SOURCE,
                **{MEMORY_SOURCE: CommandResult(0, '{"FreePhysicalMemory": 1}')},
            )
        )
        self.assertEqual("unavailable", section["status"])
        self.assertIsNone(section["data"]["total_bytes"])
        self.assertIsNone(section["data"]["used_percent"])

    def test_collect_memory_is_unavailable_for_invalid_output(self):
        section = collect_memory(
            _sources(MEMORY_SOURCE, **{MEMORY_SOURCE: CommandResult(0, "[1,2]")})
        )
        self.assertEqual("unavailable", section["status"])
        self.assertEqual(["memory data invalid"], section["diagnostics"])

    def test_collect_memory_rejects_non_mapping_payloads(self):
        with self.assertRaises(ValueError):
            parse_memory_json("[1, 2]")

    def test_collect_disk_reports_every_local_volume(self):
        section = collect_disk(_sources(DISK_SOURCE))
        self.assertEqual("available", section["status"])
        self.assertEqual(2, len(section["data"]))
        self.assertEqual("C:", section["data"][0]["mount_point"])
        self.assertEqual(1000, section["data"][0]["total_bytes"])
        self.assertEqual(600, section["data"][0]["used_bytes"])
        self.assertEqual("D:", section["data"][1]["mount_point"])

    def test_collect_disk_accepts_a_single_object_payload(self):
        data = parse_disk_json('{"DeviceID": "C:", "Size": 10, "FreeSpace": 4}')
        self.assertEqual(1, len(data))
        self.assertEqual(6, data[0]["used_bytes"])

    def test_collect_disk_rejects_a_non_list_payload(self):
        with self.assertRaises(ValueError):
            parse_disk_json('"nope"')

    def test_collect_disk_queries_fixed_local_volumes_only(self):
        runner = _sources(DISK_SOURCE)
        collect_disk(runner)
        joined = " ".join(runner.commands[0])
        self.assertIn("DriveType=3", joined)

    def test_collect_network_reports_bounded_addresses(self):
        section = collect_network(_sources(NETWORK_SOURCE))
        self.assertEqual("available", section["status"])
        self.assertEqual(
            [
                {
                    "name": "Ethernet",
                    "addresses": ["10.0.0.5", "fe80::1"],
                    "link_state": "Up",
                },
                {
                    "name": "Wi-Fi",
                    "addresses": [],
                    "link_state": "Disconnected",
                },
            ],
            section["data"],
        )

    def test_collect_network_never_resolves_or_connects(self):
        runner = _sources(NETWORK_SOURCE)
        collect_network(runner)
        joined = " ".join(" ".join(command) for command in runner.commands)
        for forbidden in (
            "Resolve-DnsName",
            "nslookup",
            "ping",
            "Test-NetConnection",
            "Invoke-WebRequest",
            "Get-NetTCPConnection",
        ):
            self.assertNotIn(forbidden, joined)

    def test_collect_network_rejects_a_non_list_payload(self):
        with self.assertRaises(ValueError):
            parse_network_json('"nope"')

    def test_collect_gpu_reports_devices_and_optional_cuda(self):
        section = collect_gpu(_sources(GPU_SOURCE))
        self.assertEqual("available", section["status"])
        self.assertEqual(2, len(section["data"]))
        self.assertEqual("12.6", section["data"][0]["cuda_version"])
        self.assertIsNone(section["data"][0]["shared_memory_bytes"])
        self.assertIsNone(section["data"][0]["shared_memory_source"])

    def test_collect_gpu_keeps_device_data_without_a_cuda_header(self):
        section = collect_gpu(
            _sources(
                GPU_SOURCE,
                **{
                    RecordingRunner.HEADER_MARKER: CommandResult(1, "", "no header")
                },
            )
        )
        self.assertEqual("available", section["status"])
        self.assertIsNone(section["data"][0]["cuda_version"])

    def test_collect_gpu_is_unavailable_without_output(self):
        section = collect_gpu(_sources())
        self.assertEqual("unavailable", section["status"])
        self.assertEqual([], section["data"])
        self.assertEqual(["gpu source unavailable"], section["diagnostics"])

    def test_collect_gpu_is_unavailable_for_invalid_output(self):
        section = collect_gpu(
            _sources(GPU_SOURCE, **{GPU_SOURCE: CommandResult(0, "garbage")})
        )
        self.assertEqual("unavailable", section["status"])
        self.assertEqual(["gpu data invalid"], section["diagnostics"])

    def test_collect_gpu_queries_the_device_fields(self):
        runner = _sources(GPU_SOURCE)
        collect_gpu(runner)
        joined = " ".join(runner.commands[0])
        for field in (
            "memory.total",
            "memory.used",
            "memory.free",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
        ):
            self.assertIn(field, joined)
        self.assertIn("--format=csv,noheader,nounits", joined)

    def test_collect_gpu_processes_keeps_dedicated_and_shared_separate(self):
        section = collect_gpu_processes(_sources(GPU_PROCESS_SOURCE, SHARED_SOURCE))
        self.assertEqual("available", section["status"])
        by_pid = {row["pid"]: row for row in section["data"]}
        self.assertEqual(512 * 1024 * 1024, by_pid[1234]["dedicated_vram_bytes"])
        self.assertEqual(268435456, by_pid[1234]["shared_memory_bytes"])
        self.assertNotEqual(
            by_pid[1234]["dedicated_vram_bytes"],
            by_pid[1234]["shared_memory_bytes"],
        )
        self.assertIsNone(by_pid[5678]["gpu_utilization_percent"])

    def test_collect_gpu_processes_is_partial_without_the_shared_counter(self):
        section = collect_gpu_processes(_sources(GPU_PROCESS_SOURCE))
        self.assertEqual("partial", section["status"])
        self.assertEqual(
            ["shared memory source unavailable"], section["diagnostics"]
        )
        for row in section["data"]:
            self.assertIsNone(row["shared_memory_bytes"])
            self.assertIsNotNone(row["dedicated_vram_bytes"])

    def test_collect_gpu_processes_is_partial_for_invalid_shared_output(self):
        section = collect_gpu_processes(
            _sources(
                GPU_PROCESS_SOURCE,
                **{SHARED_SOURCE: CommandResult(0, "garbage")},
            )
        )
        self.assertEqual("partial", section["status"])
        self.assertEqual(["shared memory data invalid"], section["diagnostics"])

    def test_collect_gpu_processes_is_unavailable_without_a_gpu_source(self):
        section = collect_gpu_processes(_sources())
        self.assertEqual("unavailable", section["status"])
        self.assertEqual(
            ["gpu process source unavailable"], section["diagnostics"]
        )

    def test_collect_hardware_state_defaults_to_overview_without_processes(self):
        runner = _all_sources()
        sections = collect_hardware_state([], runner)
        self.assertEqual(["overview"], list(sections))
        self.assertEqual(
            [CPU_SOURCE, MEMORY_SOURCE, DISK_SOURCE, GPU_SOURCE],
            runner.markers_used(),
        )
        self.assertNotIn(GPU_PROCESS_SOURCE, runner.markers_used())
        self.assertNotIn(SHARED_SOURCE, runner.markers_used())
        self.assertNotIn(NETWORK_SOURCE, runner.markers_used())

    def test_collect_hardware_state_selects_only_requested_sections(self):
        runner = _all_sources()
        sections = collect_hardware_state(["cpu", "memory"], runner)
        self.assertEqual(["cpu", "memory"], list(sections))
        self.assertEqual([CPU_SOURCE, MEMORY_SOURCE], runner.markers_used())

    def test_collect_hardware_state_all_expands_to_every_section(self):
        sections = collect_hardware_state(["all"], _all_sources())
        self.assertEqual(
            [
                "overview",
                "cpu",
                "memory",
                "gpu",
                "gpu-processes",
                "disk",
                "network",
            ],
            list(sections),
        )

    def test_collect_hardware_state_rejects_unknown_sections(self):
        with self.assertRaises(ValueError):
            collect_hardware_state(["all", "bogus"], _all_sources())
        with self.assertRaises(ValueError):
            collect_hardware_state(["bogus"], _all_sources())

    def test_overview_status_reflects_partial_sources(self):
        runner = _all_sources(
            **{CPU_SOURCE: CommandResult(1, "", "unavailable")}
        )
        sections = collect_hardware_state(["overview"], runner)
        self.assertEqual("partial", sections["overview"]["status"])
        self.assertEqual(
            ["cpu", "memory", "gpu", "disk"],
            list(sections["overview"]["data"]),
        )

    def test_overview_status_is_unavailable_when_all_sources_fail(self):
        runner = _sources()
        sections = collect_hardware_state(["overview"], runner)
        self.assertEqual("unavailable", sections["overview"]["status"])

    def test_unsupported_platform_sections_are_explicit(self):
        with patch("scripts.hardware_state.os.name", "posix"):
            for collector in (
                collect_cpu,
                collect_memory,
                collect_disk,
                collect_network,
            ):
                section = collector(_all_sources())
                self.assertEqual("unavailable", section["status"])
                self.assertTrue(section["diagnostics"][0].endswith("unsupported"))

    def test_build_payload_wraps_sections_with_metadata(self):
        payload = build_payload(["memory"], _all_sources())
        self.assertEqual(1, payload["schema_version"])
        self.assertTrue(payload["generated_at"].endswith("Z"))
        self.assertEqual(["memory"], list(payload["sections"]))


class HardwareCliTests(unittest.TestCase):
    """The CLI is driven with an injected runner so no probe can run."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _run(self, argv, runner=None):
        runner = _all_sources() if runner is None else runner
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(argv, runner=runner)
        return code, stdout.getvalue(), runner

    def test_cli_default_overview_returns_zero_when_available(self):
        code, output, runner = self._run([])
        self.assertEqual(0, code)
        payload = json.loads(output)
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual(["overview"], list(payload["sections"]))
        self.assertNotIn("gpu-processes", payload["sections"])

    def test_cli_returns_two_when_a_requested_section_is_unavailable(self):
        code, output, _ = self._run(["--section", "gpu"], _sources())
        self.assertEqual(2, code)
        self.assertIn("gpu source unavailable", output)

    def test_cli_returns_two_for_a_partial_section(self):
        code, output, _ = self._run(
            ["--section", "gpu-processes"], _sources(GPU_PROCESS_SOURCE)
        )
        self.assertEqual(2, code)
        self.assertIn("shared memory source unavailable", output)

    def test_cli_returns_zero_for_selective_available_sections(self):
        code, output, runner = self._run(
            ["--section", "cpu", "--section", "memory"]
        )
        self.assertEqual(0, code)
        payload = json.loads(output)
        self.assertEqual(["cpu", "memory"], list(payload["sections"]))
        self.assertEqual([CPU_SOURCE, MEMORY_SOURCE], runner.markers_used())

    def test_cli_all_includes_gpu_processes(self):
        code, output, _ = self._run(["--all"])
        self.assertEqual(0, code)
        payload = json.loads(output)
        self.assertIn("gpu-processes", payload["sections"])
        self.assertIn("network", payload["sections"])

    def test_cli_output_is_ascii_only_json(self):
        code, output, _ = self._run(["--all"])
        self.assertEqual(0, code)
        output.encode("ascii")
        self.assertEqual(1, json.loads(output)["schema_version"])

    def test_cli_rejects_unknown_arguments_without_probing(self):
        code, _, runner = self._run(["--section", "bogus"])
        self.assertEqual(1, code)
        self.assertEqual([], runner.commands)

    def test_cli_rejects_conflicting_selection_without_probing(self):
        code, _, runner = self._run(["--all", "--section", "cpu"])
        self.assertEqual(1, code)
        self.assertEqual([], runner.commands)

    def test_cli_help_returns_zero_without_probing(self):
        code, _, runner = self._run(["--help"])
        self.assertEqual(0, code)
        self.assertEqual([], runner.commands)

    def test_cli_returns_one_on_an_internal_failure(self):
        def exploding_runner(command):
            raise RuntimeError("internal detail")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--section", "cpu"], runner=exploding_runner)
        self.assertEqual(1, code)
        self.assertNotIn("Traceback", stdout.getvalue())
        self.assertNotIn("internal detail", stdout.getvalue())

    def test_cli_returns_one_when_output_cannot_be_written(self):
        class BrokenStdout:
            encoding = "ascii"

            def write(self, text):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        with patch("scripts.hardware_state.sys.stdout", BrokenStdout()):
            self.assertEqual(
                1, main(["--section", "cpu"], runner=_all_sources())
            )

    def test_cli_writes_no_file(self):
        before = sorted(p.name for p in self.root.rglob("*"))
        self._run(["--all"])
        self.assertEqual(before, sorted(p.name for p in self.root.rglob("*")))

    def test_cli_timeout_produces_no_raw_stderr_or_traceback(self):
        code, output, _ = self._run(
            ["--section", "gpu"],
            _sources(
                GPU_SOURCE,
                **{
                    GPU_SOURCE: CommandResult(124, "", "SECRET stderr text"),
                    RecordingRunner.HEADER_MARKER: CommandResult(0, CUDA_HEADER),
                },
            ),
        )
        self.assertEqual(2, code)
        self.assertNotIn("SECRET stderr text", output)
        self.assertNotIn("Traceback", output)

    def test_collectors_write_nothing_to_disk(self):
        before = sorted(p.name for p in self.root.rglob("*"))
        collect_hardware_state(["all"], _all_sources())
        self.assertEqual(before, sorted(p.name for p in self.root.rglob("*")))




class HardwareDocumentationTests(unittest.TestCase):
    """The agent-facing guide must document the real command contract."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    def test_hardware_guide_documents_the_contract(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(
            encoding="utf-8"
        )
        for required in (
            r"scripts\hardware_state.py",
            "--section",
            "--all",
            "overview",
            "dedicated_vram",
            "shared_memory",
            "gpu-processes",
            "unavailable",
            "partial",
            "per-process",
            "user-opened",
            "global",
        ):
            self.assertIn(required, text)

    def test_hardware_guide_states_the_safety_boundary(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "read-only",
            "never",
            "does not stop",
            "does not write",
            "does not contact",
        ):
            self.assertIn(required, text)

    def test_hardware_guide_separates_dedicated_and_shared_memory(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("never added together", text)
        self.assertIn("shared-memory", text)

    def test_hardware_guide_uses_the_repo_venv(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(r".\.venv\Scripts\python.exe", text)
        self.assertNotIn("pip install", text)

    def test_hardware_guide_contains_no_live_machine_values(self):
        text = (self.root / "docs/agent/HARDWARE_STATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(
            re.search(r"\bpid\s+#?\s*\d+", text, re.IGNORECASE),
            "the guide must not claim a live pid",
        )
        self.assertNotIn("RTX", text)
        self.assertNotIn("GeForce", text)

    def test_router_and_operations_link_to_hardware_guide(self):
        for relative in ("AGENTS.md", "docs/agent/OPERATIONS.md"):
            text = (self.root / relative).read_text(encoding="utf-8")
            self.assertIn("docs/agent/HARDWARE_STATE.md", text)

    def test_schema_documents_the_exception_registry(self):
        text = (self.root / "docs/agent/SCHEMA.md").read_text(encoding="utf-8")
        for required in (
            "context_exceptions.yaml",
            "schema_version",
            "accepted_historical",
            "decision_source",
            "malformed_table_row",
            "broken_markdown_link",
            "fail",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
