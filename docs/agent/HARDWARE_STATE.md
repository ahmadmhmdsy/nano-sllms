# Hardware state — selective, read-only local probe

'scripts/hardware_state.py' reports the current machine's hardware state as
one JSON document on standard output. It exists so an agent can answer a
narrow question ('how much dedicated VRAM is free?', 'is a process spilling
into shared memory?') without a heavyweight audit.

## When to run it

Only inside a hardware window the user has explicitly opened. This is a local
measurement, not a global process, network, or adoption audit, and it does
not satisfy that separate audit.

## Commands

~~~powershell
& .\.venv\Scripts\python.exe scripts\hardware_state.py
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section cpu --section memory
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section gpu --section gpu-processes
& .\.venv\Scripts\python.exe scripts\hardware_state.py --all
~~~

- No flag runs the default 'overview': CPU, memory, GPU, and disk only.
- '--section' may be repeated. Valid names: 'overview', 'cpu', 'memory',
  'gpu', 'gpu-processes', 'disk', 'network'.
- '--all' requests every section, including per-process GPU state.
- '--section' and '--all' are mutually exclusive.

Per-process detail is opt-in: it is collected only for 'gpu-processes' or
'--all'.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | every requested section is 'available' |
| 2 | a requested section is 'unavailable' or 'partial'; JSON was still written |
| 1 | invalid arguments, or output could not be written |

A non-zero exit code does not mean the command failed. Read the JSON.

## Sections and status values

Every section has a 'status' and a 'data' field, plus a bounded
'diagnostics' list when something was unavailable.

- 'available' — the source answered and the measurement is complete.
- 'partial' — part of the requested data is missing but the rest is usable.
- 'unavailable' — the source could not be read at all; 'data' is empty.

A value that could not be measured is reported as null, never as 0, so a
missing reading is never mistaken for an idle device.

## Dedicated VRAM versus shared GPU memory

These are two different physical things and the command never merges them.

- **Dedicated VRAM** ('dedicated_vram_*') is memory physically attached to the
  GPU. When it fills up, the workload slows because the GPU has no room left.
- **Shared GPU memory** ('shared_memory_bytes') is ordinary system RAM that a
  GPU process borrows when dedicated VRAM runs short. Windows reports it per
  process through the 'GPU Process Memory' shared-usage counter.

The two values are never added together, and one is never substituted for the
other. A rising 'shared_memory_bytes' value for a process is the signal that
the workload is spilling into system memory, a common cause of a
shared-memory slowdown that otherwise looks like an unrelated CPU problem.

## Per-process GPU rows

'gpu-processes' reports one row per process id with these fields:

| Field | Meaning |
| --- | --- |
| 'pid' | process id |
| 'process_name' | executable basename, bounded and redacted if secret-shaped |
| 'dedicated_vram_bytes' | dedicated VRAM this process holds, or null |
| 'shared_memory_bytes' | shared GPU memory this process holds, or null |
| 'shared_memory_source' | which counter produced the shared value |
| 'gpu_utilization_percent' | null; no per-process utilization source is used |

Rows are joined by process id, so a process that appears in only one source
keeps null on the missing side instead of being reported as zero. Seeing a
process here tells you that GPU work is co-running. It does not establish
ownership, and it never authorizes stopping anything.

## Safety boundary

The command is read-only and local. It does not write any file, does not
modify repository state or generated documents, does not stop, kill, or
signal any process, and does not contact the network or resolve a host name.
Its only sources are local counters and read-only queries: 'nvidia-smi',
'Get-Counter', 'Get-CimInstance', and 'Get-NetIPConfiguration'.

Only run it in a user-opened hardware window, and keep the label SKIPPED
when that window was not opened.

## Limitations

- Windows-only sources. On another platform each Windows-backed section
  reports 'unavailable' with an explicit diagnostic rather than guessing.
- Values are a single live sample. Nothing is cached, averaged, or monitored
  over time.
- The shared-memory counter depends on the Windows performance-counter set
  being present; when it is not, 'gpu-processes' is 'partial'.
- Interface addresses are configuration, not traffic: no throughput, no
  connection state, and no reachability.
