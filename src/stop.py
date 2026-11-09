"""U11 cooperative stop: checkpoint-aligned flag file (WEBUI_PRD U11).

The UI never kills a live run (PLAN hard rule). Instead it drops a STOP
file inside the trainer's output_dir; the trainer checks it only at save
boundaries (on_save), so a stop always leaves a valid checkpoint behind.
Relaunching the exact same command clears the flag and resumes - no loss
bump beyond the normal optimizer-state restore.
"""
from pathlib import Path

from transformers import TrainerCallback


def stop_flag_path(output_dir) -> Path:
    return Path(output_dir) / "STOP"


def clear_stop_flag(output_dir) -> bool:
    flag = stop_flag_path(output_dir)
    if flag.is_file():
        flag.unlink()
        return True
    return False


class CoopStopCallback(TrainerCallback):
    """Sets should_training_stop only right after a checkpoint save."""

    def on_save(self, args, state, control, **kwargs):
        if stop_flag_path(args.output_dir).is_file():
            print(f"[stop] STOP flag seen after checkpoint-{state.global_step} "
                  "-> cooperative stop (valid checkpoint on disk)", flush=True)
            control.should_training_stop = True
        return control
