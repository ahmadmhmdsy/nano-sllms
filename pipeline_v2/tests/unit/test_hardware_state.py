"""Tests for hardware state report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestHardwareState:
    """Tests for Task 6.2: hardware state report."""

    def test_hardware_state_command_exists(self):
        """Verify the hardware-state command is registered."""
        from slm_train.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["hardware-state"])
        assert args.command == "hardware-state"

    def test_hardware_state_output(self, capsys):
        """Verify hardware-state prints a report."""
        from slm_train.cli import main

        exit_code = main(["hardware-state"])
        assert exit_code == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["status"] == "PASS"
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data
        assert "gpu" in data
