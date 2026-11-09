"""Tests for dry-run and observability features."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestDryRun:
    """Tests for Task 6.1: dry-run in the plan command."""

    def test_plan_dry_run_flag_exists(self):
        """Verify the --dry-run flag is accepted by the plan command."""
        from slm_train.cli import build_parser

        parser = build_parser()
        # Parse with --dry-run
        args = parser.parse_args(["plan", "--config", "test.yaml", "--dry-run"])
        assert args.dry_run is True
        assert args.config == "test.yaml"

    def test_plan_dry_run_output_contains_config(self, tmp_path, capsys):
        """Verify plan --dry-run prints the resolved config."""
        from slm_train.cli import main

        # Use an existing config file
        repo_root = Path(__file__).resolve().parents[3]
        config_file = repo_root / "pipeline_v2" / "configs" / "base" / "r1.yaml"

        # Run plan --dry-run
        exit_code = main(["plan", "--config", str(config_file), "--dry-run"])
        assert exit_code == 0

        # Check output contains resolved config
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["status"] == "PASS"
        assert "resolved_config" in data
        assert data["resolved_config"]["run"]["name"] == "replace_me"
        assert data["resolved_config"]["train"]["gradient_accumulation"] == 8
        assert "run_command" in data
