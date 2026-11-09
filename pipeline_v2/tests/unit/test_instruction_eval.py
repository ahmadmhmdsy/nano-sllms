"""Tests for instruction-style evaluation."""
from __future__ import annotations

import pytest


class TestInstructionEval:
    """Tests for Task 7.2: instruction-style eval."""

    def test_instruction_metrics_perfect(self):
        """Verify perfect instruction following scores 1.0."""
        from slm_train.evaluation.instruction import instruction_metrics

        responses = [
            {
                "instruction": "List three colors",
                "response": "Red, Green, Blue",
                "required_elements": ["Red", "Green", "Blue"],
            }
        ]
        result = instruction_metrics(responses)
        assert result["instruction_following"] == 1.0
        assert result["mean_elements_found"] == 1.0

    def test_instruction_metrics_partial(self):
        """Verify partial instruction following scores correctly."""
        from slm_train.evaluation.instruction import instruction_metrics

        responses = [
            {
                "instruction": "List three colors",
                "response": "Red, Green",
                "required_elements": ["Red", "Green", "Blue"],
            }
        ]
        result = instruction_metrics(responses)
        assert result["instruction_following"] == 0.0
        assert result["mean_elements_found"] == pytest.approx(2 / 3)

    def test_instruction_metrics_empty_required(self):
        """Verify empty required elements means trivially satisfied."""
        from slm_train.evaluation.instruction import instruction_metrics

        responses = [
            {
                "instruction": "Say hello",
                "response": "Hello!",
                "required_elements": [],
            }
        ]
        result = instruction_metrics(responses)
        assert result["instruction_following"] == 1.0
        assert result["mean_elements_found"] == 1.0

    def test_instruction_metrics_empty_responses(self):
        """Verify empty responses raises an error."""
        from slm_train.evaluation.instruction import instruction_metrics

        with pytest.raises(ValueError, match="instruction eval needs at least one response"):
            instruction_metrics([])

    def test_check_instruction_format_json(self):
        """Verify JSON format detection."""
        from slm_train.evaluation.instruction import check_instruction_format

        assert check_instruction_format('{"key": "value"}', "json") is True
        assert check_instruction_format("not json", "json") is False

    def test_check_instruction_format_list(self):
        """Verify list format detection."""
        from slm_train.evaluation.instruction import check_instruction_format

        assert check_instruction_format("1. First\n2. Second", "list") is True
        assert check_instruction_format("- Item 1\n- Item 2", "list") is True
        assert check_instruction_format("Just text", "list") is False

    def test_check_instruction_format_code(self):
        """Verify code format detection."""
        from slm_train.evaluation.instruction import check_instruction_format

        assert check_instruction_format("```python\nprint('hello')\n```", "code") is True
        assert check_instruction_format("    indented code", "code") is True
        assert check_instruction_format("Just text", "code") is False

    def test_instruction_details(self):
        """Verify instruction details returns correct structure."""
        from slm_train.evaluation.instruction import instruction_details

        responses = [
            {
                "instruction": "List three colors",
                "response": "Red, Green, Blue",
                "required_elements": ["Red", "Green", "Blue"],
            }
        ]
        result = instruction_details(responses)
        assert result["count"] == 1
        assert len(result["responses"]) == 1
        assert result["responses"][0]["elements_found"] == ["Red", "Green", "Blue"]
