"""Tests for context extrapolation probes."""
from __future__ import annotations

import pytest


class TestContextExtrapolation:
    """Tests for Task 7.3: context extrapolation probes."""

    def test_context_extrapolation_metrics_empty_lengths(self):
        """Verify empty test lengths raises an error."""
        from slm_train.evaluation.context_extrapolation import context_extrapolation_metrics

        with pytest.raises(ValueError, match="context extrapolation needs at least one test length"):
            context_extrapolation_metrics(
                model=None, tokenizer=None,
                base_context_length=512,
                test_context_lengths=[],
            )

    def test_context_extrapolation_summary(self):
        """Verify summary calculates correct metrics."""
        from slm_train.evaluation.context_extrapolation import context_extrapolation_summary

        metrics = {
            "extrapolation_success": 0.5,
            "max_successful_context": 1024,
            "coherence_scores": [1.0, 0.0],
        }
        result = context_extrapolation_summary(metrics)
        assert result["extrapolation_success"] == 0.5
        assert result["max_successful_context"] == 1024
        assert result["mean_coherence"] == 0.5

    def test_context_extrapolation_summary_empty_coherence(self):
        """Verify summary handles empty coherence scores."""
        from slm_train.evaluation.context_extrapolation import context_extrapolation_summary

        metrics = {
            "extrapolation_success": 0.0,
            "max_successful_context": 0,
            "coherence_scores": [],
        }
        result = context_extrapolation_summary(metrics)
        assert result["mean_coherence"] == 0.0
