"""Tests for YaRN context extension."""
from __future__ import annotations

import pytest


class TestYaRNContextExtension:
    """Tests for Task 7.4: YaRN context extension."""

    def test_yarn_scaling_invalid_target(self):
        """Verify target context length must be greater than base."""
        from slm_train.models.yarn import apply_yarn_scaling

        with pytest.raises(ValueError, match="target context length must be greater than base"):
            apply_yarn_scaling(
                model=None,
                base_context_length=512,
                target_context_length=256,
            )

    def test_yarn_scaling_result_structure(self):
        """Verify YaRN scaling returns correct structure."""
        from slm_train.models.yarn import apply_yarn_scaling

        class MockConfig:
            rope_theta = 10000.0

        class MockModel:
            config = MockConfig()

        model = MockModel()
        result = apply_yarn_scaling(
            model=model,
            base_context_length=512,
            target_context_length=1024,
        )
        assert result["applied"] is True
        assert result["base_context_length"] == 512
        assert result["target_context_length"] == 1024
        assert result["scale_factor"] == 2.0
        assert result["yarn_factor"] > 1.0

    def test_yarn_scaling_without_rope_theta(self):
        """Verify YaRN scaling handles models without rope_theta."""
        from slm_train.models.yarn import apply_yarn_scaling

        class MockConfig:
            pass

        class MockModel:
            config = MockConfig()

        model = MockModel()
        result = apply_yarn_scaling(
            model=model,
            base_context_length=512,
            target_context_length=1024,
        )
        assert result["applied"] is False
        assert result["base_context_length"] == 512
        assert result["target_context_length"] == 1024

    def test_yarn_context_extension_summary(self):
        """Verify summary returns correct metrics."""
        from slm_train.models.yarn import yarn_context_extension_summary

        result = {
            "applied": True,
            "base_context_length": 512,
            "target_context_length": 1024,
            "scale_factor": 2.0,
        }
        summary = yarn_context_extension_summary(result)
        assert summary["applied"] is True
        assert summary["base_context_length"] == 512
        assert summary["target_context_length"] == 1024
        assert summary["scale_factor"] == 2.0
