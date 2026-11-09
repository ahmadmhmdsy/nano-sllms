"""Tests for generation sampling parameters."""
from __future__ import annotations

import pytest


class TestGenerationSampling:
    """Tests for Task 7.1: generation sampling parameters."""

    def test_decode_samples_accepts_sampling_params(self):
        """Verify decode_samples accepts temperature, top_k, top_p."""
        from slm_train.models.decode import decode_samples
        import inspect

        sig = inspect.signature(decode_samples)
        assert "temperature" in sig.parameters
        assert "top_k" in sig.parameters
        assert "top_p" in sig.parameters

    def test_decode_samples_validates_temperature(self):
        """Verify temperature must be non-negative."""
        from slm_train.models.decode import decode_samples, DecodeError

        with pytest.raises(DecodeError, match="temperature must be non-negative"):
            decode_samples(
                model=None, encoder=None, prompt="test",
                temperature=-1.0,
            )

    def test_decode_samples_validates_top_k(self):
        """Verify top_k must be positive."""
        from slm_train.models.decode import decode_samples, DecodeError

        with pytest.raises(DecodeError, match="top_k must be positive"):
            decode_samples(
                model=None, encoder=None, prompt="test",
                top_k=0,
            )

    def test_decode_samples_validates_top_p(self):
        """Verify top_p must be in (0, 1]."""
        from slm_train.models.decode import decode_samples, DecodeError

        with pytest.raises(DecodeError, match="top_p must be in"):
            decode_samples(
                model=None, encoder=None, prompt="test",
                top_p=0.0,
            )
        with pytest.raises(DecodeError, match="top_p must be in"):
            decode_samples(
                model=None, encoder=None, prompt="test",
                top_p=1.5,
            )

    def test_config_has_generation_sampling_fields(self):
        """Verify the config model has generation sampling fields."""
        from slm_train.config.models import EvaluatorSuiteConfig

        assert hasattr(EvaluatorSuiteConfig, "model_fields")
        assert "generation_temperature" in EvaluatorSuiteConfig.model_fields
        assert "generation_top_k" in EvaluatorSuiteConfig.model_fields
        assert "generation_top_p" in EvaluatorSuiteConfig.model_fields
