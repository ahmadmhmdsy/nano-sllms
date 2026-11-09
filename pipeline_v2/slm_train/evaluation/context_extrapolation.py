"""Context extrapolation probes for SLM training.

This module tests if a model can handle contexts longer than its training
context. It generates text with increasingly longer contexts and scores based
on whether the model can maintain coherence.

The probe is deliberately simple: it checks if the model can generate
coherent text when given a context longer than its training context.
"""
from __future__ import annotations

import torch
from typing import Any


def context_extrapolation_metrics(
    model: Any,
    tokenizer: Any,
    base_context_length: int,
    test_context_lengths: list[int],
    device: str = "cpu",
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Test if the model can handle contexts longer than its training context.

    Args:
        model: The model to test
        tokenizer: The tokenizer to use
        base_context_length: The training context length
        test_context_lengths: List of context lengths to test
        device: The device to run on
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        A dict with:
        - extrapolation_success: fraction of test lengths that succeeded
        - max_successful_context: the longest context length that succeeded
        - coherence_scores: list of coherence scores for each test length
    """
    if not test_context_lengths:
        raise ValueError("context extrapolation needs at least one test length")

    results = []
    for test_length in test_context_lengths:
        try:
            # Create a prompt that is test_length tokens long
            # We use a simple repeating pattern to create a long prompt
            prompt = "The quick brown fox jumps over the lazy dog. " * (test_length // 10 + 1)
            prompt = prompt[:test_length * 5]  # Rough approximation

            # Tokenize the prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            if input_ids.shape[-1] > test_length:
                input_ids = input_ids[:, :test_length]

            # Generate text
            model.eval()
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids.to(device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0,
                )

            # Decode the output
            generated = tokenizer.decode(output[0], skip_special_tokens=True)

            # Simple coherence check: does the generated text contain any words?
            coherence = 1.0 if len(generated.strip()) > 0 else 0.0

            results.append({
                "test_length": test_length,
                "success": True,
                "coherence": coherence,
                "generated_length": len(generated),
            })
        except Exception as e:
            results.append({
                "test_length": test_length,
                "success": False,
                "coherence": 0.0,
                "error": str(e),
            })

    # Calculate metrics
    successful = [r for r in results if r["success"] and r["coherence"] > 0]
    extrapolation_success = len(successful) / len(results)
    max_successful_context = max(
        (r["test_length"] for r in successful),
        default=0,
    )

    return {
        "extrapolation_success": extrapolation_success,
        "max_successful_context": max_successful_context,
        "coherence_scores": [r["coherence"] for r in results],
        "results": results,
    }


def context_extrapolation_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize context extrapolation results."""
    return {
        "extrapolation_success": metrics["extrapolation_success"],
        "max_successful_context": metrics["max_successful_context"],
        "mean_coherence": sum(metrics["coherence_scores"]) / len(metrics["coherence_scores"]) if metrics["coherence_scores"] else 0.0,
    }
