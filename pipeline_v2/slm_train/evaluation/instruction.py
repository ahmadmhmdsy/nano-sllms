"""Instruction-style evaluation for SLM training.

This module scores how well a model follows instructions by:
1. Generating responses to instruction prompts
2. Checking if the responses contain required elements
3. Scoring based on instruction-following criteria

The evaluator is deliberately simple: it checks for the presence of
required elements in the response, not the quality of the response.
"""
from __future__ import annotations

import re
from typing import Any


def instruction_metrics(responses: list[dict[str, Any]]) -> dict[str, float]:
    """Score instruction-following based on required elements.

    Each response is a dict with:
    - instruction: the instruction prompt
    - response: the generated response
    - required_elements: list of strings that must appear in the response

    Returns a dict with:
    - instruction_following: fraction of responses that contain all required elements
    - mean_elements_found: average fraction of required elements found per response
    """
    if not responses:
        raise ValueError("instruction eval needs at least one response")

    total = len(responses)
    full_matches = 0
    element_fractions = []

    for resp in responses:
        required = resp.get("required_elements", [])
        response_text = resp.get("response", "")

        if not required:
            # No required elements means the instruction is trivially satisfied
            full_matches += 1
            element_fractions.append(1.0)
            continue

        found = sum(1 for elem in required if elem.lower() in response_text.lower())
        fraction = found / len(required)
        element_fractions.append(fraction)
        if fraction == 1.0:
            full_matches += 1

    return {
        "instruction_following": full_matches / total,
        "mean_elements_found": sum(element_fractions) / total,
    }


def check_instruction_format(response: str, format_type: str) -> bool:
    """Check if a response follows a specific format.

    Supported formats:
    - json: response is valid JSON
    - list: response contains a numbered or bulleted list
    - code: response contains a code block
    """
    if format_type == "json":
        try:
            import json
            json.loads(response)
            return True
        except Exception:
            return False
    elif format_type == "list":
        # Check for numbered list (1., 2., etc.) or bulleted list (-, *, etc.)
        return bool(re.search(r"^\s*([-*]|\d+\.)\s+", response, re.MULTILINE))
    elif format_type == "code":
        # Check for code block markers
        return "```" in response or "    " in response
    return False


def instruction_details(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Detailed instruction evaluation results."""
    return {
        "count": len(responses),
        "responses": [
            {
                "instruction": r.get("instruction", ""),
                "response": r.get("response", ""),
                "required_elements": r.get("required_elements", []),
                "elements_found": [
                    elem for elem in r.get("required_elements", [])
                    if elem.lower() in r.get("response", "").lower()
                ],
            }
            for r in responses
        ],
    }
