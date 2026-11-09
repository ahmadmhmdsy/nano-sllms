"""Deterministic decoding from a trained checkpoint.

This is the only place in V2 that samples tokens from a model. It is kept
apart from evaluation/generation.py on purpose: that module scores text it is
given and must stay free of a model, while this one is the thing that actually
produces that text. Keeping the boundary sharp is what lets the generation
evaluator remain a pure function.

The encoder is the SAME pinned local tokenizer file the run tokenized with,
loaded through the tokenizers library exactly as the tokenize stage loads it.
That is deliberate: decoding through any other tokenizer would measure the
run against a vocabulary it was never trained on.

Decoding is GREEDY and seeded. A smoke test whose output changes between runs
cannot be compared against a recorded number, and a generation gate that
flapped would be worse than no gate at all.
"""
from __future__ import annotations

__all__ = ["DecodeError", "decode_samples"]


class DecodeError(RuntimeError):
    """Decoding could not be completed. Never returns a fabricated sample."""


def decode_samples(
    model,
    encoder,
    prompt: str,
    sample_count: int = 2,
    max_new_tokens: int = 24,
    device: str = "cpu",
    seed: int = 0,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
) -> tuple:
    """Greedy or sampled continuations of one fixed prompt, reproducible run to run.

    When temperature > 0, sampling is enabled with the given temperature, top_k,
    and top_p. When temperature == 0, greedy decoding is used.
    """
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise DecodeError("the smoke prompt must be non-empty text")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise DecodeError("sample count must be an integer")
    if sample_count < 1:
        raise DecodeError("decoding needs at least one sample")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise DecodeError("max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise DecodeError("max_new_tokens must be positive")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise DecodeError("pad id must be an integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise DecodeError("temperature must be a number")
    if temperature < 0:
        raise DecodeError("temperature must be non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise DecodeError("top_k must be an integer")
    if top_k < 1:
        raise DecodeError("top_k must be positive")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise DecodeError("top_p must be a number")
    if not 0 < top_p <= 1:
        raise DecodeError("top_p must be in (0, 1]")
    if not hasattr(encoder, "encode") or not hasattr(encoder, "decode"):
        raise DecodeError("the encoder is not a tokenizers.Tokenizer")

    input_ids = torch.tensor(
        [list(map(int, encoder.encode(prompt).ids))], dtype=torch.long
    )
    if input_ids.shape[-1] < 1:
        raise DecodeError("the prompt encoded to zero tokens")

    was_training = model.training
    model.eval()
    samples = []
    try:
        with torch.no_grad():
            for index in range(sample_count):
                # Re-seeding per sample makes each one independent of how many
                # preceded it, so changing sample_count cannot shift the others.
                torch.manual_seed(seed + index)
                use_sampling = temperature > 0
                generated = model.generate(
                    input_ids=input_ids.to(device),
                    max_new_tokens=max_new_tokens,
                    do_sample=use_sampling,
                    num_beams=1,
                    pad_token_id=pad_id,
                    temperature=temperature if use_sampling else None,
                    top_k=top_k if use_sampling else None,
                    top_p=top_p if use_sampling else None,
                )
                completion = [
                    int(token) for token in generated[0][input_ids.shape[-1]:]
                ]
                samples.append(
                    encoder.decode(completion, skip_special_tokens=True)
                )
    except DecodeError:
        raise
    except Exception as error:
        raise DecodeError("decoding failed: " + type(error).__name__) from error
    finally:
        model.train(was_training)
    return tuple(samples)
