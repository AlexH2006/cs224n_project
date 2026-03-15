"""
TLDR: Pure helpers for single-GPU batched generation.

Builds a flat list of prompts (all problems × pass_k) and metadata to map
vLLM outputs back to per-problem results. No vLLM or Modal dependency — used
by ProofGenerator.generate_all() and unit-tested in isolation.

Used by: modal_app.ProofGenerator.generate_all().
"""

from __future__ import annotations

from typing import Callable


def build_flat_prompts_and_meta(
    problems: list[dict],
    pass_k: int,
    prompt_builder: Callable[[dict], str],
) -> tuple[list[str], list[tuple[int, str]]]:
    """
    Build a flat prompt list and per-prompt metadata for batched vLLM generate().

    For each problem we build one prompt (via prompt_builder) and replicate it
    pass_k times. Order: problem0_attempt0..attemptK-1, problem1_attempt0.., ...

    Args:
        problems: List of problem dicts (formal_statement, informal_stmt, header, problem_idx).
        pass_k: Number of attempts per problem.
        prompt_builder: Callable that takes a problem dict and returns the prompt string.

    Returns:
        flat_prompts: List of prompt strings, length = len(problems) * pass_k.
        prompt_meta: List of (problem_idx, prompt_str) in same order as flat_prompts,
                     so we can map each vLLM output back to (problem_idx, attempt).
    """
    flat_prompts: list[str] = []
    prompt_meta: list[tuple[int, str]] = []

    for p in problems:
        prompt = prompt_builder(p)
        problem_idx = p["problem_idx"]
        for _ in range(pass_k):
            flat_prompts.append(prompt)
            prompt_meta.append((problem_idx, prompt))

    return flat_prompts, prompt_meta


def unflatten_results(
    prompt_meta: list[tuple[int, str]],
    flat_outputs: list[tuple[str, str]],
    problems: list[dict],
) -> list[list]:
    """
    Map flat vLLM outputs back to one result per problem in input order.

    vLLM returns one (text, finish_reason) per input prompt in the same order.
    We group by problem_idx and return a list aligned with the problems list.

    Args:
        prompt_meta: From build_flat_prompts_and_meta; (problem_idx, prompt_str) per prompt.
        flat_outputs: List of (raw_text, finish_reason) from vLLM, same length as prompt_meta.
        problems: Original problem list; order of returned list matches this.

    Returns:
        List of length len(problems). Each element is
        [prompt_str, (raw_0, reason_0), (raw_1, reason_1), ..., (raw_{pass_k-1}, reason_{pass_k-1})].
    """
    if len(prompt_meta) != len(flat_outputs):
        raise ValueError(
            f"prompt_meta length ({len(prompt_meta)}) must match flat_outputs length ({len(flat_outputs)})"
        )

    # Group outputs by problem_idx: result_by_idx[idx] = [prompt_str, (raw_0, r0), ...]
    result_by_idx: dict[int, list] = {}
    for (problem_idx, prompt_str), (raw_text, finish_reason) in zip(prompt_meta, flat_outputs):
        if problem_idx not in result_by_idx:
            result_by_idx[problem_idx] = [prompt_str]
        result_by_idx[problem_idx].append((raw_text, finish_reason))

    # Preserve order of input problems.
    return [result_by_idx[p["problem_idx"]] for p in problems]
