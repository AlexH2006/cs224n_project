"""
Batched vLLM generation helpers for imo_p6_experiment.

TLDR: Pure helpers for building a flat list of prompts and mapping vLLM outputs
back to per-problem results. Same pattern as qwen_eval/batch_generation.py:
build_flat_prompts_and_meta + unflatten_results. No vLLM or Modal dependency.

Used by: modal_trainer (generate_batch) and entrypoint when batching prompts
across problems or attempts.
"""

from __future__ import annotations

from typing import Callable


def build_flat_prompts_and_meta(
    problems: list[dict],
    pass_k_or_attempts: int,
    prompt_builder: Callable[[dict], str],
) -> tuple[list[str], list[tuple[int, int, str]]]:
    """
    Build a flat prompt list and per-prompt metadata for batched vLLM generate().

    For each problem we build one prompt (via prompt_builder) and replicate it
    pass_k_or_attempts times. Order: problem0_attempt0..attemptK-1, problem1_attempt0.., ...

    Args:
        problems: List of problem dicts (must include "problem_idx" or index is used).
        pass_k_or_attempts: Number of attempts per problem.
        prompt_builder: Callable that takes a problem dict and returns the prompt string.

    Returns:
        flat_prompts: List of prompt strings, length = len(problems) * pass_k_or_attempts.
        prompt_meta: List of (problem_idx, attempt_idx, prompt_str) in same order as flat_prompts,
                    so we can map each vLLM output back to (problem_idx, attempt).
    """
    flat_prompts: list[str] = []
    prompt_meta: list[tuple[int, int, str]] = []

    for i, p in enumerate(problems):
        problem_idx = p.get("problem_idx", i)
        prompt = prompt_builder(p)
        for attempt in range(pass_k_or_attempts):
            flat_prompts.append(prompt)
            prompt_meta.append((problem_idx, attempt, prompt))

    return flat_prompts, prompt_meta


def unflatten_results(
    prompt_meta: list[tuple[int, int, str]],
    flat_outputs: list[tuple[str, list[int]]],
    problems: list[dict],
) -> list[list[tuple[str, list[int]]]]:
    """
    Map flat vLLM outputs back to one result list per problem in input order.

    vLLM returns one (raw_text, generated_ids) per input prompt in the same order.
    We group by problem_idx and return a list aligned with the problems list.

    Args:
        prompt_meta: From build_flat_prompts_and_meta; (problem_idx, attempt_idx, prompt_str) per prompt.
        flat_outputs: List of (raw_text, generated_ids) from vLLM, same length as prompt_meta.
        problems: Original problem list; order of returned list matches this.

    Returns:
        List of length len(problems). Each element is a list of (raw_text, generated_ids)
        for that problem's attempts, in attempt order.
    """
    if len(prompt_meta) != len(flat_outputs):
        raise ValueError(
            f"prompt_meta length ({len(prompt_meta)}) must match flat_outputs length ({len(flat_outputs)})"
        )

    # Group outputs by problem_idx: result_by_idx[idx] = [(raw_0, ids_0), (raw_1, ids_1), ...]
    result_by_idx: dict[int, list[tuple[str, list[int]]]] = {}
    for (problem_idx, _attempt, _prompt_str), output in zip(prompt_meta, flat_outputs):
        if problem_idx not in result_by_idx:
            result_by_idx[problem_idx] = []
        result_by_idx[problem_idx].append(output)

    # Preserve order of input problems; use problem_idx from each problem dict.
    return [result_by_idx.get(p.get("problem_idx", i), []) for i, p in enumerate(problems)]
