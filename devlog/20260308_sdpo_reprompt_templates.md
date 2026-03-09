# SDPO Test-Time Self-Distillation Reprompt Templates

From `SDPO/verl/workers/config/actor.py` (`SelfDistillationConfig`).

## Templates (default values)

### `reprompt_template`

Main template for the reprompted teacher prompt. Placeholders: `{prompt}`, `{solution}`, `{feedback}`.

```
{prompt}{solution}{feedback}

Correctly solve the original question.
```

### `solution_template`

Section shown when a successful previous attempt exists in the same prompt group. Placeholder: `{successful_previous_attempt}`.

```
Correct solution:

{successful_previous_attempt}

```

### `feedback_template`

Section shown when environment feedback is available (e.g. Lean error message). Placeholder: `{feedback_raw}`.

```
The following is feedback from your unsuccessful earlier attempt:

{feedback_raw}

```

## How they combine

The teacher prompt is built as:

```
reprompt_template.format(
    prompt=original_user_prompt,
    solution=solution_template.format(successful_previous_attempt=...) if solution else "",
    feedback=feedback_template.format(feedback_raw=...) if feedback else ""
)
```

- If there is a solution: `solution` is filled via `solution_template`.
- If there is feedback: `feedback` is filled via `feedback_template`.
- If neither: `reprompt_text` = original `prompt` only (no extra sections).

## Related config options

| Option | Default | Description |
|--------|---------|-------------|
| `max_reprompt_len` | 10240 | Max token length of the reprompted prompt |
| `reprompt_truncation` | "right" | Truncation method ("right" or "error") |
| `dont_reprompt_on_self_success` | False | Exclude the current sample when selecting a successful demo |
| `remove_thinking_from_demonstration` | False | Strip `<think>...</think>` from solution text |
| `include_environment_feedback` | False | Whether to include feedback in reprompting |
| `environment_feedback_only_without_solution` | False | Use feedback only when no solution exists |

## Docstring-only (not in defaults)

- `reprompt_template_feedback`: template for feedback-only (no solution)
- `reprompt_template_feedback_solution`: template for feedback + solution

In the current `ray_trainer.py` logic, only the single `reprompt_template` with all three placeholders is used.
