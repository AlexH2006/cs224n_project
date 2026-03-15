"""
TLDR: Self-contained copy of get_field for dataset-agnostic field extraction.

Used by qwen_sdpo.entrypoint._load_problem() to extract theorem, informal, header,
and id from HuggingFace dataset rows. No dependency on qwen_eval or EvalConfig;
caller passes field lists from SDPOConfig.

Used by: entrypoint._load_problem.
"""


def get_field(data: dict, field_names: list[str], default: str = "") -> str:
    """
    Return the first non-empty string value found in data using the priority list.

    Handles both plain string fields and list/tuple fields (joins with space).
    Falls back to `default` if no field matches.
    """
    for name in field_names:
        val = data.get(name)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val = " ".join(str(v) for v in val if v)
        val = str(val).strip()
        if val:
            return val
    return default
