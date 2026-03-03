# Bugfixes & Observations — Verification (Kimina 2B / Goedel / DeepSeek)

**Date:** 2026-03-02  
**Topics:** bugfixes, verification, kimina, sdpo

---

Session focused on investigating and fixing a cluster of false-positive verification bugs in `lean_sdpo_kimina_2b_modal.py` (and the parallel Goedel 8B / DeepSeek 7B files). All bugs were discovered by inspecting SDPO run logs in `sdpo_results/kimina_2b/`.

---

## Bug 1: Commented-out formal statement accepted as a valid proof

### Symptom
`run_19` (`amc12a_2020_p10`) reported `success: true`, `complete: true` after iteration 1 despite the model producing nothing but repeated commented-out theorem declarations.

### Root cause (3 cooperating issues)

**1a. `_extract_tactics_from_code_block` passed comment lines through.**
Lines starting with `--` matched none of the existing skip rules (they don't start with `theorem`, `(`, etc.), so they were included in the extracted tactics as if they were real tactic code.

**1b. `_create_full_lean_code` replaced `sorry` inside a Lean comment.**
The `formal_statement` for this problem in the HF dataset (`cat-searcher/minif2f-lean4`) is entirely commented out:
```
-- Error: Real.log
-- theorem amc12a_2020_p10
--   ...
--   (Nat.digits 10 n).sum = 13 := sorry
```
The code used `rfind("sorry")` which found the `sorry` inside the comment and replaced it with the extracted "tactics" (more comment lines). This removed the literal word `sorry` from the file entirely.

**1c. The verifier was fooled.**
With no real Lean code left (just imports + comments), Lean compiled cleanly with zero errors and zero sorries. The verifier's check `"sorry" in lean_code.lower()` returned False (the word `sorry` had been replaced). Result: `complete: true`.

### Fix
- Added `if stripped.startswith("--"): continue` in `_extract_tactics_from_code_block` to discard all pure comment lines.
- Added `_theorem_code_is_commented_out()` helper that checks whether every non-empty, non-import line in `theorem_code` starts with `--`.
- Added an early-exit guard in the main training loop: if `_theorem_code_is_commented_out(lean4_code)` is true, skip immediately with `success: false` and a clear error message. No Lean verification is attempted.

Applied to all three training files: `lean_sdpo_kimina_2b_modal.py`, `lean_sdpo_goedel_8b_modal.py`, `lean_sdpo_deepseek_7b_modal.py`.

---

## Observation: 17/244 HF dataset problems are broken

### Finding
The HF dataset `cat-searcher/minif2f-lean4` was built with an older Mathlib snapshot where certain APIs weren't available. For 17 out of 244 problems (7%), the `formal_statement` field is entirely commented out with an error marker:

| Error type | Count | Affected APIs |
|---|---|---|
| `Real.log` | 5 | `Real.logb` / log functions |
| `Real^Real` | 5 | Real-number exponentiation `a ^ b` for `a b : ℝ` |
| `Real.pi` | 5 | `Real.pi` and trig functions |
| `irrational` | 1 | `irrational` predicate |
| `ℕ+^Real` | 1 | `PNat` exponentiation |

Affected indices (HF ordering): 19, 64, 90, 96, 108, 111, 117, 135, 144, 152, 154, 156, 203, 208, 209, 220, 231.

**All 17 are valid in the local `dataset/minif2f.jsonl`**, which uses a newer Mathlib and properly formalizes them.

### Dataset comparison
The local file and HF dataset contain the same 244 problems but **in a different order** (188/244 positions differ). The local file uses field names `name`/`lean4_code` while the HF dataset uses `id`/`formal_statement`+`header`. Switching to the local dataset requires ID-based lookup, not index-based.

---

## Bug 2: `:= sorry` assembled into `:= tactic` without `by`

### Symptom
`run_176` (`imo_2019_p1`) reported `success: true` after iteration 1 with `full_code` containing:
```lean4
theorem imo_2019_p1 ... := constructor
  · -- Assume that $f$ satisfies...
  intro h
```
This is syntactically broken Lean 4 (`constructor` is a tactic, not a term, so `:= constructor` is invalid), yet the Kimina verifier accepted it.

### Root cause
The HF dataset's `formal_statement` ends with `:= sorry` (term-mode placeholder, no `by`). The assembly code checks for `:= by sorry` and `:= by\n  sorry` first (both miss), then falls to the `else` branch which does:
```python
last_sorry = theorem_code.rfind("sorry")
theorem_with_proof = theorem_code[:last_sorry] + indented_proof + theorem_code[last_sorry + 5:]
```
This replaces only the word `sorry`, leaving `:= ` as a prefix. The result is `:= constructor\n  ...` instead of `:= by\n  constructor\n  ...`.

### Fix
In the `else` branch, check if the text before `sorry` ends with `:=`. If so, insert ` by\n  ` before the proof tactics:
```python
if before.rstrip().endswith(":="):
    theorem_with_proof = before.rstrip() + f" by\n  {indented_proof}" + after
else:
    theorem_with_proof = before + indented_proof + after
```
Applied to all three training files.

---

## Bug 3: Inline comment-in-bullet lines surviving extraction

### Symptom
The model generated a code block containing `· -- Assume that $f$ satisfies the functional equation...`. The extracted tactics included this comment-bearing line, which became part of the assembled `full_code`.

### Root cause
The earlier fix `if stripped.startswith("--"): continue` only caught pure comment lines. A line like `· -- comment` has `stripped = "· -- comment..."` which starts with `·`, not `--`. The comment prose rode through into the tactics.

### Fix
Added inline comment stripping in `_extract_tactics_from_code_block`:
```python
if " --" in stripped:
    stripped = stripped[:stripped.index(" --")].rstrip()
    if not stripped:
        continue
```
`· -- Assume that...` becomes just `·` (a valid empty bullet focus). Applied to all three training files.

---

## Observation: Kimina verifier has persistent false-positive problem

### Finding
Even after the `:= by` fix, `run_176` (second run) still returned `success: true, complete: true` for:
```lean4
theorem imo_2019_p1 ... := by
  constructor
  ·
  intro h
```
This is clearly incomplete — empty first bullet, second goal of `constructor` never touched. Zero errors, zero messages, zero sorries from the server.

### Root cause hypothesis
The verifier logic at `LeanVerifier.verify` only treats messages with `severity == "error"` as failures:
```python
for msg in messages:
    if isinstance(msg, dict) and msg.get("severity") == "error":
        errors.append(...)
```
Lean 4 reports "unsolved goals" as `severity: "warning"` or `severity: "information"` in some configurations (particularly when goals remain at the end of a tactic block rather than failing mid-block). These are silently ignored, so a proof with open goals passes as `complete: true`.

### Status
**Not yet fixed.** The proper fix is to also treat any message containing `"unsolved goals"` as a failure regardless of severity.

---

## Bug 4: `LeanVerifier` Modal timeout too low for Kimina 2B

### Symptom
Run crashed with:
```
FunctionTimeoutError: Task's current input ... hit its timeout of 300s
```
at `LeanVerifier().verify.remote(full_code)` during iteration 4 of `run_176`.

### Root cause
The timeout fixes from the 2026-03-01 session were applied to `lean_sdpo_goedel_8b_modal.py` but **never backported to `lean_sdpo_kimina_2b_modal.py`**. With `httpx.Client` timeout at 180s per attempt and 10 retries, the first slow attempt plus retry overhead easily exceeded the 300s `LeanVerifier` Modal task timeout.

### Fix
Brought all three timeout layers in the Kimina 2B file in line with the Goedel 8B file:

| Setting | Before | After |
|---|---|---|
| `KiminaLeanServer` Modal timeout | 600s | 1200s |
| `LeanVerifier` Modal timeout | **300s** ← crashed here | 900s |
| `httpx.Client` per-request timeout | 180s | 600s |

Note: `SDPOTrainer` itself has a 3600s (1 hour) timeout which governs the entire training run. For longer training (more iterations or harder problems), increase this value — no server restarts are needed since each Modal container restarts automatically between calls.

---

## Observation: Tactic extraction from truncated (unfinished) reasoning traces

### How Strategy 2 works
When the model's output contains `<think>` but never closes with `</think>` (i.e. the context window was exhausted mid-reasoning), Strategy 1 of `_extract_proof_tactics` fails. Strategy 2 then fires:

```python
# Strategy 2: Extract from code blocks inside <think>
if not tactics and "<think>" in output:
    think_match = re.search(r"<think>(.*?)(?:</think>|$)", output, re.DOTALL)
    ...
    matches = re.findall(code_pattern, think_content, re.DOTALL)
    # Collect ALL tactic-like content from ALL code blocks
    all_tactics = []
    for match in matches:
        extracted = _extract_tactics_from_code_block(match)
        if extracted and extracted.lower() not in ["sorry", "by"]:
            all_tactics.append(extracted)
    if all_tactics:
        tactics = "\n".join(all_tactics)
```

The regex `<think>(.*?)(?:</think>|$)` with `re.DOTALL` matches from `<think>` to end-of-string when `</think>` is absent. It then finds all ` ```tactics ``` ` / ` ```lean4 ``` ` code blocks the model wrote **during** its reasoning — even if the model later abandoned them.

### Example observed (`run_30`, iteration 2, `imo_1997_p5`)
- Raw output: 18,043 chars, has `<think>`, no `</think>`
- Model wrote a partial proof early in reasoning:
  ```lean4
  rcases h₀ with ⟨hx, hy⟩
  by_contra h
  push_neg at h
  ```
- Then got stuck in a "Wait... let's think..." loop for the remaining ~15,000 chars
- Strategy 2 found this early code block and returned those tactics

### Implication
These extracted tactics are **mid-reasoning scratchpad attempts** — the model may have already rejected them by the time it got stuck looping. They are structurally incomplete (no closing tactics, no proof of the full goal). Verifying them wastes a round-trip to the Lean server and risks false positives from the verifier.

A more conservative approach would be to only use Strategy 2 as a last resort and to require the extracted block to contain a closing tactic pattern (e.g. `omega`, `linarith`, `exact`, `decide`, `simp`, `ring`) suggesting the proof attempt was at least partially complete.

---

## Summary of code changes

| Change | Files | Purpose |
|---|---|---|
| Filter pure comment lines in tactic extraction | All 3 | Prevent comment echoes from being treated as tactics |
| Strip inline trailing comments (`· -- ...` → `·`) | All 3 | Clean tactic lines of prose commentary |
| `_theorem_code_is_commented_out()` helper + guard | All 3 | Detect and skip HF dataset's 17 broken problems |
| Fix `:= sorry` → `:= by\n  tactics` assembly | All 3 | Produce valid Lean 4 tactic-mode proofs from HF-style formal statements |
| Backport timeout fixes to Kimina 2B | `lean_sdpo_kimina_2b_modal.py` | Match Goedel 8B timeouts; prevent `FunctionTimeoutError` crashes |

---

## Follow-up fix: Unsolved goals treated as verification failure

### Problem (from Observation above)
The verifier only treated messages with `severity == "error"` as failures. Lean 4 can report "unsolved goals" as `severity: "warning"` or `"information"`, so incomplete proofs (e.g. `constructor · intro h` with open goals) were reported as `success: true`, `complete: true`.

### Fix applied
In `LeanVerifier.verify` (all four training files: `lean_sdpo_kimina_2b_modal.py`, `lean_sdpo_kimina_distill_1_7b_modal.py`, `lean_sdpo_goedel_8b_modal.py`, `lean_sdpo_deepseek_7b_modal.py`): when iterating over Kimina `messages`, treat any message whose text contains `"unsolved goals"` (case-insensitive) as an error regardless of severity. Add that message to `errors` and set `has_error = True`, so the run is no longer marked complete.

---

## Follow-up fix: No tactic extraction from truncated reasoning (strict formatting)

### Problem (from Observation above)
Strategy 2 extracted tactics from code blocks inside `<think>` when the model never closed with `</think>` (truncated output). Those tactics are mid-reasoning scratchpad attempts and can be incomplete, wasting verification and causing false positives.

### Fix applied
- **Main loop:** If output is truncated (`<think>` present, `</think>` absent), set `tactics = "sorry"` and do not call `_extract_proof_tactics`.
- **`_extract_proof_tactics`:** At the start, if `<think>` is in the output and `</think>` is not, return `"sorry"` immediately (so truncated traces never yield extracted tactics).
- **Strategy 2 removed:** The branch that extracted from code blocks inside `<think>` when there was no `</think>` was removed. Extraction now uses only: (1) content after `</think>` (complete thinking), (2) code blocks anywhere in the output when reasoning is complete, (3) the existing ":= by" pattern (Strategy 4).

Applied to all four training files: `lean_sdpo_kimina_2b_modal.py`, `lean_sdpo_kimina_distill_1_7b_modal.py`, `lean_sdpo_goedel_8b_modal.py`, and `lean_sdpo_deepseek_7b_modal.py`.

---

## Follow-up: Goedel and DeepSeek verification parity

The same verification fixes (unsolved goals, truncated → sorry, Kimina server health/restart) were originally applied only to the Kimina training files. They have been backported to **Goedel** and **DeepSeek** so all four pipelines behave consistently.

### Summary of changes applied to `lean_sdpo_goedel_8b_modal.py` and `lean_sdpo_deepseek_7b_modal.py`

| Area | Change |
|------|--------|
| **KiminaLeanServer** | Extracted `_start_lean_server()`; added `_ensure_server_alive()` called at top of every `verify()`. On connect/timeout/other failure, retry loop restarts the server instead of sleeping. HTTP timeout 60 s, 3 retries (was 600 s / 10 retries). |
| **LeanVerifier** | Message parsing: treat any message whose text contains `"unsolved goals"` (case-insensitive) as an error regardless of severity. |
| **Main loop** | If output is truncated (`<think>` without `</think>`; DeepSeek uses `_has_think_start` / `_has_think_end`), set `tactics = "sorry"` and do not call `_extract_proof_tactics`. |
| **_extract_proof_tactics** | Early return `"sorry"` when reasoning is incomplete (<think> present, </think> absent). Removed Strategy B (extract from code blocks inside <think> when truncated). |

Devlog references: verification slowness / iteration 3–4 fix is documented in `20260302_verification_slowness_iter3_4.md`; unsolved-goals and truncated-tactic fixes are in this file above.
