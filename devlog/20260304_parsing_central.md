# Parsing: Centralized Notes and Considerations

**Date:** 2026-03-04  
**TLDR:** This doc consolidates (1) the cause and logic of the current SDPO tactic-extraction parsing bug, (2) the design choice of extraction vs full code block and tradeoffs, and (3) Goedel-Prover’s parsing approach and how it contrasts with ours. Use this for any change to “model output → Lean code sent to verifier.”

---

## 1. Current SDPO parsing error (cause)

**Context:** Run 141 (mathd_numbertheory_458). The model outputs a correct full proof including `have h1 : n % 4 = 3 := by` / `omega` / `exact h1`, but the pipeline sends only `omega` and `exact h1` to the verifier, so verification fails with “no goals to be solved.”

**Cause:** In `sdpo_modal_local_verify/utils.py`, **`extract_tactics_from_code_block`** (lines 104–105) skips any line whose stripped form ends with `":="` or `":= by"`:

```python
if stripped.endswith(":=") or stripped.endswith(":= by"):
    continue
```

- **Intent:** Skip the theorem/lemma declaration line (e.g. `theorem ... := by`) so it is not treated as a tactic.
- **Effect:** Valid proof lines that introduce a hypothesis also end with `:= by`, e.g. `have h1 : n % 4 = 3 := by`, so they are dropped.
- **Result:** Only `omega` and `exact h1` are kept; `h1` is never introduced → invalid proof.

**Block selection is correct:** The chosen ```lean4 block (after `</think>`) contains the full proof. The bug is only in the per-line filtering inside that block.

---

## 2. Current SDPO parsing logic (overview)

Parsing is in two layers: **where** to get the tactics from in the model output, and **how** to clean a single block into “tactics.”

### 2.1. Where tactics come from: `extract_proof_tactics(output)`

- **Input:** Full `raw_output` (may include `<think>...</think>` and code blocks).
- **Steps:**
  1. If `<think>` present and `</think>` absent → return `"sorry"`.
  2. Prefer content **after the last** `</think>`.
  3. Find code blocks with regex `` r"```(?:lean4?|lean|tactics)?\n?(.*?)```" `` (re.DOTALL, non-greedy).
  4. For each match, run `extract_tactics_from_code_block(match)`; take the first result that is not `"sorry"` or `"by"`.
  5. Fallback: if no block, take up to 10 lines after the last `":= by"` in the output.
  6. Post-process: strip leading `"by"`, reject if too short or contains `":= sorry"`.
- **Output:** Extracted tactic string or `"sorry"`.

### 2.2. How a block is cleaned: `extract_tactics_from_code_block(block)`

- **Input:** Raw content of one code block (between `` ```lean4 `` and `` ``` ``).
- **Per line:** Skip blank/comment; strip inline ` -- ...`; keep `import`/`open`/`set_option`; skip `theorem`/`lemma`, parenthesized binders, `":= sorry"`, several numeric patterns, and **any line ending with `":="` or `":= by"`** (this removes `have ... := by`).
- **Output:** `"\n".join(lines)`.

After that, **`create_full_lean_code`** merges dataset theorem + dataset header + these tactics into one Lean file and that string is sent to the verifier.

---

## 3. Do we need to extract tactics? (design choice)

**Short answer:** No. The verifier only needs a complete Lean file. We can send the **entire** ```lean4 block (optionally with dataset header prepended) and avoid tactic extraction entirely.

**Current design (extract tactics + merge):**

- **Dataset** = source of truth for theorem statement and header.
- **Model** = only the “proof part”; we extract “tactics,” then merge with dataset theorem and header.
- **Rationale:** Canonical statement and imports; prompt asks for “proof tactics in a ```lean4 code block,” so the pipeline was built to treat the block as tactics and paste them into the dataset.
- **Cost:** Brittle, line-based filtering (e.g. `have ... := by` dropped) and more code paths.

**Alternative (full code block → verifier):**

- Take the **entire** content of the chosen ```lean4 block (imports, theorem, proof).
- Optionally prepend dataset header when the block has no/minimal imports.
- Send that string as `full_code` to Kimina. No tactic extraction, no `create_full_lean_code` merge.
- **Tradeoff:** Simpler and robust; we may verify a slightly different theorem if the model typos the statement. Mitigations: prepend dataset header when needed; optionally check theorem name in the block against the dataset.

---

## 4. Goedel-Prover parsing (full picture)

**Location:** `Goedel-Prover-main/eval/step1_inference.py`, `prover/lean/verifier.py`, `prover/lean/ast_parser.py`.

### 4.1. What gets sent to the verifier: full code block only

- **Prompt:** “Complete the following Lean 4 code … ```lean4\n{header}{informal_prefix}{formal_statement}” (no closing ```). So the model sees the full header + informal + formal statement and is asked to complete the code.
- **Extraction:**  
  `extract_code(inputs)` with `inputs = model_input + output`:
  ```python
  return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
  ```
  So they take the **first** ```lean4 … ``` span in prompt+output. That span is: the prompt’s opening ```lean4 and content (header + informal + formal statement) **plus** the model’s completion until the **first** closing ```. So the extracted string is the **full Lean file**: header + statement + proof in one block. No tactic extraction; no merge with dataset after extraction.
- **Verification:** `verify_lean4_file(code)` receives that full string. Step2 compiles each `code["code"]` as-is (`code` = full extracted block).

So for **input to the verifier**, Goedel uses **full code block only**: one regex, take everything inside the first ```lean4 ... ```, and send it to the REPL. No line-level tactic filtering, no reassembly from dataset.

### 4.2. Role of ast_parser: post-verification, not model-output parsing

- **When it runs:** After verification. The Lean REPL (`lake exe repl`) is called with the **full** code; it returns a result that includes `ast`, `tactics` (tactic positions), `messages`, etc. So “tactics” here come from the **Lean server**, not from parsing the model’s text.
- **What ast_parser does:** `lean4_parser(file_content, result['ast'])` takes the **verified file** and the REPL’s AST/tactics and produces structured info: declaration kinds, statement/proof spans, tactic list with positions (using byte/line positions to slice `file_content`). So it **analyzes** the verified code (for training or analysis), and does **tactic-level** structure only on the **REPL output**, not on the model output.
- **Conclusion:** In Goedel, “tactic-level” parsing is **server-side** (REPL returns tactic positions); client-side they only do **full code block** extraction from the model. No client-side tactic extraction for verification.

---

## 5. Contrast and recommendations

| Aspect | SDPO (current) | Goedel-Prover |
|--------|----------------|----------------|
| **What is extracted from model** | “Tactics” only (line-based filter of one ```lean4 block). | Full content of first ```lean4 … ``` (prompt content + model completion). |
| **What is sent to verifier** | Dataset header + dataset theorem + extracted tactics (reassembled). | Single extracted block (header + statement + proof) as-is. |
| **Tactic-level parsing** | Client-side: line filters in `extract_tactics_from_code_block` (buggy for `have ... := by`). | Server-side only: REPL returns tactics; ast_parser uses AST + file to get proof/tactic structure after verification. |
| **Risk** | Parsing bugs (e.g. dropping lines), many edge cases. | Wrong block if model outputs multiple ``` blocks (first match wins); no guarantee theorem text matches dataset. |

**Recommendations:**

1. **Short term:** If we keep tactic extraction, fix the bug by narrowing the “declaration” skip so it does not apply to `have ... := by` (e.g. only skip lines that start with `theorem ` / `lemma ` or that match a declaration pattern, not every line ending with `:= by`).
2. **Medium term:** Prefer **full code block → verifier** (like Goedel): extract one ```lean4 block as a single string, optionally prepend dataset header, send to Kimina. No tactic extraction, no `create_full_lean_code` merge. Optionally check theorem name in the block against the dataset.
3. **Reference:** Goedel’s approach (full block extraction, no client-side tactic parsing for verification; tactic-level only in post-verification analysis of REPL output) is a clean separation and avoids the class of bugs we hit.

---

## 6. Summary table

| Item | Description |
|------|-------------|
| **Observed bug** | `have h1 : ... := by` dropped by `extract_tactics_from_code_block` → only `omega` and `exact h1` kept → “no goals to be solved.” |
| **Cause** | Filter `stripped.endswith(":=")` / `stripped.endswith(":= by")` is too broad; it removes valid `have` lines. |
| **Block selection** | Correct: last ```lean4 block after `</think>` contains the full proof. |
| **Goedel verification input** | Full code block (first ```lean4 … ``` in prompt+output); no tactic extraction. |
| **Goedel “tactics”** | From REPL response; ast_parser uses them for post-verification structure, not for building the verified file. |
| **Fix / direction** | Option A: Restrict declaration skip so `have ... := by` is kept. Option B: Switch to full-block extraction and optional header prepend (align with Goedel). |
