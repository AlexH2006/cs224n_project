# mathlib4 Missing: FileNotFoundError in Local Lean Verifier

**TLDR:** The `FileNotFoundError: ... Goedel-Prover-main/mathlib4` in local-verify runs is caused by the mathlib4 workspace path not existing. mathlib4 is a **git submodule** of Goedel-Prover and is not cloned by default. Fix: initialize and build the submodule (or set `LEAN_WORKSPACE` to an existing mathlib4 path). No code changes are proposed here—setup and future hardening only.

---

## 1. What shows up in logs

When verification runs, the pipeline can store a long Python traceback in `logs.json` under `iteration_logs[].feedback`:

```text
Traceback (most recent call last):
  File ".../sdpo_modal_local_verify/local_lean_verifier.py", line 48, in _run_goedel_style_verify
    proc = subprocess.run(
  ...
FileNotFoundError: [Errno 2] No such file or directory: '.../Goedel-Prover-main/mathlib4'
```

So the **observable symptom** is: verification fails with a server-style error and the “feedback” is a full traceback instead of Lean compiler output.

---

## 2. Root cause

- The local verifier runs `lake exe repl` with **working directory** set to a “Lean workspace” that must be a **mathlib4** project (directory containing a Lakefile and built dependencies).
- Default workspace path is set to:
  - `{repo_root}/Goedel-Prover-main/mathlib4`
  - with `repo_root` derived from the verifier module’s location (i.e. the 224n_project root).
- In **Goedel-Prover**, `mathlib4` is a **git submodule** (see `Goedel-Prover-main/.gitmodules`). Cloning the 224n project (or Goedel-Prover) does **not** clone submodules unless you use `git clone --recurse-submodules` or later run `git submodule update --init --recursive`.
- If that directory does not exist, `subprocess.run(..., cwd=lean_workspace)` raises **FileNotFoundError** (Python 3: `cwd` must be an existing directory).
- The verifier’s broad `except Exception` in `verify()` catches this and turns it into a verification error result, putting `traceback.format_exc()` into `errors` and the string that becomes `feedback`. So the **huge** error is the full traceback being stored and printed as if it were Lean feedback.

**Summary:** The “huge error” is a **missing mathlib4 directory** (submodule not initialized) plus the fact that any exception is currently surfaced as a long traceback in the feedback field.

---

## 3. Proposed solutions (no code changes in this doc)

### 3.1 Immediate fix (setup / operations)

- **Ensure mathlib4 exists and is built** under Goedel-Prover:
  ```bash
  cd Goedel-Prover-main
  git submodule update --init --recursive
  cd mathlib4
  lake build
  cd ../..
  ```
- Run the local-verify pipeline (and sanity check) from the **repo root** so the default path `.../Goedel-Prover-main/mathlib4` resolves correctly.
- If mathlib4 lives elsewhere (e.g. a standalone clone), set the environment variable **`LEAN_WORKSPACE`** to the **absolute** path of that mathlib4 directory before running the pipeline. The verifier already uses `LEAN_WORKSPACE` when set.

This addresses the FileNotFoundError by making the default path valid, or by pointing to another valid workspace.

### 3.2 Clearer errors (future code hardening)

Proposed improvements for later implementation; not applied in this devlog:

1. **Pre-check workspace directory**  
   Before calling `subprocess.run(..., cwd=lean_workspace)`:
   - Resolve `lean_workspace` to an absolute path.
   - If `not os.path.isdir(lean_workspace)`:
     - Return a **verification_error_result** with a short, user-facing message, e.g.  
       `"mathlib4 workspace not found at <path>. Run 'git submodule update --init --recursive' in Goedel-Prover-main, then 'lake build' in mathlib4."`
     - Do **not** let subprocess raise; then the feedback stays short and actionable instead of a traceback.

2. **Handle FileNotFoundError explicitly**  
   In `verify()`, catch `FileNotFoundError` separately from the generic `Exception`:
   - If the path in the exception message looks like the workspace (e.g. contains `mathlib4`), return the same short message as above.
   - Avoid putting `traceback.format_exc()` into `feedback` for this case.

3. **Docs and first-run experience**  
   - In the same place that documents `LEAN_WORKSPACE` (e.g. `devlog/20260303_local_lean_verifier_setup.md`), state clearly that the default path is `Goedel-Prover-main/mathlib4` and that the user must init (and build) the submodule before using local verification.
   - Optionally: in the entrypoint or verifier, when a verification result has `is_server_error` and the feedback string contains `"No such file or directory"` and `"mathlib4"`, print a one-line hint: “Hint: ensure Goedel-Prover-main/mathlib4 exists (git submodule update --init) and is built (lake build).”

---

## 4. Why the feedback is so large

The verifier currently maps **any** exception in `verify()` to a verification error and sets the result’s `feedback` (and `errors`) from the exception’s string representation or traceback. So a single `FileNotFoundError` becomes a multi-line traceback. That’s why the log shows a “huge” error: it’s the full traceback, not a Lean compiler message. The fixes above would replace that with a short, actionable message when the failure is “workspace path missing.”

---

## 5. References

- Default workspace in code: `sdpo_modal_local_verify/local_lean_verifier.py` (`_DEFAULT_LEAN_WORKSPACE`, `verify()`).
- Submodule: `Goedel-Prover-main/.gitmodules` (mathlib4).
- Setup: `devlog/20260303_local_lean_verifier_setup.md`.
- Example log: `sdpo_results/kimina_2b_local_verify/minif2f-lean4/run_5_20260303_235305/logs.json` (iteration_logs feedback with traceback).
