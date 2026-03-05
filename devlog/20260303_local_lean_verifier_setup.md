# Local Lean verifier setup (sdpo_modal)

The **local** Lean verification path (`sdpo_modal.local_lean_verifier`) uses `lake exe repl` in a mathlib4 workspace. No Docker or Kimina server required.

## Requirements

1. **Lean 4 toolchain (elan)**  
   Install from [Lean 4 quickstart](https://leanprover.github.io/lean4/doc/quickstart.html). This provides `lake` (default: `~/.elan/bin/lake`).

2. **mathlib4 workspace**  
   The verifier runs with `cwd` set to a mathlib4 directory (must contain `lakefile.lean` / Lakefile and be built).

   - **Default path:** `Goedel-Prover-main/mathlib4/` (relative to repo root).  
   - Ensure the Goedel-Prover mathlib4 submodule is present and built:
     ```bash
     cd Goedel-Prover-main
     git submodule update --init --recursive   # if needed
     cd mathlib4 && lake build
     cd ../..
     ```

3. **Run from repo root**  
   So that `sdpo_modal` and `Goedel-Prover-main` resolve correctly.

## Environment (optional)

| Variable         | Meaning |
|------------------|--------|
| `LAKE_PATH`      | Path to `lake` executable (default: `~/.elan/bin/lake`). |
| `LEAN_WORKSPACE` | Path to mathlib4 directory (default: `{repo}/Goedel-Prover-main/mathlib4`). |

## Kimina backend (sdpo_modal_local_verify)

To use **Kimina Lean Server** (e.g. running in Docker on your machine) instead of local `lake exe repl`:

1. Start Kimina in Docker (e.g. `docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0` or your image).
2. Set env (or pass kwargs from the runner):
   - `LEAN_VERIFY_BACKEND=kimina` (default for the local-verify pipeline).
   - `LEAN_VERIFY_KIMINA_URL=http://localhost:8000` (optional; this is the default).
   - `LEAN_VERIFY_KIMINA_API_KEY=...` (optional; if the server requires auth).

Verification semantics stay Goedel-Prover-style; only the transport is HTTP to Kimina.

### Troubleshooting: "Request failed: [Errno 61] Connection refused"

This means the pipeline is using the **Kimina** backend and nothing is listening at the Kimina URL (default `http://localhost:8000`). The Kimina Lean Server (Docker or host process) is not running or not exposed on that port.

**Fix one of:**

1. **Start the Kimina server** (e.g. in another terminal):
   ```bash
   docker run --rm -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
   ```
   Then re-run the pipeline.

2. **Use local verification instead** (no Docker):
   ```bash
   export LEAN_VERIFY_BACKEND=local
   ```
   Then run the pipeline; verification will use `lake exe repl` in your mathlib4 workspace (see "Requirements" above).

## Sanity check

From repo root:

```bash
python debug/test_local_lean_verifier.py
```

This verifies a known-good and known-bad Lean snippet and checks the VerifyResult contract.
