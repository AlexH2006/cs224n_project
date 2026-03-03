# Verification Report: MiniF2F-test + Pass@1 + 30s Verifier Timeout

## A) Dataset correctness

### A1) Every place we call load_dataset

**File:** `baseline/lean_baseline_eval_modal.py`  
**Function:** `generate_proofs_parallel` (lines 639–645)

```python
if cfg.dataset_subset:
    ds = load_dataset(cfg.dataset_name, cfg.dataset_subset, split=cfg.dataset_split)
else:
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
```

Only location where the dataset is loaded.

### A2) Dataset name and split

**Config defaults (EvalConfig):**
- `dataset_name: str = "HaimingW/minif2f-lean4"` (line 97)
- `dataset_split: str = "test"` (line 99)

**main() defaults:**
- `dataset: str = "HaimingW/minif2f-lean4"` (line 843)
- `split: str = "test"` (line 844)

Both are passed into EvalConfig (lines 864–865), so the dataset and split used at runtime are `HaimingW/minif2f-lean4` and `test`.

### A3) dataset_subset / config

**EvalConfig:** `dataset_subset: Optional[str] = None` (line 98)

**main():** Does not pass `dataset_subset` to EvalConfig, so it remains `None`.

When `dataset_subset` is None, the code uses `load_dataset(cfg.dataset_name, split=cfg.dataset_split)` and no config/subset parameter is used.

### A4) Warning when len != 244

**Lines 647–649:**
```python
ds_len = len(ds)
if ds_len != 244:
    print(f"[WARNING] Test split has {ds_len} problems (expected 244). Dataset: {cfg.dataset_name}, split: {cfg.dataset_split}")
```

Warning is printed when the loaded split length is not 244.

**Gap:** `dataset_name`, `split`, and `len(ds)` are not logged when len == 244. Add a positive log of dataset_name, split, and len(ds) for easier auditing.

---

## B) Problem selection correctness

### B1) Selection logic location

**File:** `baseline/lean_baseline_eval_modal.py`  
**Function:** `generate_proofs_parallel` (lines 651–659)

### B2) Distinct indices

- **Full dataset:** `selected = list(range(ds_len))` — indices 0..ds_len-1, all distinct.
- **Random sample:** `selected = sorted(random.sample(range(ds_len), n_sample))` — `random.sample` returns distinct elements.

### B3) Behavior

| Condition | Behavior |
|-----------|----------|
| `n_problems >= 244` OR `n_problems >= ds_len` | `selected = list(range(ds_len))` — all indices in order |
| Else | `selected = sorted(random.sample(...))` — exactly n_sample distinct indices, reproducible with fixed seed |

**Gap:** `n_problems is None` is not handled; type is `int` with default 244, so None is not a valid value. No change needed.

### B4) Sanity check

**Current state:** No `assert len(selected) == len(set(selected))` and no log of `selected[:5]`, `selected[-5:]`.

**Proposed fix:** Add after line 659:
```python
assert len(selected) == len(set(selected)), "selected indices must be distinct"
print(f"[generate] indices: n={len(selected)}, first5={selected[:5]}, last5={selected[-5:]}")
```

---

## C) Pass@k correctness

### C1) Where k is defined and used

**main():** `pass_k: int = 1` (line 842), passed to `generate_proofs_parallel(cfg, pass_k=pass_k, ...)` (lines 897–900).

**generate_proofs_parallel:**
```python
for pidx in selected:
    for a in range(pass_k):
        jobs.append((pidx, a))
```
- Jobs: one (problem_idx, attempt) pair per (problem, attempt).

**ProofGenerator.generate_one:** Invoked once per job, with distinct `attempt` values; SamplingParams use `seed=attempt + 1337` (line 599).

### C2) k=1: exactly 1 attempt per problem

For `pass_k=1`, the loop creates one job per problem, each with `attempt=0`. Each job triggers one `generate_one` call. No retries or “try until success”; each attempt is a single generation call.

### C3) k>1: independent attempts and pass@k

- Each job is a separate `gen.generate_one.map(...)` call with different `attempt`.
- Different `attempt` ⇒ different `seed` ⇒ independent samples.
- `verify_proofs_serial` iterates attempts for a problem and stops on first success (lines 768–771):
```python
if v and v.get("success") and v.get("complete") and not v.get("has_sorry", False):
    success = True
    break
```
- Pass@k is “any success among k attempts” per problem.

**Verification:** Pass@k semantics and independent attempts are implemented correctly.

---

## D) Verifier timeout correctness

### D1) Where VERIFY_TIMEOUT_S is used

**Definition:** `VERIFY_TIMEOUT_S = 30` (line 46)

**Usage:** `KiminaLeanServer.verify` (lines 269–271):
```python
with httpx.Client(timeout=float(VERIFY_TIMEOUT_S)) as client:
    resp = client.post(
        "http://localhost:8000/verify",
        json={
            "codes": [{"custom_id": custom_id, "proof": lean_code}],
            "infotree_type": "original",
        },
    )
```

### D2) Client vs server timeout

- **Client-side:** `httpx.Client(timeout=30)` caps how long the HTTP client waits. After 30s, `httpx.TimeoutException` is raised and we treat it as a timeout.
- **Server-side:** The request body does not include a timeout. The Kimina Lean Server may have its own Lean task timeout; that is not controlled from this code.

### D3) Effective timeout

From the evaluation side, each verification attempt is treated as failed after 30 seconds. The Kimina Lean Server runs inside the same Modal container; there is no extra network hop, so the 30s limit effectively applies to the overall request.

### D4) Request payload

The POST payload is:
```json
{
  "codes": [{"custom_id": "...", "proof": "<lean code>"}],
  "infotree_type": "original"
}
```
No timeout parameter is sent. Timeout is enforced solely by the HTTP client.

**Conclusion:** Timeout behavior matches the intended 30s per attempt from the client’s perspective. If the Kimina server exposes a configurable Lean timeout, that would need to be set separately.

---

## E) Environment / reproducibility logging

### E1) Current logging

**Currently logged:**
- `[config] model: {cfg.model_name}` (line 873)

**Not logged:**
- dataset_name  
- split  
- len(ds) (only when len != 244)  
- n_problems  
- seed  
- k (pass_k)  
- verifier timeout (VERIFY_TIMEOUT_S)  
- model revision/hash  
- Kimina server endpoint  

### E2) Model revision

Model loading uses `EvalConfig().model_name` and `AutoTokenizer.from_pretrained(...)`; no revision or hash is logged. HuggingFace resolves the default revision (usually `main`).

**Proposed fix:** Add after the dataset load in `generate_proofs_parallel` (around line 647):
```python
print(f"[config] dataset={cfg.dataset_name} split={cfg.dataset_split} len={ds_len} n_problems={n_problems} seed={seed} pass_k={pass_k} verify_timeout_s={VERIFY_TIMEOUT_S}")
```
And before generation (in ProofGenerator.setup), add:
```python
from huggingface_hub import HfApi
rev = HfApi().model_info(cfg.model_name).sha or "unknown"
print(f"[config] model_revision={rev}")
```
Or, if revision tracking is not essential, at least log `model_name` (already done) and `verify_timeout_s`.

---

## F) Summary of gaps and proposed fixes

| Item | Status | Proposed fix |
|------|--------|--------------|
| Dataset HaimingW/minif2f-lean4, split=test | OK | — |
| dataset_subset not changing split | OK | — |
| Warning when len != 244 | OK | — |
| Positive log of dataset/split/len | Missing | Add log when len == 244 as well |
| Distinct-indices sanity check | Missing | Add `assert len(selected)==len(set(selected))` and log first/last 5 indices |
| n_problems=None handling | N/A | Not used (type is int) |
| Pass@k=1: 1 attempt per problem | OK | — |
| Pass@k>1: independent attempts | OK | — |
| VERIFY_TIMEOUT_S=30 | OK | — |
| Timeout is client-side HTTP | Note | Server-side Lean timeout not configurable here |
| Environment logging | Incomplete | Add dataset/split/len/n_problems/seed/pass_k/verify_timeout_s |
| Model revision logging | Missing | Optional: add HfApi().model_info(...).sha or similar |

---

## Minimal code changes to apply

1. **Selection sanity check** (after line 659):
```python
assert len(selected) == len(set(selected)), "selected indices must be distinct"
print(f"[generate] selected indices: n={len(selected)}, first5={selected[:5]}, last5={selected[-5:]}")
```

2. **Environment logging** (at start of generate_proofs_parallel, after ds_len):
```python
print(f"[config] dataset={cfg.dataset_name} split={cfg.dataset_split} len={ds_len} n_problems={n_problems} seed={seed} pass_k={pass_k} verify_timeout_s={VERIFY_TIMEOUT_S}")
```

3. **Optional model revision** (in ProofGenerator.setup): Add one-line log of resolved model revision if `huggingface_hub` is available.
