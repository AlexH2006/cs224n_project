# Qwen 3.5 4B Token Logprob + KL

This folder lets you score hand-written Lean proofs with:

- per-token log-probability under `AI-MO/Kimina-Prover-RL-1.7B` (or another policy model)
- per-token KL divergence `D_KL(policy || reference)` at each proof token

## Input format

Use JSONL (recommended) or JSON.

### JSONL example

```json
{"id":"proof_1","context":"theorem t : True := by\n","proof":"  trivial\n"}
{"id":"proof_2","context":"theorem add_comm_demo (a b : Nat) : a + b = b + a := by\n","proof":"  simpa [Nat.add_comm]\n"}
```

Fields:

- `proof` (required): Lean proof text you want to score
- `context` (optional): theorem statement/imports/prefix
- `id` (optional): identifier used in output files

## Run

From repo root:

```bash
python qwen35_token_kl/compute_token_logprobs_kl.py \
  --input qwen35_token_kl/examples/manual_proofs.jsonl \
  --output-dir qwen35_token_kl/output \
  --model AI-MO/Kimina-Prover-RL-1.7B
```

If `--reference-model` is omitted:
- For Kimina-Prover models: defaults to `Qwen/Qwen3-1.7B` (base) for non-zero KL.
- Otherwise: defaults to the policy model (KL will be 0).

## Outputs

- `qwen35_token_kl/output/per_token_stats.jsonl`
  - one row per scored proof token with token string, `policy_logprob`, `reference_logprob`, `token_kl`, cumulative KL
- `qwen35_token_kl/output/summary.json`
  - aggregate metrics per proof (`mean_policy_logprob`, `mean_token_kl`, etc.)
- `qwen35_token_kl/output/token_logprob_heatmap.png`
  - per-proof heatmaps (white→red) of per-token policy log probability; one subplot per proof, no overlap. Use `--no-plot` to skip.

## Notes

- KL is computed over full vocab per token step: `D_KL(policy || reference)`.
- Policy and reference must use compatible tokenizers (the script checks this).
- Use a GPU if available; CPU is much slower for 4B models.
