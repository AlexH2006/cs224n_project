"""
qwen_eval: Self-contained eval pipeline for Qwen/Qwen3.5-4B on MiniF2F (Lean 4).

Pipeline:
  1. Generate  — batched vLLM on Modal H100 (modal_app.py)
  2. Parse     — extract full lean4 code block from raw output (parsing.py)
  3. Verify    — Kimina Lean Server over HTTP on localhost (local_lean_verifier.py)
  4. Save      — logs.json + summary.json in baseline/ (results.py)

Entry point:
  modal run qwen_eval/modal_app.py --n-problems 20 --pass-k 4
"""
