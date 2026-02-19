---
name: SDPO Lean Adaptation Plan
overview: This plan outlines the steps to understand and adapt the SDPO (Self-Distilled Policy Optimization) framework for Lean 4 theorem proving, using Qwen as the base model, rich feedback from Lean compiler errors, and Kimina-Lean-Server for verification.
todos:
  - id: understand-sdpo
    content: Read SDPO paper and understand self-distillation mechanism
    status: completed
  - id: study-code-reward
    content: Study SDPO/verl/utils/reward_score/feedback/code.py as template for Lean reward
    status: completed
  - id: test-kimina
    content: Test Kimina-Lean-Server with sample proofs
    status: completed
  - id: prepare-minif2f
    content: Download and convert MiniF2F to SDPO parquet format
    status: completed
  - id: implement-lean-reward
    content: Implement lean.py reward function with Kimina integration
    status: completed
  - id: create-config
    content: Create lean_sdpo.yaml configuration file
    status: pending
  - id: run-experiment
    content: Run small-scale training experiment
    status: pending
isProject: false
---

