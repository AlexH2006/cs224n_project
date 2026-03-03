# SDPO output analysis: loss, reward, and why rewards are positive when proofs fail

**Date:** 2026-03-03  
**Topics:** sdpo, reward, loss, distillation, devlog

---

Analysis of `sdpo_results/kimina_distill_1_7b/minif2f-lean4/run_30_20260302_215539/logs.json` (and the shared SDPO logic in `lean_sdpo_kimina_distill_1_7b_modal.py`). Explains how **loss** and **reward** are computed and why the logged **reward** is positive even when the model never proves the theorem.

---

## 1. What gets logged each iteration

Each iteration log includes:

- **Verification:** `success`, `complete`, `has_sorry`, `feedback` (from Kimina).
- **Training metrics:** `loss`, `reward`, `kl_div`, `entropy`, `grad_norm`.

In the run above, all 5 iterations **fail** (no complete proof, `has_sorry: true` or truncated), yet **rewards** are all positive (e.g. 8.375, 7.625, 8.4375, 5.15625, 11.375). So “reward” here does **not** mean “proof quality.”

---

## 2. How reward is computed (and that it’s not from verification)

In this SDPO setup, **reward is not derived from verification**. It is computed inside `_compute_sdpo_loss()` as:

```python
# student_lp / teacher_lp = log P(response | prompt) for student / teacher
total_reward = (student_lp - teacher_lp).sum().item()
```

- **Student:** same model conditioned only on the **problem** (base prompt).
- **Teacher:** same model conditioned on **problem + feedback** (e.g. “Previous errors: … Avoid these errors. Provide corrected proof tactics.”).
- **Response:** the same generated sequence (tokens) for both.

So:

**Reward = sum over response tokens of (log P_student(token) − log P_teacher(token)).**

It measures how much more (or less) likely the **student** is to produce this exact response than the **teacher**. It is a **log-probability difference**, not a proof-correctness signal.

---

## 3. Why rewards are positive when the proof fails

- The **teacher** prompt includes “Previous errors: … Avoid these errors.” So the teacher is conditioned on “don’t do this again.”
- The **generated** response is still the kind that caused those errors (repetitive, no real tactics, ends in `sorry`).
- So:
  - **Teacher** tends to assign **lower** probability to this same (bad) response.
  - **Student** does **not** see the feedback, so it assigns **higher** probability to the same response.
- Hence **student_lp − teacher_lp > 0** on many tokens → **total_reward > 0**.

So “positive reward” means “the student preferred this output more than the teacher did,” not “the proof was good.” That can happen even when the proof fails every time.

---

## 4. How loss is computed (what is actually optimized)

The **loss** used for backprop is **not** reward-weighted. It is:

```python
# KL(teacher || student) per token (on top-K), then mean
per_token_kl = F.kl_div(teacher_with_tail.detach(), student_with_tail, ...).sum(dim=-1)
loss = per_token_kl.mean()
loss.backward()
```

So:

- **Loss = mean over tokens of KL(teacher ‖ student)** on the generated response (over the top-K vocabulary).
- The gradient comes only from this KL term. The scalar **reward** is **not** used in the loss or in `backward()`.

Training therefore always does one step: pull the **student** distribution toward the **teacher** on the current response. The teacher’s distribution is the “target” because it has seen the feedback.

---

## 5. Role of verification in the pipeline

- **Verification** (Kimina) is used to:
  - decide whether the attempt is a full proof (`success` and `complete`),
  - update **feedback** (e.g. “Reasoning was cut off…”, “Be concise…”),
  - and to build the **teacher prompt** for the *next* iteration (problem + feedback history).
- **Feedback** only affects the **teacher** prompt. It does **not** change the formula for **reward** or **loss** on the *current* iteration.
- The **reward** in the logs is a **diagnostic**: it is the (student − teacher) log-prob sum. For a “proof quality” metric, use verification outcome (e.g. `success` and `complete`), not this reward.

---

## 6. Summary table

| Quantity    | Formula / meaning | Used in backward? |
|------------|--------------------|------------------|
| **Loss**   | Mean over tokens of KL(teacher ‖ student) on top-K | Yes (only this) |
| **Reward** | Sum over tokens of (log P_student − log P_teacher) | No (logged only) |
| Verification | success, complete, has_sorry, feedback from Kimina | Feeds next teacher prompt only |

So: **loss** is the only training signal; **reward** explains why the student often looks “more confident” than the teacher on the same (failing) output; **verification** drives feedback and thus the teacher’s context for the next try.
