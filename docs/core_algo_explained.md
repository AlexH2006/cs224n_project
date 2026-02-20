# SDPO (Self-Distilled Policy Optimization) Algorithm — Detailed Analysis

## High-Level Idea

SDPO is a **test-time RL** method where:
- **Student** = model conditioned on **base prompt** (problem only, no feedback)
- **Teacher** = **same model** conditioned on **feedback prompt** (problem + compiler errors)

The idea: the teacher "knows" what went wrong (it sees the error), so we distill its distribution onto the student so the student learns to avoid that mistake without needing the error at inference time.

---

## Chronological Pipeline

### Iteration $t$:

```
1. Generate response y ~ p_θ(· | base_prompt)           [sampling, no grad]
2. Verify y in Lean → success or error feedback
3. If success: done
4. If failure:
   a. Build feedback_prompt = base_prompt + error messages
   b. Compute SDPO loss = KL(teacher || student) on response y
   c. Backprop and update θ
5. Repeat
```

---

## Detailed Formula Walkthrough

### Notation

| Symbol | Meaning |
|--------|---------|
| $x_b$ | base prompt token ids |
| $x_f$ | feedback prompt token ids (longer, includes errors) |
| $y = (y_1, \ldots, y_L)$ | response token ids (generated proof) |
| $\theta$ | model parameters |
| $p_\theta(\cdot \mid x)$ | model's next-token distribution given context $x$ |
| $K$ | `distillation_topk` (default 20) |

### Step-by-step loss computation (`_compute_sdpo_loss`)

#### 1. Build full sequences

```python
base_input_ids = [x_b; y]        # shape (1, len(x_b) + L)
feedback_input_ids = [x_f; y]   # shape (1, len(x_f) + L)
```

#### 2. Get logits for response positions

For each position $i = 0, \ldots, L-1$, we want the logits that **predict** $y_i$ given the preceding context.

```python
student_logits = model(base_input_ids).logits[0, len(x_b)-1 : len(x_b)-1+L]
teacher_logits = model(feedback_input_ids).logits[0, len(x_f)-1 : len(x_f)-1+L]
```

So:
- `student_logits[i]` = logits for predicting $y_i$ given $(x_b, y_{<i})$
- `teacher_logits[i]` = logits for predicting $y_i$ given $(x_f, y_{<i})$

Both have shape `(L, vocab_size)`.

#### 3. Top-K log-probs

Instead of using the full vocabulary (expensive), we use only the **top-K tokens by student** plus a "tail" bucket:

```python
student_topk_logits, topk_indices = topk(student_logits, K)  # shape (L, K)
student_topk_logps = student_topk_logits - logsumexp(student_logits)  # log p_student for top-K
teacher_topk_logps = gather(teacher_logits, topk_indices) - logsumexp(teacher_logits)  # same indices
```

#### 4. Tail bucket (`_add_tail`)

For each position, we add one extra bucket representing "all other tokens":

$$\text{tail\_log} = \log\Bigl(1 - \sum_{k \in \text{top-}K} p(k)\Bigr)$$

Implemented as:
```python
log_s = logsumexp(topk_logps)  # log of sum of top-K probs
tail_log = log(1 - exp(log_s)) = log(-expm1(log_s))
```

So we get `(L, K+1)` log-probs: $[\log p_{k_1}, \ldots, \log p_{k_K}, \log p_{\text{tail}}]$.

#### 5. KL divergence

```python
kl_per_bucket = F.kl_div(teacher_with_tail.detach(), student_with_tail, log_target=True)
per_token_kl = kl_per_bucket.sum(dim=-1)  # shape (L,)
```

This computes:

$$\text{KL}_i = \sum_{j=1}^{K+1} q_j \cdot (\log q_j - \log p_j)$$

where $q$ = teacher (target), $p$ = student (input).

**Important**: `F.kl_div(input, target)` computes $\text{KL}(\text{target} \| \text{input})$, so:
- `input` = student (what we're training)
- `target` = teacher (what we're distilling from)

So we're computing **KL(teacher || student)** and minimizing it.

#### 6. Loss

```python
loss = per_token_kl.mean()  # mean over L positions
loss.backward()
optimizer.step()
```

---

## Potential Issues and Concerns

### ⚠️ Issue 1: KL direction and `F.kl_div` argument order

**Current code:**
```python
kl_per_bucket = F.kl_div(
    teacher_with_tail.detach(),  # input
    student_with_tail,            # target
    reduction="none",
    log_target=True,
)
```

**PyTorch `F.kl_div` signature:**
```
kl_div(input, target, ..., log_target=False)
```

With `log_target=True`, it computes:

$$\text{KL} = \sum_i \exp(\text{target}_i) \cdot (\text{target}_i - \text{input}_i)$$

So:
- `input` = teacher (log-probs)
- `target` = student (log-probs, since `log_target=True`)

This means we're computing **KL(student || teacher)**, not KL(teacher || student).

**This is backwards from the stated intent!**

If the goal is to **push student toward teacher** (distillation), we want:

$$\min_\theta \text{KL}(p_{\text{teacher}} \| p_{\text{student}})$$

which requires:
```python
kl_per_bucket = F.kl_div(
    student_with_tail,            # input (what we're training)
    teacher_with_tail.detach(),   # target (fixed)
    reduction="none",
    log_target=True,
)
```

**Current code has the arguments swapped.**

---

### ⚠️ Issue 2: Gradient flow through teacher

The teacher logits are computed inside `torch.no_grad()`, which is correct. But then:

```python
teacher_with_tail = self._add_tail(teacher_topk_logps)
```

This is still inside the `no_grad` block, so `teacher_with_tail` has no grad. Good.

But then:
```python
kl_per_bucket = F.kl_div(
    teacher_with_tail.detach(),  # already no grad, but .detach() is redundant
    student_with_tail,            # has grad
    ...
)
```

The `.detach()` on teacher is redundant (it's already no-grad), but harmless.

**However**, if the KL direction is wrong (Issue 1), then the gradient is wrong.

---

### ⚠️ Issue 3: What does the loss actually optimize?

With the **current** (possibly wrong) code:

- We compute KL(student || teacher) = $\sum_i p_{\text{student}}(i) \cdot (\log p_{\text{student}}(i) - \log p_{\text{teacher}}(i))$
- Minimizing this pushes student to be **more uniform** where teacher is uniform, and **sharper** where teacher is sharp — but in a mode-seeking way (student avoids putting mass where teacher doesn't).

With the **intended** (distillation) direction:

- KL(teacher || student) = $\sum_i p_{\text{teacher}}(i) \cdot (\log p_{\text{teacher}}(i) - \log p_{\text{student}}(i))$
- Minimizing this pushes student to **cover** all modes of teacher (mean-seeking).

For distillation, **KL(teacher || student)** is standard. The current code may be doing the opposite.

---

### ⚠️ Issue 4: Response generated from base prompt, but loss computed on same response

The response $y$ is sampled from the **base prompt** (no feedback). Then we compute:
- Student logits: $p_\theta(y_i \mid x_b, y_{<i})$ — same as what generated $y$
- Teacher logits: $p_\theta(y_i \mid x_f, y_{<i})$ — different context

This is **on-policy** for the student but **off-policy** for the teacher. This is fine for distillation — we're asking "how would the teacher have scored this response?" and pushing student toward that.

**No issue here**, this is standard.

---

### ⚠️ Issue 5: Reward computation (for logging)

```python
student_lp = -F.cross_entropy(student_logits.detach(), target_ids, reduction="none")
teacher_lp = -F.cross_entropy(teacher_logits, target_ids, reduction="none")
total_reward = (student_lp - teacher_lp).sum().item()
```

This computes:

$$R = \sum_{i=0}^{L-1} \bigl[\log p_{\text{student}}(y_i) - \log p_{\text{teacher}}(y_i)\bigr]$$

This is **not** a reward in the RL sense — it's just a log-likelihood ratio. It's used for logging only, so no issue, but the name is misleading.

---

### ⚠️ Issue 6: Tail bucket computation

```python
def _add_tail(log_probs: "torch.Tensor") -> "torch.Tensor":
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)
```

**Problem**: `log_probs` here are the **top-K log-probs** (already normalized over full vocab), not raw logits. So `logsumexp(log_probs)` gives:

$$\log\Bigl(\sum_{k \in \text{top-}K} p(k)\Bigr)$$

which is correct for computing the tail.

**But**: `log_s = clamp(log_s, max=-1e-7)` ensures `log_s < 0`, i.e., the top-K probs sum to less than 1. If they sum to ≥1 (which shouldn't happen for true probs), this clamps. This is a numerical safeguard.

Then:

$$\text{tail\_log} = \log(1 - \exp(\log\_s)) = \log(1 - \sum_{k \in \text{top-}K} p(k))$$

This is correct if `log_probs` are true log-probs (normalized over full vocab). Let me check:

```python
student_topk_logps = student_topk_logits - student_logsumexp
```

Yes, `student_logsumexp = logsumexp(student_logits)` over the full vocab, so `student_topk_logps` are true log-probs. **Correct.**

---

### ⚠️ Issue 7: Model in train mode during generation

```python
self.model.train()  # in setup()
...
with torch.no_grad():
    outputs = self.model.generate(...)  # generation
```

The model is in **train mode** during generation. This affects:
- Dropout (if any): will be active, adding randomness
- BatchNorm (if any): will use batch stats

For most LLMs (no dropout in inference, no batchnorm), this is fine. But it's cleaner to do:
```python
self.model.eval()
with torch.no_grad():
    outputs = self.model.generate(...)
self.model.train()
```

**Minor issue**, but could affect reproducibility.

---

### ⚠️ Issue 8: Feedback prompt may be truncated

```python
feedback_prompt_ids = self.tokenizer(
    feedback_prompt, return_tensors="pt", truncation=True, max_length=2048
).input_ids
```

If the feedback prompt (base + all accumulated errors) exceeds 2048 tokens, it gets truncated. This could cut off important error messages.

**Potential issue** for long feedback histories.

---

## Summary of Issues

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | **KL direction wrong**: `F.kl_div` args swapped | **HIGH** | Swap `teacher_with_tail` and `student_with_tail` |
| 2 | Redundant `.detach()` | Low | Remove (cosmetic) |
| 3 | Loss semantics unclear | Medium | Depends on Issue 1 |
| 4 | On-policy vs off-policy | None | Correct as-is |
| 5 | "Reward" naming misleading | Low | Rename to `log_likelihood_ratio` |
| 6 | Tail bucket | None | Correct |
| 7 | Model in train mode during generation | Low | Switch to eval for generation |
| 8 | Feedback truncation | Medium | Increase max_length or summarize feedback |

---

## Recommended Fix for Issue 1 (Critical)

The KL divergence arguments are swapped. To fix:

```python
kl_per_bucket = F.kl_div(
    student_with_tail,            # input (what we're training)
    teacher_with_tail.detach(),   # target (fixed teacher)
    reduction="none",
    log_target=True,
)
```

This computes **KL(teacher || student)**, which is the standard distillation loss: minimize the cross-entropy from teacher to student, or equivalently, make student cover teacher's modes.
