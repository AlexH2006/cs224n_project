import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

open Nat Real Int

theorem mathd_algebra_392
  (n : ℕ)
  (h₀ : Even n)
  (h₁ : ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296) :
  ((n - 2) * n * (n + 2)) / 8 = 32736 := by
  have h₂ : (n : ℤ) = 64 := by
    have h₃ : (n : ℤ) ≥ 0 := by positivity
    have h₄ : ((n : ℤ) - 2)^2 + (n : ℤ)^2 + ((n : ℤ) + 2)^2 = 12296 := h₁
    have h₅ : 3 * (n : ℤ)^2 + 8 = 12296 := by
      ring_nf at h₄ ⊢
      linarith
    have h₆ : (n : ℤ)^2 = 4096 := by
      linarith
    have h₇ : (n : ℤ) = 64 := by
      have h₈ : (n : ℤ) ≥ 0 := by positivity
      have h₉ : (n : ℤ) * (n : ℤ) = 4096 := by linarith
      have h₁₀ : (n : ℤ) = 64 := by
        nlinarith [sq_nonneg ((n : ℤ) - 64)]
      exact h₁₀
    exact h₇

  have h₃ : n = 64 := by
    have h₄ : (n : ℤ) = 64 := h₂
    have h₅ : (n : ℕ) = 64 := by
      norm_cast at h₄ ⊢
      <;> omega
    exact h₅

  rw [h₃]
  <;> norm_num
  <;> decide
