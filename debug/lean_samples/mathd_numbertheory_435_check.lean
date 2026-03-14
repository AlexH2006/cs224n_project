import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem mathd_numbertheory_435
  (k : ℕ)
  (h₀ : 0 < k)
  (h₁ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 3) = 1)
  (h₂ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 2) = 1)
  (h₃ : ∀ n, Nat.gcd (6 * n + k) (6 * n + 1) = 1) :
  5 ≤ k := by
  by_contra! h
  have h₄ : k ≤ 4 := by linarith
  interval_cases k <;> norm_num [Nat.gcd_eq_right, Nat.gcd_eq_left, Nat.gcd_comm] at h₁ h₂ h₃ <;>
    (try omega) <;>
    (try {
      have h₅ := h₁ 1
      have h₆ := h₂ 1
      have h₇ := h₃ 1
      norm_num at h₅ h₆ h₇
      omega
    })
