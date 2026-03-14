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
open Nat

theorem imo_1959_p1
  (n : ℕ)
  (h₀ : 0 < n) :
  Nat.gcd (21*n + 4) (14*n + 3) = 1 := by
  have h₁ : Nat.gcd (21*n + 4) (14*n + 3) = Nat.gcd (7*n + 1) (14*n + 3) := by
    rw [Nat.gcd_comm]
    rw [Nat.gcd_comm]
    rw [show 21*n + 4 = (7*n + 1) + (14*n + 3) by ring]
    simp [Nat.gcd_add_mul_right_right]
  rw [h₁]
  have h₂ : Nat.gcd (7*n + 1) (14*n + 3) = Nat.gcd (7*n + 1) 1 := by
    have h₃ : 14*n + 3 = 2*(7*n + 1) + 1 := by ring
    rw [h₃]
    simp [Nat.gcd_add_mul_left_right, Nat.gcd_one_right]
  rw [h₂]
  simp [Nat.gcd_one_left]
