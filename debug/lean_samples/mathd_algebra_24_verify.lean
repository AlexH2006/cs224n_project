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
theorem mathd_algebra_24
  (x : ℝ)
  (h₀ : x / 50 = 40) :
  x = 2000 := by
  have h₁ : x = 40 * 50 := by
    have h₂ : x / 50 = 40 := h₀
    field_simp [h₂] at h₂ ⊢
    linarith
  linarith
