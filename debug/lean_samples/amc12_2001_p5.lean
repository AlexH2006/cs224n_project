import Mathlib

theorem amc12_2001_p5 :
  Finset.prod (Finset.filter (λ x => ¬ Even x) (Finset.range 10000)) (id : ℕ → ℕ) = Nat.factorial 10000 / ((2^5000) * Nat.factorial 5000) := by
  native_decide
