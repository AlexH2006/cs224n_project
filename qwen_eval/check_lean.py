"""
Quick script to verify a Lean 4 snippet with Kimina.

1. Start Kimina:  docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
2. Paste your full Lean code in the LEAN_CODE string below (use real newlines).
3. Run:  python -m qwen_eval.check_lean
"""

from qwen_eval.local_lean_verifier import verify

# Paste your full Lean 4 code below (imports + theorem + proof). Use real newlines.
LEAN_CODE = """
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

open Nat

theorem imo_1981_p6
  (f : ℕ → ℕ → ℕ)
  (h₀ : ∀ y, f 0 y = y + 1)
  (h₁ : ∀ x, f (x + 1) 0 = f x 1)
  (h₂ : ∀ x y, f (x + 1) (y + 1) = f x (f (x + 1) y)) :
  ∀ y, f 4 (y + 1) = 2^(f 4 y + 3) - 3 := by
  have h₁_form : ∀ y, f 1 y = y + 2 := by
    intro y
    induction y with
    | zero =>
      simp [h₁, h₀]
      <;> norm_num
    | succ y ih =>
      have := h₂ 0 y
      simp [h₀, ih, h₁] at this ⊢
      <;> ring_nf at this ⊢
      <;> omega

  have h₂_form : ∀ y, f 2 y = 2 * y + 3 := by
    intro y
    induction y with
    | zero =>
      simp [h₁, h₁_form, h₀]
      <;> norm_num
    | succ y ih =>
      have := h₂ 1 y
      simp [h₁_form, h₀, ih, h₂] at this ⊢
      <;> ring_nf at this ⊢
      <;> omega

  have h₃_form : ∀ y, f 3 y = 2^(y+3) - 3 := by
    intro y
    induction y with
    | zero =>
      simp [h₁, h₂_form, h₁_form, h₀]
      <;> norm_num
      <;> simp [pow_succ]
      <;> ring_nf
      <;> norm_num
    | succ y ih =>
      have h_step : f 3 (y + 1) = f 2 (f 3 y) := by
        simp [h₂]
      rw [h_step]
      rw [ih]
      rw [h₂_form]
      have h_pow_ge : 2^(y+3) ≥ 3 := by
        have h_y_ge_0 : y ≥ 0 := by omega
        have h_exp_ge_8 : 2^(y+3) ≥ 8 := by
          have h_ge_3 : y + 3 ≥ 3 := by omega
          calc
            2^(y+3) ≥ 2^3 := Nat.pow_le_pow_of_le_right (by norm_num) (by omega)
            _ = 8 := by norm_num
        omega
      have h_sub_nonzero : 2^(y+3) - 3 ≥ 0 := by omega
      have h_main : 2 * (2^(y+3) - 3) + 3 = 2^(y+4) - 3 := by
        have h_ge_6 : 2^(y+3) - 3 ≥ 0 := by omega
        have h_ge_6 : 2^(y+3) ≥ 3 := by omega
        have h_ge_6 : 2^(y+3) - 3 ≥ 0 := by omega
        have h_ge_8 : 2^(y+3) ≥ 8 := by
          have h_ge_3 : y + 3 ≥ 3 := by omega
          calc
            2^(y+3) ≥ 2^3 := Nat.pow_le_pow_of_le_right (by norm_num) (by omega)
            _ = 8 := by norm_num
        have h_ge_9 : 2^(y+3) - 3 ≥ 5 := by
          omega
        have h_ge_10 : 2 * (2^(y+3) - 3) ≥ 10 := by
          omega
        have h_ge_11 : 2^(y+4) ≥ 16 := by
          have h_ge_4 : y + 4 ≥ 4 := by omega
          calc
            2^(y+4) ≥ 2^4 := Nat.pow_le_pow_of_le_right (by norm_num) (by omega)
            _ = 16 := by norm_num
        calc
          2 * (2^(y+3) - 3) + 3 = 2 * 2^(y+3) - 6 + 3 := by
            rw [Nat.mul_sub_left_distrib (2 : ℕ) (2^(y+3)) (3)]
            <;> omega
          _ = 2^(y+4) - 6 + 3 := by
            rw [pow_succ]
            <;> ring_nf
          _ = 2^(y+4) - 3 := by
            have h_ge_12 : 2^(y+4) ≥ 6 := by
              have h_ge_4 : y + 4 ≥ 4 := by omega
              calc
                2^(y+4) ≥ 2^4 := Nat.pow_le_pow_of_le_right (by norm_num) (by omega)
                _ = 16 := by norm_num
              omega
            rw [← Nat.sub_add_cancel h_ge_12]
            <;> norm_num
            <;> omega
      rw [h_main]
      <;> norm_num

  intro y
  have h₄ : f 4 (y + 1) = f 3 (f 4 y) := by
    simp [h₂]
  rw [h₄]
  rw [h₃_form (f 4 y)]
  <;> rfl
"""


def main() -> None:
    code = LEAN_CODE.strip()
    if not code or "PASTE YOUR LEAN CODE HERE" in code:
        print("Paste your Lean code into the LEAN_CODE string in this file, then run again.")
        return
    print("Verifying with Kimina at http://localhost:8000 ...")
    result = verify(code, kimina_url="http://localhost:8000", timeout=60)
    print("success:", result["success"])
    print("complete:", result["complete"])
    print("has_sorry:", result.get("has_sorry"))
    if result.get("errors"):
        print("errors:", result["errors"])
    if result.get("feedback"):
        print("feedback:\n", result["feedback"])
    if result.get("is_server_error"):
        print("(Kimina server error — is Docker running?)")


if __name__ == "__main__":
    main()
