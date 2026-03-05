import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/- The volume of a cone is given by the formula \( V = \frac{1}{3}Bh \), where \( B \) is the area of the base and \( h \) is the height.
   The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65. -/
theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
                            (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
  -- Substitute b and h in the formula for v
  rw [h₂, h₃] at h₁
  -- Simplify the expression
  have h₁' : v = 1 / 3 * (30 * (13 / 2)) := h₁
  simp only [mul_div_assoc, mul_assoc] at h₁'
  -- Calculate the value on the right side of the equation
  have h₁'' : v = 1 / 3 * (390 / 2) := h₁'
  have h₁''' : v = 1 / 3 * 195 := h₁''
  have h₁'''' : v = 195 / 3 := h₁'''
  exact div_eq_of_mul_eq (show v * 3 = 195 by rw [h₁'''']; rfl)
