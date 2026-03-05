import data.real.basic

theorem mathd_algebra_478
  (b h v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2) :
  v = 65 := by
  variable (b h v : ℝ)
  variable (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  variable (h₁ : v = 1 / 3 * (b * h))
  variable (h₂ : b = 30)
  variable (h₃ : h = 13 / 2)
  begin
  have h₄ : v = 1 / 3 * (b * (13 / 2)) := h₁.subst h₃,
  have h₅ : v = 1 / 3 * (30 * (13 / 2)) := h₄.subst h₂,
  simp [h₅],
  have h₆ : 195 = 65 * 3 := by ring,
  have h₇ : v = 65 * 3 := h₆,
  show v = 65, from h₇
  end
