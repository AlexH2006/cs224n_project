import tactic

theorem mathd_numbertheory_3 :
  (∑ x in Finset.range 10, ((x + 1)^2)) % 10 = 5 := by
  (∑ x in Finset.range (n + 1), x^2)
  let sum := sum_of_squares_of_first_n_integers 9
  sum % 10
  by
  let sum := sum_of_squares_of_first_n_integers 9
  let units_digit := sum % 10
  exact units_digit = 5
  #check mathd_numbertheory_3
