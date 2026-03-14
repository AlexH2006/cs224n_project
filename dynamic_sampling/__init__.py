"""
Dynamic sampling: multi-round evaluation on MiniF2F with a total attempt budget.

Run pass@1 for all problems, then repeatedly for remaining problems with
pass@floor(remaining_budget/n), stopping when total attempts reach the budget.
"""
