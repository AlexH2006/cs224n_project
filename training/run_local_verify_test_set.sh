#!/usr/bin/env bash
# Run local-verify SDPO pipeline on a set of problem indices (test set).
# Prerequisite: mathlib4 built (cd Goedel-Prover-main/mathlib4 && lake build).
#
# Usage:
#   ./training/run_local_verify_test_set.sh           # problems 0..4 (default)
#   ./training/run_local_verify_test_set.sh 0 1 2    # problems 0, 1, 2
#   ./training/run_local_verify_test_set.sh --count 3 # first 3: 0, 1, 2

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Default: indices 0 1 2 3 4
INDICES=(0 1 2 3 4)
if [[ "$1" == "--count" ]]; then
  N="$2"
  INDICES=()
  for ((i=0;i<N;i++)); do INDICES+=( "$i" ); done
  shift 2
elif [[ $# -gt 0 ]]; then
  INDICES=("$@")
fi

echo "Running local-verify test set: ${INDICES[*]}"
for idx in "${INDICES[@]}"; do
  echo ""
  echo "========== Problem index $idx =========="
  python3 -m modal run training/lean_sdpo_local_verify_modal.py --problem-idx "$idx" || true
done
echo ""
echo "Test set run complete."
