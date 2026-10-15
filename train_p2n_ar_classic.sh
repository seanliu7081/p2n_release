#!/usr/bin/env bash
set -euo pipefail
P2N_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${TRAIN_PY:-python}" "$P2N_ROOT/scripts/train_policy_variant.py" --variant p2n_ar_classic "$@"
