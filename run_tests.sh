#!/usr/bin/env bash
# Run the Open Duck Mini skill-library test suite and write tests/last_run.md (+ .json, .log)
# for the dashboard.
#
#   ./run_tests.sh              full suite (Blender round trip included, ~1 min)
#   ./run_tests.sh --fast       skip tests marked slow
#   ./run_tests.sh -k clip      anything else is passed to pytest
#
# Uses sim/.venv (mujoco, onnxruntime, pytest). Never touches the GPU: JAX pinned to CPU and
# CUDA devices hidden, so it is safe to run while training is going.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/sim/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "no venv at sim/.venv; run the sim setup first" >&2
  exit 2
fi
if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "pytest missing in sim/.venv: $PY -m pip install pytest   (or: uv pip install --python $PY pytest)" >&2
  exit 2
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" XLA_PYTHON_CLIENT_PREALLOCATE=false
export DUCK_TESTS_LAST_RUN="$ROOT/tests/last_run.md"
unset REF_CLIP REF_HYBRID
ARGS=()
for a in "$@"; do
  if [ "$a" = "--fast" ]; then ARGS+=(-m "not slow"); else ARGS+=("$a"); fi
done
"$PY" -m pytest tests -q -rfs --durations=10 -p no:cacheprovider "${ARGS[@]}" 2>&1 | tee "$ROOT/tests/last_run.log"
status=${PIPESTATUS[0]}
echo "wrote tests/last_run.md and tests/last_run.json (pytest exit $status)"
exit "$status"
