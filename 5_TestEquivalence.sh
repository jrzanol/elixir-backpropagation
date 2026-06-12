#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 5_TestEquivalence.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$PROJECT_ROOT/scripts/prepared_dataset"
RESULT_DIR="$PROJECT_ROOT/reports/equivalence_complete_$(date +%Y%m%d_%H%M%S)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_DATASET="$DATASET"
export BACKPROP_TRAIN_RATIO="0.8"
export BACKPROP_EPOCHS="10"
export BACKPROP_LEARN_RATE="0.01"
export BACKPROP_SEED="42"
export BACKPROP_BATCH_SIZE="$(awk -F= '$1 == "batch_size" {print $2}' "$DATASET/metadata.txt")"
export BACKPROP_PROFILE="0"

mkdir -p "$RESULT_DIR/cuda" "$RESULT_DIR/polyhok"
cd "$PROJECT_ROOT"

echo "[INFO] Compilando e executando CUDA/NIF..."
./1_CompileCUDA.sh > "$RESULT_DIR/cuda/compile.log" 2>&1
BACKPROP_IMPL=cuda MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda" \
  BACKPROP_TEST_OUTPUT="$RESULT_DIR/cuda" \
  mix run scripts/run_equivalence_case.exs > "$RESULT_DIR/cuda/run.log" 2>&1

echo "[INFO] Compilando e executando PolyHok..."
./1_CompilePolyHok.sh > "$RESULT_DIR/polyhok/compile.log" 2>&1
BACKPROP_IMPL=polyhok MIX_BUILD_PATH="$PROJECT_ROOT/_build/polyhok" \
  BACKPROP_TEST_OUTPUT="$RESULT_DIR/polyhok" \
  mix run scripts/run_equivalence_case.exs > "$RESULT_DIR/polyhok/run.log" 2>&1

echo "[INFO] Comparando resultados..."
python3 scripts/compare_equivalence.py \
  --cuda "$RESULT_DIR/cuda" \
  --polyhok "$RESULT_DIR/polyhok" \
  --output "$RESULT_DIR"

echo "[OK] Equivalencia confirmada: $RESULT_DIR"
