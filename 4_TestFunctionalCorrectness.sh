#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 4_TestFunctionalCorrectness.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="$PROJECT_ROOT/reports/functional_correctness_$(date +%Y%m%d_%H%M%S)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

echo "[INFO] Compilando CUDA/NIF..."
./1_CompileCUDA.sh > "$RESULT_DIR/compile_cuda.log" 2>&1
echo "[INFO] Executando caso funcional CUDA/NIF..."
BACKPROP_IMPL=cuda MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda" BACKPROP_DEBUG=1 BACKPROP_DEBUG_VALUES=16 \
  mix run scripts/run_functional_case.exs > "$RESULT_DIR/cuda.log" 2>&1

echo "[INFO] Compilando PolyHok..."
./1_CompilePolyHok.sh > "$RESULT_DIR/compile_polyhok.log" 2>&1
echo "[INFO] Executando caso funcional PolyHok..."
BACKPROP_IMPL=polyhok MIX_BUILD_PATH="$PROJECT_ROOT/_build/polyhok" BACKPROP_DEBUG=1 BACKPROP_DEBUG_VALUES=16 \
  mix run scripts/run_functional_case.exs > "$RESULT_DIR/polyhok.log" 2>&1

echo "[INFO] Executando referencia Python..."
python3 scripts/functional_reference.py > "$RESULT_DIR/python.log"

python3 scripts/compare_functional_correctness.py \
  --output "$RESULT_DIR" \
  "$RESULT_DIR/cuda.log" "$RESULT_DIR/polyhok.log" "$RESULT_DIR/python.log"

echo "[OK] Corretude funcional confirmada: $RESULT_DIR"
