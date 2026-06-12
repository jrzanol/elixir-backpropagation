#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 7_AnalyzeDevelopmentComplexity.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="$PROJECT_ROOT/reports/development_complexity_$(date +%Y%m%d_%H%M%S)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

echo "[INFO] Medindo compilacao CUDA/NIF..."
cuda_started="$(date +%s%N)"
./1_CompileCUDA.sh > "$RESULT_DIR/compile_cuda.log" 2>&1
cuda_finished="$(date +%s%N)"
cuda_compile_us=$(( (cuda_finished - cuda_started) / 1000 ))

echo "[INFO] Medindo compilacao PolyHok..."
polyhok_started="$(date +%s%N)"
if [ -d "$PROJECT_ROOT/deps/poly_hok" ]; then
  BACKPROP_SKIP_POLYHOK_DOWNLOAD=1 ./1_CompilePolyHok.sh > "$RESULT_DIR/compile_polyhok.log" 2>&1
else
  ./1_CompilePolyHok.sh > "$RESULT_DIR/compile_polyhok.log" 2>&1
fi
polyhok_finished="$(date +%s%N)"
polyhok_compile_us=$(( (polyhok_finished - polyhok_started) / 1000 ))

cat > "$RESULT_DIR/compilation_times.csv" <<EOF
implementation,microseconds,milliseconds
cuda,$cuda_compile_us,$((cuda_compile_us / 1000))
polyhok,$polyhok_compile_us,$((polyhok_compile_us / 1000))
EOF

echo "[INFO] Analisando codigo e criterios qualitativos..."
python3 scripts/analyze_development_complexity.py \
  --project "$PROJECT_ROOT" \
  --output "$RESULT_DIR" \
  --cuda-compile-us "$cuda_compile_us" \
  --polyhok-compile-us "$polyhok_compile_us" \
  --cuda-log "$RESULT_DIR/compile_cuda.log" \
  --polyhok-log "$RESULT_DIR/compile_polyhok.log"

echo "[OK] Analise concluida: $RESULT_DIR"
