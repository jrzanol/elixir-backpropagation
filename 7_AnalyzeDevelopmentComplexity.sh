#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 7_AnalyzeDevelopmentComplexity.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="$PROJECT_ROOT/reports/development_complexity_$(date +%Y%m%d_%H%M%S)"
COMPLEXITY_COMPILE_RUNS="${COMPLEXITY_COMPILE_RUNS:-5}"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

echo "implementation,run,microseconds,milliseconds,log" > "$RESULT_DIR/compilation_times.csv"

measure_compile() {
  local implementation="$1"
  local run="$2"
  local log_file="$3"
  local started
  local finished
  local elapsed_us

  started="$(date +%s%N)"
  if [ "$implementation" = "cuda" ]; then
    ./1_CompileCUDA.sh > "$log_file" 2>&1
  else
    if [ -d "$PROJECT_ROOT/deps/poly_hok" ]; then
      BACKPROP_SKIP_POLYHOK_DOWNLOAD=1 ./1_CompilePolyHok.sh > "$log_file" 2>&1
    else
      ./1_CompilePolyHok.sh > "$log_file" 2>&1
    fi
  fi
  finished="$(date +%s%N)"
  elapsed_us=$(( (finished - started) / 1000 ))

  echo "$implementation,$run,$elapsed_us,$((elapsed_us / 1000)),$log_file" >> "$RESULT_DIR/compilation_times.csv"
}

echo "[INFO] Medindo compilacao CUDA/NIF ($COMPLEXITY_COMPILE_RUNS execucoes)..."
for run in $(seq 1 "$COMPLEXITY_COMPILE_RUNS"); do
  echo "[INFO] Compilacao CUDA/NIF $run/$COMPLEXITY_COMPILE_RUNS"
  measure_compile "cuda" "$run" "$RESULT_DIR/compile_cuda_${run}.log"
done

echo "[INFO] Medindo compilacao PolyHok ($COMPLEXITY_COMPILE_RUNS execucoes)..."
for run in $(seq 1 "$COMPLEXITY_COMPILE_RUNS"); do
  echo "[INFO] Compilacao PolyHok $run/$COMPLEXITY_COMPILE_RUNS"
  measure_compile "polyhok" "$run" "$RESULT_DIR/compile_polyhok_${run}.log"
done

echo "[INFO] Analisando codigo e criterios qualitativos..."
python3 scripts/analyze_development_complexity_v2.py \
  --project "$PROJECT_ROOT" \
  --output "$RESULT_DIR" \
  --compilation-csv "$RESULT_DIR/compilation_times.csv" \
  --cuda-log "$RESULT_DIR/compile_cuda_${COMPLEXITY_COMPILE_RUNS}.log" \
  --polyhok-log "$RESULT_DIR/compile_polyhok_${COMPLEXITY_COMPILE_RUNS}.log"

echo "[OK] Analise concluida: $RESULT_DIR"
