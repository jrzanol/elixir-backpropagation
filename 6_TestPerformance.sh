#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 6_TestPerformance.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$PROJECT_ROOT/scripts/prepared_dataset"
RESULT_DIR="$PROJECT_ROOT/reports/performance_$(date +%Y%m%d_%H%M%S)"
RUNS="${BACKPROP_PERFORMANCE_RUNS:-101}"
IMPLEMENTATIONS="${BACKPROP_PERFORMANCE_IMPLS:-cuda polyhok}"
IMPLEMENTATIONS="${IMPLEMENTATIONS//,/ }"
read -r -a IMPLEMENTATION_LIST <<< "$IMPLEMENTATIONS"

contains_implementation() {
  local expected="$1"
  local implementation
  for implementation in "${IMPLEMENTATION_LIST[@]}"; do
    if [ "$implementation" = "$expected" ]; then
      return 0
    fi
  done
  return 1
}

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: BACKPROP_PERFORMANCE_RUNS invalido: $RUNS" >&2
  exit 1
fi
if [ "$RUNS" -lt 101 ] && [ "${BACKPROP_PERFORMANCE_SMOKE:-0}" != "1" ]; then
  echo "ERRO: o benchmark exige ao menos 101 execucoes: 1 aquecimento e 100 amostras." >&2
  exit 1
fi
if [ "${#IMPLEMENTATION_LIST[@]}" -eq 0 ]; then
  echo "ERRO: BACKPROP_PERFORMANCE_IMPLS nao informou nenhuma implementacao." >&2
  exit 1
fi
for implementation in "${IMPLEMENTATION_LIST[@]}"; do
  case "$implementation" in
    cuda | polyhok) ;;
    *)
      echo "ERRO: BACKPROP_PERFORMANCE_IMPLS invalido: $implementation. Use cuda, polyhok ou ambos." >&2
      exit 1
      ;;
  esac
done

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_DATASET="$DATASET"
export BACKPROP_TRAIN_RATIO="0.8"
export BACKPROP_EPOCHS="100"
export BACKPROP_LEARN_RATE="0.01"
export BACKPROP_SEED="42"
export BACKPROP_BATCH_SIZE="$(awk -F= '$1 == "batch_size" {print $2}' "$DATASET/metadata.txt")"
export BACKPROP_PROFILE="1"

mkdir -p "$RESULT_DIR"
RAW="$RESULT_DIR/raw_executions.csv"
echo "implementation,run,warmup,compile_microseconds,application_microseconds,profile_file" > "$RAW"
cd "$PROJECT_ROOT"

echo "[INFO] Implementacoes: ${IMPLEMENTATION_LIST[*]}"

if contains_implementation "polyhok" && [ ! -d "$PROJECT_ROOT/deps/poly_hok" ]; then
  echo "[INFO] Preparando dependencia PolyHok antes das medicoes..."
  ./1_CompilePolyHok.sh > "$RESULT_DIR/polyhok_preparation.log" 2>&1
fi

for implementation in "${IMPLEMENTATION_LIST[@]}"; do
  mkdir -p "$RESULT_DIR/$implementation"

  for run in $(seq 1 "$RUNS"); do
    warmup=0
    if [ "$run" -eq 1 ]; then warmup=1; fi
    echo "[INFO] $implementation execucao $run/$RUNS (aquecimento=$warmup)..."

    compile_log="$RESULT_DIR/$implementation/compile_$run.log"
    compile_started="$(date +%s%N)"
    if [ "$implementation" = "cuda" ]; then
      ./1_CompileCUDA.sh > "$compile_log" 2>&1
      build_path="$PROJECT_ROOT/_build/cuda"
    else
      BACKPROP_SKIP_POLYHOK_DOWNLOAD=1 ./1_CompilePolyHok.sh > "$compile_log" 2>&1
      build_path="$PROJECT_ROOT/_build/polyhok"
    fi
    compile_finished="$(date +%s%N)"
    compile_us=$(( (compile_finished - compile_started) / 1000 ))

    profile_file="$RESULT_DIR/$implementation/profile_$run.csv"
    run_log="$RESULT_DIR/$implementation/run_$run.log"
    application_started="$(date +%s%N)"
    BACKPROP_IMPL="$implementation" MIX_BUILD_PATH="$build_path" \
      BACKPROP_PROFILE_FILE="$profile_file" \
      mix run -e 'Main.run()' > "$run_log" 2>&1
    application_finished="$(date +%s%N)"
    application_us=$(( (application_finished - application_started) / 1000 ))

    if [ ! -s "$profile_file" ]; then
      echo "ERRO: perfil nao gerado em $profile_file" >&2
      exit 1
    fi

    echo "$implementation,$run,$warmup,$compile_us,$application_us,$profile_file" >> "$RAW"
  done
done

python3 scripts/summarize_performance.py --raw "$RAW" --output "$RESULT_DIR"
echo "[OK] Benchmark concluido: $RESULT_DIR"
