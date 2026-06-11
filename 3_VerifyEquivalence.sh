#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 3_VerifyEquivalence.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${BACKPROP_DATASET:-$PROJECT_ROOT/scripts/prepared_dataset}"
REPORT_DIR="$PROJECT_ROOT/reports/equivalence_$(date +%Y%m%d_%H%M%S)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_DATASET="$DATASET"
export BACKPROP_TRAIN_RATIO="${BACKPROP_TRAIN_RATIO:-0.8}"
export BACKPROP_EPOCHS="${BACKPROP_EPOCHS:-1}"
export BACKPROP_LEARN_RATE="${BACKPROP_LEARN_RATE:-0.01}"
export BACKPROP_SEED="${BACKPROP_SEED:-42}"
export BACKPROP_PROFILE="${BACKPROP_PROFILE:-0}"
export BACKPROP_DEBUG="${BACKPROP_DEBUG:-1}"
export BACKPROP_DEBUG_VALUES="${BACKPROP_DEBUG_VALUES:-100000}"

if [ ! -f "$DATASET/metadata.txt" ]; then
  echo "ERRO: dataset preparado nao encontrado em $DATASET." >&2
  exit 1
fi

export BACKPROP_BATCH_SIZE="$(awk -F= '$1 == "batch_size" {print $2}' "$DATASET/metadata.txt")"
if ! [[ "$BACKPROP_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: batch_size invalido no metadata: $BACKPROP_BATCH_SIZE" >&2
  exit 1
fi

PYTHON="$PROJECT_ROOT/.venv-pytorch/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "ERRO: ambiente PyTorch nao encontrado. Execute ./0_InstallPyTorch.sh." >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"
cd "$PROJECT_ROOT"

require_snapshots() {
  local implementation="$1"
  local log_file="$2"

  if ! grep -q "\[DEBUG_SNAPSHOT\] impl=$implementation epoch=0" "$log_file" ||
     ! grep -q "\[DEBUG_SNAPSHOT\] impl=$implementation epoch=1" "$log_file"; then
    echo "ERRO: snapshots de $implementation nao foram gerados em $log_file." >&2
    echo "Confirme que a implementacao foi recompilada com a instrumentacao atual." >&2
    exit 1
  fi
}

echo "[INFO] Recompilando CUDA/NIF..."
./1_CompileCUDA.sh

echo "[INFO] Recompilando PolyHok..."
./1_CompilePolyHok.sh

echo "[INFO] Capturando primeira atualizacao CUDA/NIF..."
export BACKPROP_IMPL="cuda"
export MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda"
mix run -e 'Main.run()' 2>&1 | tee "$REPORT_DIR/cuda.log"
require_snapshots cuda "$REPORT_DIR/cuda.log"

echo "[INFO] Capturando primeira atualizacao PolyHok..."
export BACKPROP_IMPL="polyhok"
export MIX_BUILD_PATH="$PROJECT_ROOT/_build/polyhok"
mix run -e 'Main.run()' 2>&1 | tee "$REPORT_DIR/polyhok.log"
require_snapshots polyhok "$REPORT_DIR/polyhok.log"

echo "[INFO] Capturando primeira atualizacao PyTorch..."
"$PYTHON" "$PROJECT_ROOT/scripts/run_pytorch_cuda.py" \
  --dataset "$DATASET" \
  --train-ratio "$BACKPROP_TRAIN_RATIO" \
  --epochs "$BACKPROP_EPOCHS" \
  --learn-rate "$BACKPROP_LEARN_RATE" \
  --batch-size "$BACKPROP_BATCH_SIZE" \
  --seed "$BACKPROP_SEED" 2>&1 | tee "$REPORT_DIR/pytorch.log"
require_snapshots pytorch "$REPORT_DIR/pytorch.log"

echo "[INFO] Comparando pesos, biases e gradientes..."
"$PYTHON" "$PROJECT_ROOT/scripts/compare_debug_snapshots.py" \
  "$REPORT_DIR/cuda.log" \
  "$REPORT_DIR/polyhok.log" \
  "$REPORT_DIR/pytorch.log" | tee "$REPORT_DIR/comparison.txt"

echo "[OK] Verificacao concluida: $REPORT_DIR"
