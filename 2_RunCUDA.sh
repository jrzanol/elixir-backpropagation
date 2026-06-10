#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 2_RunCUDA.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SIZE_ARG="${1:-}"

if [ "$BATCH_SIZE_ARG" = "-h" ] || [ "$BATCH_SIZE_ARG" = "--help" ]; then
  echo "Uso: ./2_RunCUDA.sh [batch_size]"
  echo "Exemplo: ./2_RunCUDA.sh 32768"
  echo "Sem argumento, usa o batch_size do scripts/prepared_dataset/metadata.txt."
  exit 0
fi

echo "[INFO] Iniciando execucao CUDA/NIF..."

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_IMPL="cuda"
export BACKPROP_DATASET="$PROJECT_ROOT/scripts/prepared_dataset"
export BACKPROP_TRAIN_RATIO="0.8"
export BACKPROP_EPOCHS="10"
export BACKPROP_LEARN_RATE="0.01"
export BACKPROP_SEED="42"
export BACKPROP_PROFILE="0"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
export MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda"
echo "[INFO] MIX_BUILD_PATH=$MIX_BUILD_PATH"
echo "[INFO] Dataset: $BACKPROP_DATASET"

if [ -n "$BATCH_SIZE_ARG" ]; then
  BACKPROP_BATCH_SIZE="$BATCH_SIZE_ARG"
else
  BACKPROP_BATCH_SIZE="$(awk -F= '$1 == "batch_size" {print $2}' "$BACKPROP_DATASET/metadata.txt")"
fi

if ! [[ "$BACKPROP_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: batch_size invalido: $BACKPROP_BATCH_SIZE" >&2
  exit 1
fi

export BACKPROP_BATCH_SIZE
echo "[INFO] epochs=$BACKPROP_EPOCHS batch_size=$BACKPROP_BATCH_SIZE lr=$BACKPROP_LEARN_RATE seed=$BACKPROP_SEED"

echo "[INFO] Rodando Main.run() com CUDA/NIF..."
mix run -e 'Main.run()'

echo "[OK] Execucao CUDA/NIF finalizada."
