#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 2_RunCUDA.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando execucao CUDA/NIF..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_IMPL="cuda"
export BACKPROP_DATASET="$PROJECT_ROOT/scripts/prepared_dataset"
export BACKPROP_TRAIN_RATIO="0.8"
export BACKPROP_EPOCHS="10"
export BACKPROP_LEARN_RATE="0.01"
export BACKPROP_BATCH_SIZE="1024"
export BACKPROP_SEED="42"
export BACKPROP_PROFILE="0"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
export MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda"
echo "[INFO] MIX_BUILD_PATH=$MIX_BUILD_PATH"
echo "[INFO] Dataset: $BACKPROP_DATASET"
echo "[INFO] epochs=$BACKPROP_EPOCHS batch_size=$BACKPROP_BATCH_SIZE lr=$BACKPROP_LEARN_RATE seed=$BACKPROP_SEED"

echo "[INFO] Rodando Main.run() com CUDA/NIF..."
mix run -e 'Main.run()'

echo "[OK] Execucao CUDA/NIF finalizada."
