#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 2_RunPyTorch.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$PROJECT_ROOT/scripts/prepared_dataset"
EPOCHS=10
LEARN_RATE=0.01
SEED=42
TRAIN_RATIO=0.8
VENV_PYTHON="$PROJECT_ROOT/.venv-pytorch/bin/python"

echo "[INFO] Iniciando execucao PyTorch/CUDA..."
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] Dataset: $DATASET"

cd "$PROJECT_ROOT"

if [ -x "$VENV_PYTHON" ]; then
  PYTHON="$VENV_PYTHON"
else
  PYTHON="python3"
fi

if ! "$PYTHON" -c 'import numpy, torch' >/dev/null 2>&1; then
  echo "ERRO: PyTorch ou NumPy nao esta instalado no ambiente .venv-pytorch." >&2
  echo "Execute: ./0_InstallPyTorch.sh" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  echo "ERRO: PyTorch foi encontrado, mas nao possui acesso a GPU CUDA." >&2
  echo "Verifique o driver com: nvidia-smi" >&2
  exit 1
fi

BATCH_SIZE="$(awk -F= '$1 == "batch_size" {print $2}' "$DATASET/metadata.txt")"
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: batch_size invalido: $BATCH_SIZE" >&2
  exit 1
fi

echo "[INFO] epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LEARN_RATE seed=$SEED"

"$PYTHON" scripts/run_pytorch_cuda.py \
  --dataset "$DATASET" \
  --train-ratio "$TRAIN_RATIO" \
  --epochs "$EPOCHS" \
  --learn-rate "$LEARN_RATE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED"

echo "[OK] Execucao PyTorch/CUDA finalizada."
