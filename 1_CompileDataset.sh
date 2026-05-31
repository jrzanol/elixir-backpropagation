#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompileDataset.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando preparo do dataset..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_CSV="$PROJECT_ROOT/scripts/datasets/heart.csv"
OUTPUT_DIR="$PROJECT_ROOT/scripts/prepared_dataset"
TARGET_COLUMN="HeartDisease"
TRAIN_RATIO="0.8"
BATCH_SIZE="1024"
SEED="42"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] CSV: $DATASET_CSV"
echo "[INFO] Saida: $OUTPUT_DIR"
echo "[INFO] target_column=$TARGET_COLUMN train_ratio=$TRAIN_RATIO batch_size=$BATCH_SIZE seed=$SEED"

if [ ! -f "$DATASET_CSV" ]; then
  echo "ERRO: dataset cru nao encontrado: $DATASET_CSV" >&2
  exit 1
fi

python3 "$PROJECT_ROOT/scripts/prepare_dataset.py" \
  "$DATASET_CSV" \
  "$OUTPUT_DIR" \
  --target-column "$TARGET_COLUMN" \
  --train-ratio "$TRAIN_RATIO" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --force

echo "[OK] Dataset preparado em $OUTPUT_DIR"
