#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompileDataset.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_NAME="${1:-heart.csv}"
TARGET_SELECTOR="${2:-HeartDisease}"
BATCH_SIZE="${3:-1024}"
POSITIVE_THRESHOLD="${4:-0.5}"
DATASET_CSV="$PROJECT_ROOT/scripts/datasets/$DATASET_NAME"
OUTPUT_DIR="$PROJECT_ROOT/scripts/prepared_dataset"
TRAIN_RATIO="0.8"
SEED="42"

if [ "$DATASET_NAME" = "-h" ] || [ "$DATASET_NAME" = "--help" ]; then
  echo "Uso: ./1_CompileDataset.sh [arquivo_csv] [coluna_alvo] [batch_size] [positive_threshold]"
  echo "Exemplo: ./1_CompileDataset.sh heart.csv HeartDisease"
  echo "Exemplo: ./1_CompileDataset.sh heart.csv 11 32768 0.5"
  echo "Exemplo Severity 3/4: ./1_CompileDataset.sh US_Accidents_March23.csv Severity 32768 2.5"
  exit 0
fi

echo "[INFO] Iniciando preparo do dataset..."

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] CSV: $DATASET_CSV"
echo "[INFO] Saida: $OUTPUT_DIR"
echo "[INFO] target_selector=$TARGET_SELECTOR train_ratio=$TRAIN_RATIO batch_size=$BATCH_SIZE positive_threshold=$POSITIVE_THRESHOLD seed=$SEED"

if [ ! -f "$DATASET_CSV" ]; then
  echo "ERRO: dataset cru nao encontrado: $DATASET_CSV" >&2
  exit 1
fi

if ! python3 "$PROJECT_ROOT/scripts/prepare_dataset.py" --help | grep -q -- "--positive-threshold"; then
  echo "ERRO: scripts/prepare_dataset.py esta desatualizado e nao suporta --positive-threshold." >&2
  echo "Atualize/copiei o arquivo scripts/prepare_dataset.py antes de preparar targets nao binarios." >&2
  exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERRO: batch_size deve ser um inteiro positivo. Valor recebido: $BATCH_SIZE" >&2
  exit 1
fi

if ! python3 - "$POSITIVE_THRESHOLD" <<'PY'
import sys
float(sys.argv[1])
PY
then
  echo "ERRO: positive_threshold deve ser numerico. Valor recebido: $POSITIVE_THRESHOLD" >&2
  exit 1
fi

TARGET_COLUMN="$TARGET_SELECTOR"

if [[ "$TARGET_SELECTOR" =~ ^[0-9]+$ ]]; then
  TARGET_COLUMN="$(
    python3 - "$DATASET_CSV" "$TARGET_SELECTOR" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
target_index = int(sys.argv[2])

with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.reader(handle)
    header = next(reader)

if target_index < 0 or target_index >= len(header):
    raise SystemExit(f"indice da coluna alvo fora do intervalo: {target_index}. Total de colunas: {len(header)}")

print(header[target_index].strip())
PY
  )"
fi

echo "[INFO] target_column=$TARGET_COLUMN"

python3 "$PROJECT_ROOT/scripts/prepare_dataset.py" \
  "$DATASET_CSV" \
  "$OUTPUT_DIR" \
  --target-column "$TARGET_COLUMN" \
  --train-ratio "$TRAIN_RATIO" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --positive-threshold "$POSITIVE_THRESHOLD" \
  --force

echo "[OK] Dataset preparado em $OUTPUT_DIR"
