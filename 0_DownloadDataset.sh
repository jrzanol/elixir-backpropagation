#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_DownloadDataset.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando download do dataset..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$PROJECT_ROOT/scripts/datasets"
DATASET_URL="${1:-}"
DATASET_NAME="${2:-heart.csv}"
DATASET_FILE="$DATASET_DIR/$DATASET_NAME"
TMP_FILE="$DATASET_FILE.tmp"
EXPECTED_HEADER="Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease"

if [ -z "$DATASET_URL" ]; then
  echo "Uso: ./0_DownloadDataset.sh URL [nome_do_arquivo]" >&2
  echo "Exemplo: ./0_DownloadDataset.sh https://huggingface.co/datasets/aai530-group6/heart-failure-prediction-dataset/resolve/main/heart.csv" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] URL: $DATASET_URL"
echo "[INFO] Saida: $DATASET_FILE"

mkdir -p "$DATASET_DIR"

if command -v curl >/dev/null 2>&1; then
  curl -fL "$DATASET_URL" -o "$TMP_FILE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_FILE" "$DATASET_URL"
else
  echo "ERRO: curl ou wget nao encontrado." >&2
  exit 1
fi

header="$(head -n 1 "$TMP_FILE" | tr -d '\r')"
if [ "$header" != "$EXPECTED_HEADER" ]; then
  echo "ERRO: o arquivo baixado nao parece ser o heart.csv esperado." >&2
  echo "Header recebido: $header" >&2
  rm -f "$TMP_FILE"
  exit 1
fi

mv "$TMP_FILE" "$DATASET_FILE"

echo "[OK] Dataset baixado em $DATASET_FILE"
echo "[INFO] Linhas: $(wc -l < "$DATASET_FILE")"
