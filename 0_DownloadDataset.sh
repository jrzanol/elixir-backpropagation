#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_DownloadDataset.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando download do dataset..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$PROJECT_ROOT/scripts/datasets"
DATASET_URL="${1:-}"
DATASET_NAME="${2:-heart.csv}"
DATASET_FILE="$DATASET_DIR/$DATASET_NAME"
EXPECTED_HEADER="Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease"

if [ -z "$DATASET_URL" ]; then
  echo "Uso: ./0_DownloadDataset.sh KAGGLE_URL_OU_SLUG [nome_do_arquivo]" >&2
  echo "Exemplo: ./0_DownloadDataset.sh https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction" >&2
  echo "Exemplo: ./0_DownloadDataset.sh fedesoriano/heart-failure-prediction heart.csv" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] URL: $DATASET_URL"
echo "[INFO] Saida: $DATASET_FILE"

mkdir -p "$DATASET_DIR"

if [[ "$DATASET_URL" == *"kaggle.com"* || "$DATASET_URL" != http* ]]; then
  if [ -f "$PROJECT_ROOT/.kaggle/kaggle.json" ]; then
    export KAGGLE_CONFIG_DIR="$PROJECT_ROOT/.kaggle"
    echo "[INFO] Credenciais Kaggle encontradas em $PROJECT_ROOT/.kaggle/kaggle.json"
  elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
    echo "[INFO] Credenciais Kaggle encontradas em $HOME/.kaggle/kaggle.json"
  else
    echo "[INFO] Credenciais Kaggle nao encontradas em $PROJECT_ROOT/.kaggle/kaggle.json nem em $HOME/.kaggle/kaggle.json"
    echo "[INFO] O opendatasets pode pedir usuario Kaggle e API token nesta execucao."
  fi
fi

if ! python3 -c "import opendatasets" >/dev/null 2>&1; then
  echo "ERRO: pacote Python opendatasets nao encontrado." >&2
  echo "Rode ./0_InstallDeps.sh ou instale com: python3 -m pip install --user opendatasets" >&2
  exit 1
fi

export DATASET_URL DATASET_DIR DATASET_FILE EXPECTED_HEADER

python3 <<'PY'
import json
import os
import shutil
from pathlib import Path

import opendatasets as od
import opendatasets.utils.kaggle_api as kaggle_api

dataset_url = os.environ["DATASET_URL"]
dataset_dir = Path(os.environ["DATASET_DIR"])
dataset_file = Path(os.environ["DATASET_FILE"])
expected_header = os.environ["EXPECTED_HEADER"]
kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")

if kaggle_config_dir:
    kaggle_json = Path(kaggle_config_dir) / "kaggle.json"
    if kaggle_json.exists():
        with kaggle_json.open("r", encoding="utf-8") as handle:
            credentials = json.load(handle)

        os.environ.setdefault("KAGGLE_USERNAME", credentials.get("username", ""))
        os.environ.setdefault("KAGGLE_KEY", credentials.get("key", ""))

        if os.environ["KAGGLE_USERNAME"] and os.environ["KAGGLE_KEY"]:
            kaggle_api.read_kaggle_creds = lambda: True

before = {path.resolve() for path in dataset_dir.rglob("*.csv")}
od.download(dataset_url, data_dir=str(dataset_dir))
after = sorted(
    (path for path in dataset_dir.rglob("*.csv") if path.resolve() not in before),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not after:
    candidates = sorted(dataset_dir.rglob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
else:
    candidates = after

source = None
for candidate in candidates:
    with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip().replace("\r", "")

    if header == expected_header:
        source = candidate
        break

if source is None:
    raise SystemExit("ERRO: nenhum CSV baixado possui o header esperado.")

if source.resolve() != dataset_file.resolve():
    shutil.copyfile(source, dataset_file)

print(f"[INFO] CSV fonte: {source}")
PY

header="$(head -n 1 "$DATASET_FILE" | tr -d '\r')"
if [ "$header" != "$EXPECTED_HEADER" ]; then
  echo "ERRO: o arquivo baixado nao parece ser o heart.csv esperado." >&2
  echo "Header recebido: $header" >&2
  exit 1
fi

echo "[OK] Dataset baixado em $DATASET_FILE"
echo "[INFO] Linhas: $(wc -l < "$DATASET_FILE")"
