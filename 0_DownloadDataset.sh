#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_DownloadDataset.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando download do dataset..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$PROJECT_ROOT/scripts/datasets"
DATASET_URL="${1:-}"
OUTPUT_NAME="${2:-}"

if [ -z "$DATASET_URL" ]; then
  echo "Uso: ./0_DownloadDataset.sh KAGGLE_URL_OU_SLUG [nome_do_arquivo]" >&2
  echo "Exemplo: ./0_DownloadDataset.sh https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction" >&2
  echo "Exemplo: ./0_DownloadDataset.sh 'https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data?select=US_Accidents_March23.csv'" >&2
  echo "Exemplo: ./0_DownloadDataset.sh fedesoriano/heart-failure-prediction heart.csv" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] URL: $DATASET_URL"
if [ -n "$OUTPUT_NAME" ]; then
  echo "[INFO] Nome de saida solicitado: $OUTPUT_NAME"
fi

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

export DATASET_URL DATASET_DIR OUTPUT_NAME

python3 <<'PY'
import json
import os
import shutil
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import opendatasets as od
import opendatasets.utils.kaggle_api as kaggle_api

dataset_url = os.environ["DATASET_URL"]
dataset_dir = Path(os.environ["DATASET_DIR"])
output_name = os.environ.get("OUTPUT_NAME", "").strip()
kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")

parsed = urlparse(dataset_url)
selected_name = parse_qs(parsed.query).get("select", [""])[0]
selected_name = Path(unquote(selected_name)).name if selected_name else ""

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

if selected_name:
    selected_candidates = [path for path in candidates if path.name == selected_name]
    if not selected_candidates:
        available = ", ".join(path.name for path in candidates[:20])
        raise SystemExit(f"ERRO: CSV selecionado pela URL nao encontrado: {selected_name}. CSVs encontrados: {available}")
    source = selected_candidates[0]
elif len(candidates) == 1:
    source = candidates[0]
else:
    source = max(candidates, key=lambda path: path.stat().st_size)

dataset_file = dataset_dir / (output_name or source.name)

if source.resolve() != dataset_file.resolve():
    shutil.copyfile(source, dataset_file)

print(f"[INFO] CSV fonte: {source}")
print(f"[INFO] Saida: {dataset_file}")
PY

DATASET_FILE="$(python3 - <<'PY'
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

dataset_dir = Path(os.environ["DATASET_DIR"])
output_name = os.environ.get("OUTPUT_NAME", "").strip()
selected_name = parse_qs(urlparse(os.environ["DATASET_URL"]).query).get("select", [""])[0]
selected_name = Path(unquote(selected_name)).name if selected_name else ""

if output_name:
    print(dataset_dir / output_name)
elif selected_name:
    print(dataset_dir / selected_name)
else:
    newest = max(dataset_dir.rglob("*.csv"), key=lambda path: path.stat().st_mtime)
    print(dataset_dir / newest.name)
PY
)"

echo "[OK] Dataset baixado em $DATASET_FILE"
echo "[INFO] Linhas: $(wc -l < "$DATASET_FILE")"
