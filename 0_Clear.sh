#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_Clear.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Limpando arquivos gerados por compilacao..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"

remove_path() {
  local path="$1"

  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "[INFO] Removendo: $path"
    rm -rf "$path"
  else
    echo "[INFO] Ignorando inexistente: $path"
  fi
}

remove_path "$PROJECT_ROOT/_build"
remove_path "$PROJECT_ROOT/_build_verify"
remove_path "$PROJECT_ROOT/priv/CudaBackpropNif.so"
remove_path "$PROJECT_ROOT/priv/gpu_nifs.so"
remove_path "$PROJECT_ROOT/priv/Elixir.App.so"
remove_path "$PROJECT_ROOT/priv/Elixir.MLPClassifier.so"
remove_path "$PROJECT_ROOT/priv/Elixir.MLPClassifierDevice.so"
remove_path "$PROJECT_ROOT/c_src/Elixir.App.cu"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifier.asts"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifier.cu"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifier.types"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifierDevice.asts"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifierDevice.cu"
remove_path "$PROJECT_ROOT/c_src/Elixir.MLPClassifierDevice.types"
remove_path "$PROJECT_ROOT/scripts/prepared_dataset"
remove_path "$PROJECT_ROOT/scripts/__pycache__"
remove_path "$PROJECT_ROOT/scripts/datasets/heart.csv.tmp"
remove_path "$PROJECT_ROOT/deps/poly_hok/priv/gpu_nifs.so"
remove_path "$PROJECT_ROOT/deps/poly_hok/priv/bmp_nifs.so"
remove_path "$PROJECT_ROOT/deps/poly_hok/_build"

echo "[OK] Limpeza finalizada."
