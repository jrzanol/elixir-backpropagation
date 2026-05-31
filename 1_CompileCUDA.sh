#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompileCUDA.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando compilacao CUDA/NIF..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_IMPL="cuda"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"

export MIX_BUILD_PATH="$PROJECT_ROOT/_build/cuda"
echo "[INFO] MIX_BUILD_PATH=$MIX_BUILD_PATH"

NIF_ROOT="$PROJECT_ROOT/c_src/MLPClassifierNIF"
PRIV_DIR="$PROJECT_ROOT/priv"
OUT="$PRIV_DIR/CudaBackpropNif.so"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERRO: nvcc nao encontrado." >&2
  exit 127
fi

if [ -z "${ERL_INCLUDE_DIR:-}" ]; then
  echo "[INFO] Procurando erl_nif.h..."
  ERL_INCLUDE_DIR=""
  for dir in "$HOME/.asdf/installs/erlang" "$HOME/.asdl/installs/erlang" /usr/lib/erlang /usr/local/lib/erlang; do
    if [ -d "$dir" ]; then
      ERL_INCLUDE_DIR="$(find "$dir" -name erl_nif.h -print -quit 2>/dev/null | xargs -r dirname)"
      if [ -n "$ERL_INCLUDE_DIR" ]; then
        break
      fi
    fi
  done
fi

if [ -z "$ERL_INCLUDE_DIR" ] || [ ! -f "$ERL_INCLUDE_DIR/erl_nif.h" ]; then
  echo "ERRO: erl_nif.h nao encontrado." >&2
  echo "Instale Erlang/OTP com headers ou exporte ERL_INCLUDE_DIR antes de rodar o script." >&2
  echo "Exemplo: export ERL_INCLUDE_DIR=/usr/lib/erlang/usr/include" >&2
  exit 1
fi
echo "[INFO] ERL_INCLUDE_DIR=$ERL_INCLUDE_DIR"

echo "[INFO] Baixando dependencias Elixir..."
mix deps.get

echo "[INFO] Compilando dependencias Elixir..."
mix deps.compile --force

echo "[INFO] Compilando NIF CUDA com nvcc..."
mkdir -p "$PRIV_DIR"

nvcc \
  -std=c++17 \
  -O2 \
  -gencode=arch=compute_50,code=sm_50 \
  -Xcompiler -fPIC \
  -shared \
  -I"$ERL_INCLUDE_DIR" \
  -I"$NIF_ROOT" \
  -o "$OUT" \
  "$NIF_ROOT/MLPClassifierNIF.cu" \
  "$NIF_ROOT/MLPClassifierNIFDevice.cu" \
  "$NIF_ROOT/CudaBackpropNif.cpp"

echo "NIF gerado: $OUT"

echo "[INFO] Compilando projeto com BACKPROP_IMPL=cuda..."
mix compile

echo "[OK] Compilacao CUDA/NIF finalizada."
