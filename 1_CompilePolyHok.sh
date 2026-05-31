#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompilePolyHok.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando compilacao PolyHok..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_IMPL="polyhok"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"

export MIX_BUILD_PATH="$PROJECT_ROOT/_build/polyhok"
echo "[INFO] MIX_BUILD_PATH=$MIX_BUILD_PATH"

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

export ERL_INCLUDE_DIR
export CPATH="$ERL_INCLUDE_DIR:${CPATH:-}"
echo "[INFO] ERL_INCLUDE_DIR=$ERL_INCLUDE_DIR"

if [ ! -d "deps/poly_hok" ]; then
  echo "ERRO: deps/poly_hok nao encontrado." >&2
  exit 1
fi

echo "[INFO] Baixando dependencias Elixir..."
mix deps.get

echo "[INFO] Compilando dependencias Elixir..."
mix deps.compile

if [ ! -s "deps/poly_hok/priv/gpu_nifs.so" ]; then
  echo "[INFO] Compilando deps/poly_hok..."
  mkdir -p deps/poly_hok/priv
  make -C deps/poly_hok
else
  echo "[INFO] PolyHok ja possui deps/poly_hok/priv/gpu_nifs.so"
fi

echo "[INFO] Copiando gpu_nifs.so para priv..."
mkdir -p priv
cp "deps/poly_hok/priv/gpu_nifs.so" "priv/gpu_nifs.so"

echo "[INFO] Compilando projeto com BACKPROP_IMPL=polyhok..."
mix compile

echo "[OK] Compilacao PolyHok finalizada."
