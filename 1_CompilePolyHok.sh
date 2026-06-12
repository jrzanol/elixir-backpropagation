#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompilePolyHok.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando compilacao PolyHok..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLYHOK_DIR="$PROJECT_ROOT/deps/poly_hok"
POLYHOK_REPOSITORY="https://github.com/jrzanol/poly_hok.git"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export MATREX_BLAS="${MATREX_BLAS:-noblas}"
export BACKPROP_IMPL="polyhok"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERRO: nvcc nao encontrado." >&2
  exit 127
fi

CUDA_ARCH="${BACKPROP_CUDA_ARCH:-}"
NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [ -z "$NVIDIA_SMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
fi

if [ -z "$CUDA_ARCH" ] && [ -n "$NVIDIA_SMI" ]; then
  CUDA_ARCH="$($NVIDIA_SMI --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1{s/[.[:space:]]//g;p;}')"
fi

if [ -z "$CUDA_ARCH" ]; then
  echo "ERRO: nao foi possivel detectar a arquitetura CUDA." >&2
  echo "Informe-a sem ponto. Exemplo para RTX 3060: BACKPROP_CUDA_ARCH=86 ./1_CompilePolyHok.sh" >&2
  exit 1
fi

if ! [[ "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
  echo "ERRO: arquitetura CUDA invalida: $CUDA_ARCH" >&2
  exit 1
fi

export NVCC_PREPEND_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} ${NVCC_PREPEND_FLAGS:-}"
echo "[INFO] Arquitetura CUDA: sm_$CUDA_ARCH"

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

if [ "${BACKPROP_SKIP_POLYHOK_DOWNLOAD:-0}" = "1" ]; then
  if [ ! -d "$POLYHOK_DIR" ]; then
    echo "ERRO: BACKPROP_SKIP_POLYHOK_DOWNLOAD=1, mas $POLYHOK_DIR nao existe." >&2
    exit 1
  fi
  echo "[INFO] Reutilizando fork do PolyHok existente."
else
  echo "[INFO] Baixando fork do PolyHok..."
  if [ -e "$POLYHOK_DIR" ]; then
    case "$POLYHOK_DIR" in
      "$PROJECT_ROOT"/deps/poly_hok) rm -rf -- "$POLYHOK_DIR" ;;
      *) echo "ERRO: caminho PolyHok inesperado: $POLYHOK_DIR" >&2; exit 1 ;;
    esac
  fi
  git clone "$POLYHOK_REPOSITORY" "$POLYHOK_DIR"
fi

echo "[INFO] Corrigindo compatibilidade da AST com Elixir atual..."
sed -i 's/{:__block__, \[\], definitions}/{:__block__, _, definitions}/g' \
  "$POLYHOK_DIR/lib/poly_hok/JIT.ex" \
  "$POLYHOK_DIR/lib/poly_hok/cuda_backend.ex"

if grep -R -F '{:__block__, [], definitions}' "$POLYHOK_DIR/lib" >/dev/null 2>&1; then
  echo "ERRO: nem todos os padroes AST do PolyHok foram corrigidos." >&2
  exit 1
fi

echo "[INFO] Compilando NIF principal do PolyHok..."
mkdir -p "$POLYHOK_DIR/priv"
nvcc \
  --shared \
  -g \
  -lcuda \
  -lnvrtc \
  --compiler-options '-fPIC' \
  -o "$POLYHOK_DIR/priv/gpu_nifs.so" \
  "$POLYHOK_DIR/c_src/gpu_nifs.cu"

echo "[INFO] Copiando gpu_nifs.so para priv..."
mkdir -p priv
cp "$POLYHOK_DIR/priv/gpu_nifs.so" "priv/gpu_nifs.so"

echo "[INFO] Removendo build anterior do PolyHok..."
POLYHOK_BUILD_DIR="$MIX_BUILD_PATH/lib/poly_hok"
case "$POLYHOK_BUILD_DIR" in
  "$PROJECT_ROOT"/_build/polyhok/lib/poly_hok) rm -rf -- "$POLYHOK_BUILD_DIR" ;;
  *) echo "ERRO: caminho de build PolyHok inesperado: $POLYHOK_BUILD_DIR" >&2; exit 1 ;;
esac

echo "[INFO] Removendo build anterior do projeto PolyHok..."
BACKPROP_BUILD_DIR="$MIX_BUILD_PATH/lib/backprop"
case "$BACKPROP_BUILD_DIR" in
  "$PROJECT_ROOT"/_build/polyhok/lib/backprop) rm -rf -- "$BACKPROP_BUILD_DIR" ;;
  *) echo "ERRO: caminho de build do projeto inesperado: $BACKPROP_BUILD_DIR" >&2; exit 1 ;;
esac

echo "[INFO] Baixando dependencias Elixir..."
mix deps.get

echo "[INFO] Compilando dependencias Elixir..."
mix deps.compile

echo "[INFO] Compilando projeto com BACKPROP_IMPL=polyhok..."
mix compile --force

echo "[OK] Compilacao PolyHok finalizada."
