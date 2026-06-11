#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 1_CompileCUDA.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando compilacao CUDA/NIF..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.asdl/shims:$HOME/.asdl/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:/usr/lib/wsl/lib:$PATH"
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

CUDA_ARCH="${BACKPROP_CUDA_ARCH:-}"
NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [ -z "$NVIDIA_SMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
fi

if [ -z "$CUDA_ARCH" ] && [ -n "$NVIDIA_SMI" ]; then
  CUDA_ARCH="$($NVIDIA_SMI --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1{s/[.[:space:]]//g;p;}')"
fi

if ! [[ "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
  echo "ERRO: nao foi possivel detectar uma arquitetura CUDA valida." >&2
  echo "Informe-a sem ponto. Exemplo: BACKPROP_CUDA_ARCH=86 ./1_CompileCUDA.sh" >&2
  exit 1
fi

echo "[INFO] Arquitetura CUDA: sm_$CUDA_ARCH"

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
  -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
  -Xcompiler -fPIC \
  -shared \
  -I"$ERL_INCLUDE_DIR" \
  -I"$NIF_ROOT" \
  -o "$OUT" \
  "$NIF_ROOT/MLPClassifierNIF.cu" \
  "$NIF_ROOT/MLPClassifierNIFDevice.cu" \
  "$NIF_ROOT/CudaBackpropNif.cpp"

echo "NIF gerado: $OUT"

if command -v cuobjdump >/dev/null 2>&1; then
  if ! cuobjdump --list-elf "$OUT" | grep -q "sm_${CUDA_ARCH}"; then
    echo "ERRO: o NIF nao contem codigo para sm_$CUDA_ARCH." >&2
    exit 1
  fi
fi

echo "[INFO] Removendo build anterior do projeto CUDA..."
BACKPROP_BUILD_DIR="$MIX_BUILD_PATH/lib/backprop"
case "$BACKPROP_BUILD_DIR" in
  "$PROJECT_ROOT"/_build/cuda/lib/backprop) rm -rf -- "$BACKPROP_BUILD_DIR" ;;
  *) echo "ERRO: caminho de build CUDA inesperado: $BACKPROP_BUILD_DIR" >&2; exit 1 ;;
esac

echo "[INFO] Compilando projeto com BACKPROP_IMPL=cuda..."
mix compile --force

echo "[OK] Compilacao CUDA/NIF finalizada."
