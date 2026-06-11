#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_InstallPyTorch.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-pytorch"
PYTORCH_VERSION="2.11.0"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "[INFO] Instalando PyTorch $PYTORCH_VERSION com CUDA 12.8..."
echo "[INFO] Ambiente: $VENV_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERRO: python3 nao encontrado." >&2
  exit 127
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "ERRO: modulo venv nao encontrado." >&2
  echo "Ubuntu/Debian: apt-get update && apt-get install -y python3-venv" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  "torch==$PYTORCH_VERSION" \
  --index-url "$PYTORCH_INDEX"
"$VENV_DIR/bin/python" -m pip install numpy

"$VENV_DIR/bin/python" - <<'PY'
import sys
import numpy
import torch

print(f"[INFO] Python: {sys.version.split()[0]}")
print(f"[INFO] PyTorch: {torch.__version__}")
print(f"[INFO] NumPy: {numpy.__version__}")
print(f"[INFO] CUDA do PyTorch: {torch.version.cuda}")
print(f"[INFO] CUDA disponivel: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("ERRO: PyTorch instalado, mas a GPU CUDA nao esta acessivel.")

print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY

echo "[OK] PyTorch/CUDA instalado."
