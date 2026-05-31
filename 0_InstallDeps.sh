#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 0_InstallDeps.sh falhou na linha $LINENO." >&2' ERR

echo "[INFO] Iniciando instalacao das dependencias Elixir/Erlang..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERLANG_VERSION="27.3.4"
ELIXIR_VERSION="1.18.4-otp-27"
ASDF_DIR="${ASDF_DIR:-$HOME/.asdf}"

cd "$PROJECT_ROOT"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] Erlang: $ERLANG_VERSION"
echo "[INFO] Elixir: $ELIXIR_VERSION"

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

if command -v apt-get >/dev/null 2>&1; then
  echo "[INFO] Instalando pacotes base via apt..."
  run_sudo apt-get update
  run_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    autoconf \
    build-essential \
    ca-certificates \
    curl \
    git \
    libncurses-dev \
    libssl-dev \
    libwxgtk3.2-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libssh-dev \
    libxml2-utils \
    m4 \
    make \
    openjdk-17-jdk \
    python3 \
    python3-pip \
    unzip \
    xsltproc
else
  echo "[WARN] apt-get nao encontrado. Pulando instalacao de pacotes do sistema."
fi

if [ ! -d "$ASDF_DIR" ]; then
  echo "[INFO] Instalando asdf em $ASDF_DIR..."
  git clone https://github.com/asdf-vm/asdf.git "$ASDF_DIR" --branch v0.14.1
else
  echo "[INFO] asdf ja encontrado em $ASDF_DIR"
fi

# shellcheck source=/dev/null
. "$ASDF_DIR/asdf.sh"

export PATH="$ASDF_DIR/bin:$ASDF_DIR/shims:$PATH"

if ! asdf plugin list | grep -qx "erlang"; then
  echo "[INFO] Adicionando plugin erlang..."
  asdf plugin add erlang https://github.com/asdf-vm/asdf-erlang.git
fi

if ! asdf plugin list | grep -qx "elixir"; then
  echo "[INFO] Adicionando plugin elixir..."
  asdf plugin add elixir https://github.com/asdf-vm/asdf-elixir.git
fi

if ! asdf list erlang 2>/dev/null | sed 's/^[ *]*//' | grep -qx "$ERLANG_VERSION"; then
  echo "[INFO] Instalando Erlang $ERLANG_VERSION..."
  asdf install erlang "$ERLANG_VERSION"
else
  echo "[INFO] Erlang $ERLANG_VERSION ja instalado."
fi

if ! asdf list elixir 2>/dev/null | sed 's/^[ *]*//' | grep -qx "$ELIXIR_VERSION"; then
  echo "[INFO] Instalando Elixir $ELIXIR_VERSION..."
  asdf install elixir "$ELIXIR_VERSION"
else
  echo "[INFO] Elixir $ELIXIR_VERSION ja instalado."
fi

echo "[INFO] Definindo versoes locais do projeto..."
asdf local erlang "$ERLANG_VERSION"
asdf local elixir "$ELIXIR_VERSION"
asdf reshim

echo "[INFO] Instalando Hex e Rebar..."
mix local.hex --force
mix local.rebar --force

ERL_INCLUDE_DIR="$(find "$ASDF_DIR/installs/erlang/$ERLANG_VERSION" -name erl_nif.h -print -quit 2>/dev/null | xargs -r dirname)"

if [ -z "$ERL_INCLUDE_DIR" ] || [ ! -f "$ERL_INCLUDE_DIR/erl_nif.h" ]; then
  echo "ERRO: erl_nif.h nao encontrado apos instalar Erlang." >&2
  exit 1
fi

echo "[OK] Dependencias instaladas."
echo "[INFO] Elixir: $(elixir --version | tail -n 1)"
echo "[INFO] ERL_INCLUDE_DIR=$ERL_INCLUDE_DIR"
echo "[INFO] Para esta sessao, se necessario:"
echo "export PATH=\"$ASDF_DIR/shims:$ASDF_DIR/bin:\$PATH\""
echo "export ERL_INCLUDE_DIR=\"$ERL_INCLUDE_DIR\""
