#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 8_ReportEnvironment.sh falhou na linha $LINENO." >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_ROOT="$PROJECT_ROOT/reports"
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
REPORT_DIR="$REPORT_ROOT/environment_$RUN_ID"
RAW_DIR="$REPORT_DIR/raw"
SUMMARY_CSV="$REPORT_DIR/environment_summary.csv"
REPORT_MD="$REPORT_DIR/environment_report.md"
AUTO_INSTALL="${ENV_REPORT_AUTO_INSTALL:-1}"
COMMAND_TIMEOUT="${ENV_REPORT_COMMAND_TIMEOUT:-20}"
APT_TIMEOUT="${ENV_REPORT_APT_TIMEOUT:-120}"

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:/usr/local/cuda/bin:/usr/local/cuda-12/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$RAW_DIR"
cd "$PROJECT_ROOT"

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

install_basic_tools() {
  if [ "$AUTO_INSTALL" != "1" ] || ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  local packages=()
  command -v lspci >/dev/null 2>&1 || packages+=(pciutils)
  command -v lsb_release >/dev/null 2>&1 || packages+=(lsb-release)
  command -v free >/dev/null 2>&1 || packages+=(procps)
  command -v lscpu >/dev/null 2>&1 || packages+=(util-linux)
  command -v lsblk >/dev/null 2>&1 || packages+=(util-linux)

  if [ "${#packages[@]}" -eq 0 ]; then
    return
  fi

  echo "[INFO] Instalando ferramentas ausentes: ${packages[*]}"
  {
    if command -v sudo >/dev/null 2>&1; then
      timeout "$APT_TIMEOUT" sudo apt-get update
      timeout "$APT_TIMEOUT" sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    else
      timeout "$APT_TIMEOUT" apt-get update
      timeout "$APT_TIMEOUT" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    fi
  } > "$RAW_DIR/apt_install_environment_tools.txt" 2>&1 || {
    echo "[WARN] Nao foi possivel instalar todas as ferramentas auxiliares. Continuando com o que estiver disponivel."
  }
}

csv_escape() {
  local value="${1//$'\r'/}"
  value="${value//$'\n'/; }"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

add_kv() {
  local section="$1"
  local key="$2"
  local value="${3:-nao detectado}"
  {
    csv_escape "$section"
    printf ","
    csv_escape "$key"
    printf ","
    csv_escape "$value"
    printf "\n"
  } >> "$SUMMARY_CSV"
}

capture_command() {
  local name="$1"
  shift

  if timeout "$COMMAND_TIMEOUT" "$@" > "$RAW_DIR/$name.txt" 2>&1; then
    return 0
  fi

  printf "comando falhou: " > "$RAW_DIR/$name.txt"
  printf "%q " "$@" >> "$RAW_DIR/$name.txt"
  printf "\n" >> "$RAW_DIR/$name.txt"
  return 0
}

capture_shell() {
  local name="$1"
  local command="$2"

  if timeout "$COMMAND_TIMEOUT" bash -lc "$command" > "$RAW_DIR/$name.txt" 2>&1; then
    return 0
  fi

  {
    echo "comando falhou:"
    echo "$command"
  } > "$RAW_DIR/$name.txt"
  return 0
}

one_line() {
  timeout "$COMMAND_TIMEOUT" "$@" 2>/dev/null | head -n 1 | tr -d '\r' || true
}

shell_one_line() {
  timeout "$COMMAND_TIMEOUT" bash -lc "$1" 2>/dev/null | head -n 1 | tr -d '\r' || true
}

trim() {
  sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

powershell_value() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    return 0
  fi

  timeout "$COMMAND_TIMEOUT" powershell.exe -NoProfile -Command "$1" 2>/dev/null | tr -d '\r' | head -n 1 || true
}

install_basic_tools

echo "section,key,value" > "$SUMMARY_CSV"

echo "[INFO] Coletando informacoes do ambiente..."
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] Relatorio: $REPORT_DIR"

capture_command os_release cat /etc/os-release
capture_command uname uname -a
capture_command kernel_version cat /proc/version
capture_command cpu_lscpu lscpu
capture_command cpu_procinfo cat /proc/cpuinfo
capture_command memory_free free -h
capture_command memory_meminfo cat /proc/meminfo
capture_shell disk_df "df -h ."
capture_command disk_lsblk lsblk
capture_command pci_lspci lspci
capture_command nvidia_smi nvidia-smi
capture_command nvidia_smi_query nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,pci.bus_id --format=csv
capture_command nvcc_version nvcc --version
capture_command gcc_version gcc --version
capture_command gpp_version g++ --version
capture_command make_version make --version
capture_command git_version git --version
capture_command python3_version python3 --version
capture_command pip3_version python3 -m pip --version
capture_command elixir_version elixir --version
capture_command mix_version mix --version
capture_command erlang_otp erl -noshell -eval 'io:format("~s~n", [erlang:system_info(otp_release)]), halt().'
capture_command asdf_current asdf current
capture_command project_git_status git status --short
capture_command project_git_log git log -1 --decorate --stat
capture_shell project_backprop_env "env | grep '^BACKPROP_' | sort"

if command -v powershell.exe >/dev/null 2>&1; then
  capture_command windows_cmd_version cmd.exe /c ver
  capture_command windows_os powershell.exe -NoProfile -Command \
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::UTF8; Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory | Format-List"
  capture_command windows_cpu powershell.exe -NoProfile -Command \
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::UTF8; Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List"
  capture_command windows_gpu powershell.exe -NoProfile -Command \
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::UTF8; Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | Format-List"
else
  echo "powershell.exe nao encontrado; provavelmente Linux nativo ou PATH sem integracao WSL." > "$RAW_DIR/windows_os.txt"
fi

PYTORCH_PYTHON=""
if [ -x "$PROJECT_ROOT/.venv-pytorch/bin/python" ]; then
  PYTORCH_PYTHON="$PROJECT_ROOT/.venv-pytorch/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTORCH_PYTHON="$(command -v python3)"
fi

if [ -n "$PYTORCH_PYTHON" ]; then
  timeout "$COMMAND_TIMEOUT" "$PYTORCH_PYTHON" - <<'PY' > "$RAW_DIR/python_libraries.txt" 2>&1 || true
import importlib

for name in ["numpy", "torch", "sklearn", "pandas"]:
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'sem __version__')}")
        if name == "torch":
            print(f"torch_cuda={module.version.cuda}")
            print(f"torch_cuda_available={module.cuda.is_available()}")
            if module.cuda.is_available():
                print(f"torch_gpu={module.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"{name}=nao encontrado ({exc})")
PY
else
  echo "python3 nao encontrado." > "$RAW_DIR/python_libraries.txt"
fi

add_kv "Execucao" "data_hora_local" "$(date +"%Y-%m-%d %H:%M:%S %z")"
add_kv "Execucao" "projeto" "$PROJECT_ROOT"
add_kv "Execucao" "relatorio" "$REPORT_DIR"
add_kv "Execucao" "usuario" "$(whoami 2>/dev/null || true)"
add_kv "Execucao" "hostname" "$(hostname 2>/dev/null || true)"

if [ -f /etc/os-release ]; then
  # shellcheck source=/dev/null
  . /etc/os-release
  add_kv "Linux" "distribuicao" "${PRETTY_NAME:-nao detectado}"
  add_kv "Linux" "id" "${ID:-nao detectado}"
  add_kv "Linux" "versao" "${VERSION_ID:-nao detectado}"
fi
add_kv "Linux" "kernel" "$(uname -srmo 2>/dev/null || true)"
add_kv "Linux" "wsl_detectado" "$(grep -qi microsoft /proc/version 2>/dev/null && echo sim || echo nao)"
if command -v wslinfo >/dev/null 2>&1; then
  add_kv "Linux" "wslinfo" "$(wslinfo --version 2>/dev/null | tr '\n' '; ' | trim)"
fi

add_kv "Windows" "caption" "$(powershell_value "(Get-CimInstance Win32_OperatingSystem).Caption")"
add_kv "Windows" "version" "$(powershell_value "(Get-CimInstance Win32_OperatingSystem).Version")"
add_kv "Windows" "build" "$(powershell_value "(Get-CimInstance Win32_OperatingSystem).BuildNumber")"
add_kv "Windows" "architecture" "$(powershell_value "(Get-CimInstance Win32_OperatingSystem).OSArchitecture")"

add_kv "CPU" "modelo" "$(shell_one_line "lscpu | awk -F: '/Model name/ {print \$2; exit}' | sed 's/^[[:space:]]*//'")"
add_kv "CPU" "arquitetura" "$(shell_one_line "lscpu | awk -F: '/Architecture/ {print \$2; exit}' | sed 's/^[[:space:]]*//'")"
add_kv "CPU" "cpus_logicas" "$(nproc 2>/dev/null || true)"
add_kv "CPU" "sockets" "$(shell_one_line "lscpu | awk -F: '/Socket\\(s\\)/ {print \$2; exit}' | sed 's/^[[:space:]]*//'")"
add_kv "CPU" "cores_por_socket" "$(shell_one_line "lscpu | awk -F: '/Core\\(s\\) per socket/ {print \$2; exit}' | sed 's/^[[:space:]]*//'")"
add_kv "CPU" "threads_por_core" "$(shell_one_line "lscpu | awk -F: '/Thread\\(s\\) per core/ {print \$2; exit}' | sed 's/^[[:space:]]*//'")"

add_kv "Memoria" "ram_total" "$(shell_one_line "free -h | awk '/^Mem:/ {print \$2}'")"
add_kv "Memoria" "ram_disponivel" "$(shell_one_line "free -h | awk '/^Mem:/ {print \$7}'")"
add_kv "Memoria" "swap_total" "$(shell_one_line "free -h | awk '/^Swap:/ {print \$2}'")"

if command -v nvidia-smi >/dev/null 2>&1; then
  add_kv "GPU" "cuda_driver_runtime" "$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([^ |]*\).*/\1/p' | head -n 1)"
  while IFS=, read -r index name driver memory compute bus; do
    index="$(echo "$index" | trim)"
    name="$(echo "$name" | trim)"
    driver="$(echo "$driver" | trim)"
    memory="$(echo "$memory" | trim)"
    compute="$(echo "$compute" | trim)"
    bus="$(echo "$bus" | trim)"
    if [ "$index" != "index" ] && [ -n "$index" ]; then
      add_kv "GPU" "gpu_${index}_nome" "$name"
      add_kv "GPU" "gpu_${index}_driver" "$driver"
      add_kv "GPU" "gpu_${index}_memoria_total" "$memory"
      add_kv "GPU" "gpu_${index}_compute_capability" "$compute"
      add_kv "GPU" "gpu_${index}_pci_bus" "$bus"
    fi
  done < "$RAW_DIR/nvidia_smi_query.txt"
else
  add_kv "GPU" "nvidia_smi" "nao encontrado"
fi

add_kv "CUDA" "nvcc" "$(one_line nvcc --version)"
add_kv "CUDA" "nvcc_release" "$(nvcc --version 2>/dev/null | sed -n 's/.*release \([^,]*\).*/\1/p' | head -n 1)"
add_kv "CUDA" "CUDA_HOME" "${CUDA_HOME:-nao definido}"
add_kv "CUDA" "CUDA_PATH" "${CUDA_PATH:-nao definido}"

add_kv "Elixir/Erlang" "elixir" "$(elixir --version 2>/dev/null | tr '\n' '; ' | trim)"
add_kv "Elixir/Erlang" "mix" "$(mix --version 2>/dev/null | tr '\n' '; ' | trim)"
add_kv "Elixir/Erlang" "erlang_otp" "$(erl -noshell -eval 'io:format("~s", [erlang:system_info(otp_release)]), halt().' 2>/dev/null || true)"
add_kv "Elixir/Erlang" "asdf" "$(asdf current 2>/dev/null | tr '\n' '; ' | trim)"

add_kv "Compiladores" "gcc" "$(one_line gcc --version)"
add_kv "Compiladores" "g++" "$(one_line g++ --version)"
add_kv "Compiladores" "make" "$(one_line make --version)"
add_kv "Compiladores" "git" "$(git --version 2>/dev/null || true)"

add_kv "Python" "python3" "$(python3 --version 2>/dev/null || true)"
add_kv "Python" "pip3" "$(python3 -m pip --version 2>/dev/null || true)"
if [ -f "$RAW_DIR/python_libraries.txt" ]; then
  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    [ -n "$key" ] && add_kv "Python" "$key" "$value"
  done < "$RAW_DIR/python_libraries.txt"
fi

add_kv "Projeto" "branch" "$(git branch --show-current 2>/dev/null || true)"
add_kv "Projeto" "commit" "$(git rev-parse HEAD 2>/dev/null || true)"
add_kv "Projeto" "dirty_files" "$(git status --short 2>/dev/null | wc -l | tr -d ' ')"
add_kv "Projeto" "remote_origin" "$(git remote get-url origin 2>/dev/null || true)"
if [ -f "$PROJECT_ROOT/.tool-versions" ]; then
  add_kv "Projeto" "tool_versions" "$(tr '\n' '; ' < "$PROJECT_ROOT/.tool-versions" | trim)"
fi

if [ -d "$PROJECT_ROOT/deps/poly_hok/.git" ]; then
  add_kv "PolyHok" "deps_poly_hok_commit" "$(git -C "$PROJECT_ROOT/deps/poly_hok" rev-parse HEAD 2>/dev/null || true)"
  add_kv "PolyHok" "deps_poly_hok_branch" "$(git -C "$PROJECT_ROOT/deps/poly_hok" branch --show-current 2>/dev/null || true)"
  add_kv "PolyHok" "deps_poly_hok_remote" "$(git -C "$PROJECT_ROOT/deps/poly_hok" remote get-url origin 2>/dev/null || true)"
else
  add_kv "PolyHok" "deps_poly_hok" "nao encontrado"
fi

write_section_table() {
  local section="$1"
  awk -F, -v section="$section" '
    NR > 1 {
      s = $1; k = $2; v = $3
      gsub(/^"|"$/, "", s); gsub(/^"|"$/, "", k); gsub(/^"|"$/, "", v)
      gsub(/""/, "\"", s); gsub(/""/, "\"", k); gsub(/""/, "\"", v)
      if (s == section) {
        gsub(/\|/, "\\|", k); gsub(/\|/, "\\|", v)
        print "| `" k "` | " v " |"
      }
    }
  ' "$SUMMARY_CSV"
}

{
  echo "# Relatorio do ambiente de execucao"
  echo
  echo "Gerado em: $(date +"%Y-%m-%d %H:%M:%S %z")"
  echo
  echo "Este relatorio registra o ambiente onde os benchmarks foram executados. Ele serve para tornar os resultados reproduziveis e para documentar diferencas de hardware, sistema operacional, drivers, CUDA, Elixir/Erlang e dependencias relevantes."
  echo
  echo "## Como ler"
  echo
  echo "- \`environment_summary.csv\`: tabela chave/valor consolidada para importar em planilha."
  echo "- \`raw/\`: saidas brutas dos comandos usados, preservadas para auditoria."
  echo "- Valores marcados como \`nao detectado\` indicam que a ferramenta ou integracao nao estava disponivel no ambiente."
  echo

  for section in "Execucao" "Linux" "Windows" "CPU" "Memoria" "GPU" "CUDA" "Elixir/Erlang" "Compiladores" "Python" "Projeto" "PolyHok"; do
    echo "## $section"
    echo
    echo "| Item | Valor |"
    echo "|---|---|"
    write_section_table "$section"
    echo
  done

  echo "## Arquivos brutos"
  echo
  find "$RAW_DIR" -maxdepth 1 -type f -printf "- \`raw/%f\`\n" | sort
} > "$REPORT_MD"

echo "[OK] Relatorio do ambiente gerado:"
echo "  Markdown: $REPORT_MD"
echo "  CSV:      $SUMMARY_CSV"
echo "  Raw:      $RAW_DIR"
