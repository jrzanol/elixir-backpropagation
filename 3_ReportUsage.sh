#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERRO] 3_ReportUsage.sh falhou na linha $LINENO." >&2' ERR

if [ "$#" -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Uso: ./3_ReportUsage.sh COMANDO [ARGUMENTOS...]"
  echo "Exemplo CUDA: ./3_ReportUsage.sh ./2_RunCUDA.sh"
  echo "Exemplo CUDA batch: ./3_ReportUsage.sh ./2_RunCUDA.sh 32768"
  echo "Exemplo PolyHok: ./3_ReportUsage.sh ./2_RunPolyHok.sh"
  exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_ROOT="$PROJECT_ROOT/reports"
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
REPORT_DIR="$REPORT_ROOT/resource_usage_$RUN_ID"
SAMPLES_FILE="$REPORT_DIR/samples.csv"
SUMMARY_FILE="$REPORT_DIR/summary.txt"
COMMAND_FILE="$REPORT_DIR/command.txt"
INTERVAL_SECONDS="${RESOURCE_SAMPLE_INTERVAL:-1}"

mkdir -p "$REPORT_DIR"

cd "$PROJECT_ROOT"

printf "%q " "$@" > "$COMMAND_FILE"
printf "\n" >> "$COMMAND_FILE"

echo "[INFO] Iniciando comando monitorado..."
echo "[INFO] Comando: $(cat "$COMMAND_FILE")"
echo "[INFO] Relatorio: $REPORT_DIR"
echo "[INFO] Intervalo de coleta: ${INTERVAL_SECONDS}s"

echo "timestamp,elapsed_seconds,process_cpu_percent,process_mem_percent,process_rss_mb,system_mem_used_mb,system_mem_total_mb,gpu_util_percent,gpu_mem_used_mb,gpu_mem_total_mb,gpu_power_w,gpu_temp_c" > "$SAMPLES_FILE"

"$@" &
TRAIN_PID="$!"
START_EPOCH="$(date +%s)"

children_of() {
  local parent="$1"
  local child

  pgrep -P "$parent" 2>/dev/null | while read -r child; do
    echo "$child"
    children_of "$child"
  done
}

process_tree_pids() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "$TRAIN_PID"
  fi

  children_of "$TRAIN_PID"
}

sample_process_usage() {
  local pids
  pids="$(process_tree_pids | paste -sd, -)"

  if [ -z "$pids" ]; then
    echo "0,0,0"
    return
  fi

  ps -p "$pids" -o pcpu=,pmem=,rss= 2>/dev/null | awk '
    BEGIN { cpu = 0; mem = 0; rss = 0 }
    { cpu += $1; mem += $2; rss += $3 }
    END { printf "%.2f,%.2f,%.2f", cpu, mem, rss / 1024.0 }
  '
}

sample_system_memory() {
  free -m | awk '/^Mem:/ { printf "%s,%s", $3, $2 }'
}

sample_gpu_usage() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0,0,0,0,0"
    return
  fi

  nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null |
    head -n 1 |
    awk -F, '{
      for (i = 1; i <= NF; i++) {
        gsub(/^ +| +$/, "", $i)
      }
      printf "%s,%s,%s,%s,%s", $1 + 0, $2 + 0, $3 + 0, $4 + 0, $5 + 0
    }'
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  now_epoch="$(date +%s)"
  timestamp="$(date --iso-8601=seconds)"
  elapsed="$((now_epoch - START_EPOCH))"
  process_usage="$(sample_process_usage)"
  system_memory="$(sample_system_memory)"
  gpu_usage="$(sample_gpu_usage)"

  echo "$timestamp,$elapsed,$process_usage,$system_memory,$gpu_usage" >> "$SAMPLES_FILE"
  sleep "$INTERVAL_SECONDS"
done

set +e
wait "$TRAIN_PID"
TRAIN_EXIT="$?"
set -e

awk -F, -v exit_code="$TRAIN_EXIT" -v command="$(cat "$COMMAND_FILE")" '
  NR == 1 { next }
  {
    samples += 1
    cpu_sum += $3
    mem_sum += $4
    rss_sum += $5
    gpu_sum += $8
    gpu_mem_sum += $9
    gpu_power_sum += $11

    if ($3 > cpu_max) cpu_max = $3
    if ($5 > rss_max) rss_max = $5
    if ($8 > gpu_max) gpu_max = $8
    if ($9 > gpu_mem_max) gpu_mem_max = $9
    if ($11 > gpu_power_max) gpu_power_max = $11

    last_elapsed = $2
    system_mem_used = $6
    system_mem_total = $7
    gpu_mem_total = $10
  }
  END {
    print "command=" command
    print "exit_code=" exit_code
    print "samples=" samples
    print "elapsed_seconds=" last_elapsed

    if (samples > 0) {
      printf "process_cpu_avg_percent=%.2f\n", cpu_sum / samples
      printf "process_cpu_max_percent=%.2f\n", cpu_max
      printf "process_mem_avg_percent=%.2f\n", mem_sum / samples
      printf "process_rss_avg_mb=%.2f\n", rss_sum / samples
      printf "process_rss_max_mb=%.2f\n", rss_max
      printf "system_mem_last_mb=%s/%s\n", system_mem_used, system_mem_total
      printf "gpu_util_avg_percent=%.2f\n", gpu_sum / samples
      printf "gpu_util_max_percent=%.2f\n", gpu_max
      printf "gpu_mem_avg_mb=%.2f\n", gpu_mem_sum / samples
      printf "gpu_mem_max_mb=%.2f\n", gpu_mem_max
      printf "gpu_mem_total_mb=%s\n", gpu_mem_total
      printf "gpu_power_avg_w=%.2f\n", gpu_power_sum / samples
      printf "gpu_power_max_w=%.2f\n", gpu_power_max
    }
  }
' "$SAMPLES_FILE" > "$SUMMARY_FILE"

echo "[INFO] Amostras: $SAMPLES_FILE"
echo "[INFO] Resumo: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

exit "$TRAIN_EXIT"
