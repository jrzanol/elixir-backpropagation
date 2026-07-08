#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/reports/manual_sequence_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

show_dataset_metadata() {
  if [ -f "$PROJECT_ROOT/scripts/prepared_dataset/metadata.txt" ]; then
    grep -E "batch_size|train_batch_count|test_batch_count" \
      "$PROJECT_ROOT/scripts/prepared_dataset/metadata.txt"
  else
    echo "ERRO: dataset preparado nao encontrado em scripts/prepared_dataset." >&2
    return 1
  fi
}

run_step() {
  local name="$1"
  shift

  echo
  echo "=== $name ==="
  echo "Log: $LOG_DIR/$name.log"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

echo "Logs em: $LOG_DIR"

echo
echo "=== Dataset atual antes do primeiro teste ==="
show_dataset_metadata | tee "$LOG_DIR/dataset_atual_metadata.log"

run_step "performance_atual" ./6_TestPerformance.sh

run_step "dataset_16384" ./1_CompileDataset.sh HIGGS.csv 0 16384 0.5 no_header
show_dataset_metadata | tee "$LOG_DIR/dataset_16384_metadata.log"

run_step "performance_16384" ./6_TestPerformance.sh

run_step "dataset_65536" ./1_CompileDataset.sh HIGGS.csv 0 65536 0.5 no_header
show_dataset_metadata | tee "$LOG_DIR/dataset_65536_metadata.log"

run_step "performance_65536" ./6_TestPerformance.sh

echo
echo "=== Sequencia concluida ==="
echo "Logs em: $LOG_DIR"
