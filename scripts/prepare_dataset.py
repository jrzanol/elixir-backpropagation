#!/usr/bin/env python3
"""
Converte um CSV para um dataset normalizado em batches binarios float32.

O formato gerado foi feito para ser simples de ler em Elixir:
- metadata.txt: metadados chave=valor.
- train/*.bpbatch e test/*.bpbatch:
  "BPBATCH1" + uint32_le(count) + uint32_le(n_features) +
  features float32 little-endian row-major + labels float32 little-endian.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import struct
import time
from pathlib import Path


MASK64 = 0xFFFFFFFFFFFFFFFF
XORSHIFT_MUL = 2685821657736338717
ROW_SEED_MUL = 11_400_714_819_323_198_485
BATCH_MAGIC = b"BPBATCH1"


def seed_xor_shift(value: int) -> int:
    value &= MASK64
    return 1 if value == 0 else value


def next_xor_shift(state: int) -> tuple[int, int]:
    x = state & MASK64
    x ^= x >> 12
    x ^= (x << 25) & MASK64
    x ^= x >> 27
    x &= MASK64
    return x, (x * XORSHIFT_MUL) & MASK64


def next_float(state: int) -> tuple[int, float]:
    new_state, value = next_xor_shift(state)
    top53 = value >> 11
    return new_state, top53 * (1.0 / math.pow(2, 53))


def train_row(row_index: int, train_ratio: float, seed: int) -> bool:
    if train_ratio >= 1.0:
        return True
    if train_ratio <= 0.0:
        return False

    state = seed_xor_shift(seed + row_index * ROW_SEED_MUL)
    _, value = next_float(state)
    return value < train_ratio


def numeric_value(value: str) -> float | None:
    value = value.strip()
    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_label(value: str, positive_threshold: float) -> float:
    parsed = numeric_value(value)
    if parsed is None:
        parsed = 0.0
    return 1.0 if parsed > positive_threshold else 0.0


def detect_target(header: list[str], target_column: str | None) -> int:
    if target_column and target_column in header:
        return header.index(target_column)
    if "fraud_bool" in header:
        return header.index("fraud_bool")
    return len(header) - 1


def read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                return [col.strip() for col in row]
    raise ValueError(f"CSV vazio: {csv_path}")


def scan_csv(
    csv_path: Path,
    feature_indices: list[int],
    target_index: int,
    train_ratio: float,
    seed: int,
) -> dict:
    label_maps: dict[int, dict[str, int]] = {}
    mins = [float("inf")] * len(feature_indices)
    maxs = [float("-inf")] * len(feature_indices)
    train_count = 0
    test_count = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        for row_index, row in enumerate(reader):
            if not row:
                continue

            encoded = []
            for feature_pos, col_index in enumerate(feature_indices):
                raw = row[col_index].strip()
                parsed = numeric_value(raw)

                if col_index in label_maps:
                    col_map = label_maps[col_index]
                    if parsed is None and raw not in col_map:
                        col_map[raw] = len(col_map)
                    parsed = float(col_map.get(raw, 0))
                elif parsed is None:
                    col_map = label_maps.setdefault(col_index, {})
                    if raw not in col_map:
                        col_map[raw] = len(col_map)
                    parsed = float(col_map[raw])

                encoded.append(parsed)

            if train_row(row_index, train_ratio, seed):
                train_count += 1
                for idx, value in enumerate(encoded):
                    if value < mins[idx]:
                        mins[idx] = value
                    if value > maxs[idx]:
                        maxs[idx] = value
            else:
                test_count += 1

    for idx, value in enumerate(mins):
        if value == float("inf"):
            mins[idx] = 0.0
            maxs[idx] = 1.0

    return {
        "label_maps": label_maps,
        "mins": mins,
        "maxs": maxs,
        "train_count": train_count,
        "test_count": test_count,
    }


def encode_row(row: list[str], feature_indices: list[int], label_maps: dict[int, dict[str, int]]) -> list[float]:
    encoded = []
    for col_index in feature_indices:
        raw = row[col_index].strip()
        if col_index in label_maps:
            encoded.append(float(label_maps[col_index].get(raw, 0)))
        else:
            encoded.append(numeric_value(raw) or 0.0)
    return encoded


def normalize_row(row: list[float], mins: list[float], maxs: list[float]) -> list[float]:
    normalized = []
    for value, min_value, max_value in zip(row, mins, maxs):
        value_range = max_value - min_value
        if value_range < 1.0e-8:
            value_range = 1.0
        normalized.append((value - min_value) / value_range)
    return normalized


def write_batch(path: Path, rows: list[list[float]], labels: list[float], n_features: int) -> None:
    count = len(rows)
    with path.open("wb") as handle:
        handle.write(BATCH_MAGIC)
        handle.write(struct.pack("<II", count, n_features))

        for row in rows:
            handle.write(struct.pack(f"<{n_features}f", *row))

        handle.write(struct.pack(f"<{count}f", *labels))


def materialize(
    csv_path: Path,
    output_dir: Path,
    feature_indices: list[int],
    target_index: int,
    scan: dict,
    train_ratio: float,
    seed: int,
    batch_size: int,
    positive_threshold: float,
) -> tuple[int, int, int, int, int, int]:
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[list[float]] = []
    train_labels: list[float] = []
    test_rows: list[list[float]] = []
    test_labels: list[float] = []
    train_batch_index = 0
    test_batch_index = 0
    train_positive = 0
    train_negative = 0
    test_positive = 0
    test_negative = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        for row_index, row in enumerate(reader):
            if not row:
                continue

            features = encode_row(row, feature_indices, scan["label_maps"])
            features = normalize_row(features, scan["mins"], scan["maxs"])
            label = parse_label(row[target_index], positive_threshold)

            if train_row(row_index, train_ratio, seed):
                if label > 0.5:
                    train_positive += 1
                else:
                    train_negative += 1

                train_rows.append(features)
                train_labels.append(label)

                if len(train_rows) >= batch_size:
                    write_batch(train_dir / f"{train_batch_index}.bpbatch", train_rows, train_labels, len(feature_indices))
                    train_batch_index += 1
                    train_rows = []
                    train_labels = []
            else:
                if label > 0.5:
                    test_positive += 1
                else:
                    test_negative += 1

                test_rows.append(features)
                test_labels.append(label)

                if len(test_rows) >= batch_size:
                    write_batch(test_dir / f"{test_batch_index}.bpbatch", test_rows, test_labels, len(feature_indices))
                    test_batch_index += 1
                    test_rows = []
                    test_labels = []

    if train_rows:
        write_batch(train_dir / f"{train_batch_index}.bpbatch", train_rows, train_labels, len(feature_indices))
        train_batch_index += 1

    if test_rows:
        write_batch(test_dir / f"{test_batch_index}.bpbatch", test_rows, test_labels, len(feature_indices))
        test_batch_index += 1

    return train_batch_index, test_batch_index, train_positive, train_negative, test_positive, test_negative


def write_metadata(output_dir: Path, values: dict[str, object]) -> None:
    with (output_dir / "metadata.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara dataset normalizado em batches binarios para os projetos Elixir.")
    parser.add_argument("csv", type=Path, help="Arquivo CSV de entrada.")
    parser.add_argument("output", type=Path, help="Diretorio de saida do dataset preparado.")
    parser.add_argument("--target-column", default=None, help="Coluna alvo. Padrao: fraud_bool se existir, senao ultima coluna.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="Remove o diretorio de saida antes de gerar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output.resolve()

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser maior que zero")

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Diretorio de saida ja existe: {output_dir}. Use --force para sobrescrever.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    started = time.perf_counter()
    header = read_header(csv_path)
    target_index = detect_target(header, args.target_column)
    feature_indices = [idx for idx in range(len(header)) if idx != target_index]

    print("Primeira passada: detectando categorias, split e min/max...")
    scan_started = time.perf_counter()
    scan = scan_csv(csv_path, feature_indices, target_index, args.train_ratio, args.seed)
    print(f"scan_seconds={time.perf_counter() - scan_started:.3f}")

    print("Segunda passada: normalizando e gravando batches binarios...")
    materialize_started = time.perf_counter()
    train_batches, test_batches, train_positive, train_negative, test_positive, test_negative = materialize(
        csv_path,
        output_dir,
        feature_indices,
        target_index,
        scan,
        args.train_ratio,
        args.seed,
        args.batch_size,
        args.positive_threshold,
    )
    print(f"materialize_seconds={time.perf_counter() - materialize_started:.3f}")

    write_metadata(
        output_dir,
        {
            "format": "BPNORM1",
            "source": csv_path.name,
            "target_column": header[target_index],
            "n_features": len(feature_indices),
            "train_count": scan["train_count"],
            "test_count": scan["test_count"],
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "positive_threshold": args.positive_threshold,
            "train_positive": train_positive,
            "train_negative": train_negative,
            "test_positive": test_positive,
            "test_negative": test_negative,
            "train_batch_count": train_batches,
            "test_batch_count": test_batches,
        },
    )

    print(f"total_seconds={time.perf_counter() - started:.3f}")
    print(f"output={output_dir}")
    print(f"train_count={scan['train_count']} test_count={scan['test_count']}")
    print(f"train_positive={train_positive} train_negative={train_negative}")
    print(f"test_positive={test_positive} test_negative={test_negative}")
    print(f"train_batches={train_batches} test_batches={test_batches}")


if __name__ == "__main__":
    main()
