#!/usr/bin/env python3
"""
Executa uma baseline scikit-learn usando o mesmo dataset preparado do projeto.

Objetivo: comparar tecnica e fluxo com as implementacoes CUDA/NIF e PolyHok:
- mesmo split ja preparado em train/test;
- mesma topologia padrao [n_features, max(8, 2*n), max(4, n), 1];
- ReLU nas camadas ocultas;
- saida binaria logistica;
- SGD com taxa constante;
- treino por batches na ordem dos arquivos .bpbatch.

Requerimentos:
numpy
scikit-learn
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier


BATCH_MAGIC = b"BPBATCH1"
MASK64 = 0xFFFFFFFFFFFFFFFF
XORSHIFT_MUL = 2685821657736338717


@dataclass
class Batch:
    features: np.ndarray
    labels: np.ndarray


@dataclass
class Metrics:
    total: int = 0
    correct: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn_val: int = 0

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        for pred, truth in zip(predictions.astype(np.int32), labels.astype(np.int32)):
            self.total += 1
            self.correct += int(pred == truth)

            if truth == 1 and pred == 1:
                self.tp += 1
            elif truth == 0 and pred == 0:
                self.tn += 1
            elif truth == 0 and pred == 1:
                self.fp += 1
            elif truth == 1 and pred == 0:
                self.fn_val += 1

    def summary(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0.0
        recall = self.tp / (self.tp + self.fn_val) if self.tp + self.fn_val > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = self.correct / self.total if self.total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn_val": self.fn_val,
        }


def read_metadata(dataset: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in (dataset / "metadata.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        metadata[key] = value
    return metadata


def batch_paths(dataset: Path, kind: str) -> list[Path]:
    return sorted((dataset / kind).glob("*.bpbatch"), key=lambda path: int(path.stem))


def load_batch(path: Path) -> Batch:
    data = path.read_bytes()
    if data[:8] != BATCH_MAGIC:
        raise ValueError(f"batch invalido: {path}")

    count, n_features = struct.unpack_from("<II", data, 8)
    offset = 16
    feature_count = count * n_features
    feature_bytes = feature_count * 4
    label_bytes = count * 4

    features = np.frombuffer(data, dtype="<f4", count=feature_count, offset=offset)
    offset += feature_bytes
    labels = np.frombuffer(data, dtype="<f4", count=count, offset=offset)

    if len(data) != 16 + feature_bytes + label_bytes:
        raise ValueError(f"tamanho de batch invalido: {path}")

    return Batch(
        features=features.reshape((count, n_features)).astype(np.float32, copy=True),
        labels=labels.astype(np.int32, copy=True),
    )


def validate_metadata(metadata: dict[str, str], train_ratio: float, seed: int, batch_size: int) -> None:
    if metadata.get("format") != "BPNORM1":
        raise ValueError("dataset preparado invalido: formato nao suportado")
    if abs(float(metadata["train_ratio"]) - train_ratio) > 1.0e-12:
        raise ValueError(f"train_ratio={train_ratio} nao corresponde ao dataset preparado ({metadata['train_ratio']})")
    if int(metadata["seed"]) != seed:
        raise ValueError(f"seed={seed} nao corresponde ao dataset preparado ({metadata['seed']})")
    if int(metadata["batch_size"]) != batch_size:
        raise ValueError(f"batch_size={batch_size} nao corresponde ao dataset preparado ({metadata['batch_size']})")


def evaluate(model: MLPClassifier, paths: list[Path]) -> Metrics:
    metrics = Metrics()
    for path in paths:
        batch = load_batch(path)
        predictions = model.predict(batch.features)
        metrics.update(predictions, batch.labels)
    return metrics


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
    return new_state, top53 * (1.0 / float(1 << 53))


def initialize_like_cuda_polyhok(model: MLPClassifier, layers: list[int], seed: int) -> None:
    state = seed_xor_shift(seed)

    for layer_index, (prev_size, curr_size) in enumerate(zip(layers[:-1], layers[1:])):
        limit = np.sqrt(6.0 / float(prev_size))
        weights = np.empty((prev_size, curr_size), dtype=np.float64)

        for i in range(prev_size):
            for j in range(curr_size):
                state, u = next_float(state)
                weights[i, j] = (u * 2.0 - 1.0) * limit

        model.coefs_[layer_index][:] = weights
        model.intercepts_[layer_index][:] = 0.0


def print_results(train_metrics: Metrics, test_metrics: Metrics) -> None:
    train = train_metrics.summary()
    test = test_metrics.summary()

    print("\n--- Resultados finais ---")
    print(f"Acuracia (treino) : {round(train['accuracy'] * 100.0, 2)}%")
    print(f"Acuracia (teste)  : {round(test['accuracy'] * 100.0, 2)}%")
    print(f"Precisao : {round(test['precision'], 4)}")
    print(f"Recall   : {round(test['recall'], 4)}")
    print(f"F1-Score : {round(test['f1'], 4)}")
    print("Matriz:")
    print(f"  Real 0: TN={test['tn']}  FP={test['fp']}")
    print(f"  Real 1: FN={test['fn_val']}  TP={test['tp']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline scikit-learn para o dataset preparado do backprop.")
    parser.add_argument("--dataset", type=Path, default=Path("scripts/prepared_dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learn-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = read_metadata(args.dataset)
    validate_metadata(metadata, args.train_ratio, args.seed, args.batch_size)

    n_features = int(metadata["n_features"])
    train_count = int(metadata["train_count"])
    test_count = int(metadata["test_count"])
    hidden_layers = (256, 128)

    train_paths = batch_paths(args.dataset, "train")
    test_paths = batch_paths(args.dataset, "test")

    print("Carregando dataset preparado...")
    print(f"Amostras: {train_count + test_count}, Features: {n_features}")
    print(f"Treino: {train_count}, Teste: {test_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches treino: {len(train_paths)}, teste: {len(test_paths)}")
    print(f"Topologia: [{n_features}, {hidden_layers[0]}, {hidden_layers[1]}, 1]")
    print(f"\nTreinando {args.epochs} epochs com lr={args.learn_rate}...")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="sgd",
        alpha=0.0,
        batch_size=args.batch_size,
        learning_rate="constant",
        learning_rate_init=args.learn_rate,
        max_iter=1,
        shuffle=False,
        random_state=args.seed,
        momentum=0.0,
        nesterovs_momentum=False,
        tol=0.0,
        warm_start=True,
    )

    classes = np.array([0, 1], dtype=np.int32)
    first_batch = load_batch(train_paths[0])
    model.batch_size = len(first_batch.labels)
    model.partial_fit(first_batch.features, first_batch.labels, classes=classes)
    initialize_like_cuda_polyhok(model, [n_features, *hidden_layers, 1], args.seed)

    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        for path in train_paths:
            batch = load_batch(path)
            model.batch_size = len(batch.labels)
            model.partial_fit(batch.features, batch.labels, classes=classes)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{args.epochs} concluida")

    train_metrics = evaluate(model, train_paths)
    test_metrics = evaluate(model, test_paths)
    print_results(train_metrics, test_metrics)
    print(f"Tempo total: {round(time.perf_counter() - started, 3)}s")


if __name__ == "__main__":
    main()
