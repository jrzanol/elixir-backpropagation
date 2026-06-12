#!/usr/bin/env python3
"""Baseline PyTorch/CUDA para o mesmo fluxo das implementacoes do projeto."""

from __future__ import annotations

import argparse
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


BATCH_MAGIC = b"BPBATCH1"
MASK64 = 0xFFFFFFFFFFFFFFFF
XORSHIFT_MUL = 2685821657736338717


@dataclass
class Batch:
    features: torch.Tensor
    labels: torch.Tensor


@dataclass
class Metrics:
    total: int = 0
    correct: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn_val: int = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = predictions.to(torch.int32)
        labels = labels.to(torch.int32)
        self.total += labels.numel()
        self.correct += int((predictions == labels).sum())
        self.tp += int(((labels == 1) & (predictions == 1)).sum())
        self.tn += int(((labels == 0) & (predictions == 0)).sum())
        self.fp += int(((labels == 0) & (predictions == 1)).sum())
        self.fn_val += int(((labels == 1) & (predictions == 0)).sum())

    def summary(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn_val) if self.tp + self.fn_val else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = self.correct / self.total if self.total else 0.0
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


class MLP(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(input_size, output_size, bias=True))
            if index < len(layers) - 2:
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def read_metadata(dataset: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in (dataset / "metadata.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
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
    feature_count = count * n_features
    feature_bytes = feature_count * 4
    expected_size = 16 + feature_bytes + count * 4
    if len(data) != expected_size:
        raise ValueError(f"tamanho de batch invalido: {path}")

    features = np.frombuffer(data, dtype="<f4", count=feature_count, offset=16).copy()
    labels = np.frombuffer(data, dtype="<f4", count=count, offset=16 + feature_bytes).copy()

    return Batch(
        features=torch.from_numpy(features.reshape(count, n_features)).pin_memory(),
        labels=torch.from_numpy(labels).pin_memory(),
    )


def prefetched_batches(paths: list[Path]):
    if not paths:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(load_batch, paths[0])
        for path in paths[1:]:
            batch = pending.result()
            pending = executor.submit(load_batch, path)
            yield batch
        yield pending.result()


def next_float(state: int) -> tuple[int, float]:
    x = state & MASK64
    x ^= x >> 12
    x ^= (x << 25) & MASK64
    x ^= x >> 27
    x &= MASK64
    value = (x * XORSHIFT_MUL) & MASK64
    return x, (value >> 11) * (1.0 / float(1 << 53))


def initialize_like_cuda_polyhok(model: MLP, layers: list[int], seed: int) -> None:
    state = 1 if seed == 0 else seed & MASK64
    linear_layers = [module for module in model.network if isinstance(module, nn.Linear)]

    with torch.no_grad():
        for linear, (input_size, output_size) in zip(linear_layers, zip(layers[:-1], layers[1:])):
            limit = np.sqrt(6.0 / float(input_size))
            weights = np.empty((input_size, output_size), dtype=np.float32)
            for index in range(weights.size):
                state, value = next_float(state)
                weights.flat[index] = (value * 2.0 - 1.0) * limit

            linear.weight.copy_(torch.from_numpy(weights.T))
            linear.bias.zero_()


def debug_value_count() -> int:
    try:
        value = int(os.environ.get("BACKPROP_DEBUG_VALUES", "8"))
        return value if value > 0 else 8
    except ValueError:
        return 8


def sampled_parameters(model: MLP, gradients: bool, batch_count: int) -> tuple[list[float], list[float]]:
    weights: list[float] = []
    biases: list[float] = []
    count = debug_value_count()

    for linear in (module for module in model.network if isinstance(module, nn.Linear)):
        if gradients:
            weight_tensor = linear.weight.grad.T * batch_count
            bias_tensor = linear.bias.grad * batch_count
        else:
            weight_tensor = linear.weight.detach().T
            bias_tensor = linear.bias.detach()

        weights.extend(weight_tensor.reshape(-1)[: max(0, count - len(weights))].tolist())
        biases.extend(bias_tensor.reshape(-1)[: max(0, count - len(biases))].tolist())

        if len(weights) >= count and len(biases) >= count:
            break

    return weights[:count], biases[:count]


def format_debug_values(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.9f}" for value in values) + "]"


def print_debug_snapshot(model: MLP, epoch: int, batch_count: int, include_gradients: bool) -> None:
    if os.environ.get("BACKPROP_DEBUG") != "1":
        return

    weights, biases = sampled_parameters(model, False, batch_count)
    if include_gradients:
        grad_w, grad_b = sampled_parameters(model, True, batch_count)
    else:
        grad_w, grad_b = [], []

    print(
        f"[DEBUG_SNAPSHOT] impl=pytorch epoch={epoch}"
        f" weights={format_debug_values(weights)}"
        f" biases={format_debug_values(biases)}"
        f" grad_w={format_debug_values(grad_w)}"
        f" grad_b={format_debug_values(grad_b)}"
    )


def evaluate(model: MLP, paths: list[Path], device: torch.device) -> Metrics:
    metrics = Metrics()
    model.eval()
    with torch.inference_mode():
        for batch in prefetched_batches(paths):
            features = batch.features.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            predictions = (model(features) >= 0.0).to(torch.int32)
            metrics.update(predictions, labels)
    return metrics


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


def validate(metadata: dict[str, str], train_ratio: float, seed: int, batch_size: int) -> None:
    if metadata.get("format") != "BPNORM1":
        raise ValueError("dataset preparado invalido")
    if abs(float(metadata["train_ratio"]) - train_ratio) > 1.0e-12:
        raise ValueError("train_ratio nao corresponde ao dataset preparado")
    if int(metadata["seed"]) != seed:
        raise ValueError("seed nao corresponde ao dataset preparado")
    if int(metadata["batch_size"]) != batch_size:
        raise ValueError("batch_size nao corresponde ao dataset preparado")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline PyTorch usando CUDA.")
    parser.add_argument("--dataset", type=Path, default=Path("scripts/prepared_dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learn-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch nao encontrou uma GPU CUDA")

    metadata = read_metadata(args.dataset)
    validate(metadata, args.train_ratio, args.seed, args.batch_size)
    n_features = int(metadata["n_features"])
    train_count = int(metadata["train_count"])
    test_count = int(metadata["test_count"])
    layers = [n_features, 256, 128, 1]
    train_paths = batch_paths(args.dataset, "train")
    test_paths = batch_paths(args.dataset, "test")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    model = MLP(layers)
    initialize_like_cuda_polyhok(model, layers, args.seed)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learn_rate, momentum=0.0)
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")

    print("Carregando dataset preparado...")
    print(f"Amostras: {train_count + test_count}, Features: {n_features}")
    print(f"Treino: {train_count}, Teste: {test_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches treino: {len(train_paths)}, teste: {len(test_paths)}")
    print(f"Topologia: {layers}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"\nTreinando {args.epochs} epochs com lr={args.learn_rate}...")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    debug_snapshot_printed = False
    for epoch in range(1, args.epochs + 1):
        for batch in prefetched_batches(train_paths):
            features = batch.features.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            if not debug_snapshot_printed:
                print_debug_snapshot(model, 0, batch.labels.numel(), False)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            if not debug_snapshot_printed:
                print_debug_snapshot(model, 1, batch.labels.numel(), True)
                debug_snapshot_printed = os.environ.get("BACKPROP_DEBUG") == "1"

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{args.epochs} concluida")

    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    train_metrics = evaluate(model, train_paths, device)
    test_metrics = evaluate(model, test_paths, device)
    torch.cuda.synchronize(device)

    print_results(train_metrics, test_metrics)
    print(f"Tempo treino: {train_seconds:.3f}s")
    print(f"Pico de memoria PyTorch: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
