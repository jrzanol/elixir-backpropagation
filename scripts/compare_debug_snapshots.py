#!/usr/bin/env python3
"""Compara snapshots da primeira atualizacao entre implementacoes."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np


SNAPSHOT_PATTERN = re.compile(
    r"\[DEBUG_SNAPSHOT\] impl=(\w+) epoch=(\d+)"
    r" weights=(\[[^]]*\]) biases=(\[[^]]*\])"
    r" grad_w=(\[[^]]*\]) grad_b=(\[[^]]*\])"
)


def parse_values(value: str) -> np.ndarray:
    content = value[1:-1]
    if not content:
        return np.empty(0, dtype=np.float64)
    return np.fromstring(content, sep=",", dtype=np.float64)


def read_snapshots(path: Path) -> dict[int, dict[str, np.ndarray | str]]:
    snapshots: dict[int, dict[str, np.ndarray | str]] = {}
    for match in SNAPSHOT_PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
        implementation, epoch, weights, biases, grad_w, grad_b = match.groups()
        snapshots[int(epoch)] = {
            "implementation": implementation,
            "weights": parse_values(weights),
            "biases": parse_values(biases),
            "grad_w": parse_values(grad_w),
            "grad_b": parse_values(grad_b),
        }

    if set(snapshots) != {0, 1}:
        found = ", ".join(str(epoch) for epoch in sorted(snapshots)) or "nenhum"
        raise ValueError(
            f"{path}: snapshots esperados epoch=0 e epoch=1; encontrados: {found}"
        )
    return snapshots


def compare(
    reference: dict[int, dict[str, np.ndarray | str]],
    candidate: dict[int, dict[str, np.ndarray | str]],
) -> bool:
    reference_name = str(reference[0]["implementation"])
    candidate_name = str(candidate[0]["implementation"])
    passed = True

    tolerances = {
        (0, "weights"): (1.0e-7, 1.0e-7),
        (0, "biases"): (1.0e-7, 1.0e-7),
        (1, "weights"): (2.0e-5, 2.0e-5),
        (1, "biases"): (2.0e-5, 2.0e-5),
        (1, "grad_w"): (2.0e-2, 2.0e-3),
        (1, "grad_b"): (2.0e-2, 2.0e-3),
    }

    print(f"\n{reference_name} x {candidate_name}")
    for (epoch, field), (absolute_tolerance, relative_tolerance) in tolerances.items():
        expected = reference[epoch][field]
        actual = candidate[epoch][field]
        if expected.shape != actual.shape:
            print(f"  FALHA epoch={epoch} {field}: tamanhos {expected.size} e {actual.size}")
            passed = False
            continue

        difference = np.abs(expected - actual)
        max_absolute = float(difference.max()) if difference.size else 0.0
        max_index = int(difference.argmax()) if difference.size else 0
        expected_at_max = float(expected[max_index]) if difference.size else 0.0
        actual_at_max = float(actual[max_index]) if difference.size else 0.0
        denominator = np.maximum(np.abs(expected), 1.0e-12)
        max_relative = float((difference / denominator).max()) if difference.size else 0.0
        matches = np.allclose(
            expected,
            actual,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        status = "OK" if matches else "FALHA"
        print(
            f"  {status} epoch={epoch} {field}: "
            f"erro_abs_max={max_absolute:.9g} erro_rel_max={max_relative:.9g} "
            f"indice={max_index} esperado={expected_at_max:.9g} atual={actual_at_max:.9g}"
        )
        passed &= bool(matches)

    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    if len(args.logs) < 2:
        parser.error("informe pelo menos dois logs")

    snapshots = [read_snapshots(path) for path in args.logs]
    results = [compare(reference, candidate) for reference, candidate in itertools.combinations(snapshots, 2)]
    passed = all(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
