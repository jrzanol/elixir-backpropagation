#!/usr/bin/env python3
"""Compara o caso funcional entre CUDA, PolyHok e Python."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SNAPSHOT = re.compile(
    r"\[DEBUG_SNAPSHOT\] impl=(\w+) epoch=(\d+)"
    r" weights=(\[[^]]*\]) biases=(\[[^]]*\])"
    r" grad_w=(\[[^]]*\]) grad_b=(\[[^]]*\])"
)
FUNCTIONAL = re.compile(
    r"\[FUNCTIONAL\] impl=(\w+) probability_before=([-+0-9.eE]+)"
    r" prediction_before=(\d+) probability_after=([-+0-9.eE]+) prediction_after=(\d+)"
)
FUNCTIONAL_MODEL = re.compile(
    r"\[FUNCTIONAL_MODEL\] impl=(\w+) weights=(\[[^]]*\]) biases=(\[[^]]*\])"
)


def numbers(text: str) -> list[float]:
    content = text[1:-1]
    return [] if not content else [float(value) for value in content.split(",")]


def read_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    snapshots: dict[int, dict[str, list[float]]] = {}
    implementation = ""
    for match in SNAPSHOT.finditer(text):
        implementation, epoch, weights, biases, grad_w, grad_b = match.groups()
        snapshots[int(epoch)] = {
            "weights": numbers(weights),
            "biases": numbers(biases),
            "grad_w": numbers(grad_w),
            "grad_b": numbers(grad_b),
        }
    functional = FUNCTIONAL.search(text)
    if set(snapshots) != {0, 1} or functional is None:
        raise ValueError(f"log funcional incompleto: {path}")
    impl, probability_before, prediction_before, probability_after, prediction_after = functional.groups()
    if implementation != impl:
        raise ValueError(f"implementacoes inconsistentes em {path}")
    model_state = FUNCTIONAL_MODEL.search(text)
    if model_state is not None:
        state_impl, weights, biases = model_state.groups()
        if state_impl != impl:
            raise ValueError(f"estado funcional inconsistente em {path}")
        snapshots[1]["weights"] = numbers(weights)
        snapshots[1]["biases"] = numbers(biases)

    return {
        "implementation": impl,
        "snapshots": snapshots,
        "probability_before": float(probability_before),
        "prediction_before": int(prediction_before),
        "probability_after": float(probability_after),
        "prediction_after": int(prediction_after),
    }


def max_difference(expected: list[float], actual: list[float]) -> float:
    if len(expected) != len(actual):
        return float("inf")
    return max((abs(left - right) for left, right in zip(expected, actual)), default=0.0)


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    results = [read_log(path) for path in args.logs]
    reference = next(result for result in results if result["implementation"] == "python")
    rows: list[dict[str, object]] = []
    passed = True

    for result in results:
        name = str(result["implementation"])
        comparisons = {
            "forward_probability": abs(float(reference["probability_before"]) - float(result["probability_before"])),
            "probability_after_update": abs(float(reference["probability_after"]) - float(result["probability_after"])),
            "updated_weights": max_difference(reference["snapshots"][1]["weights"], result["snapshots"][1]["weights"]),
            "updated_biases": max_difference(reference["snapshots"][1]["biases"], result["snapshots"][1]["biases"]),
            "weight_gradients": max_difference(reference["snapshots"][1]["grad_w"], result["snapshots"][1]["grad_w"]),
            "deltas_via_bias_gradients": max_difference(reference["snapshots"][1]["grad_b"], result["snapshots"][1]["grad_b"]),
        }
        predictions_match = (
            reference["prediction_before"] == result["prediction_before"]
            and reference["prediction_after"] == result["prediction_after"]
        )
        for comparison, difference in comparisons.items():
            tolerance = 2.0e-5
            ok = difference <= tolerance
            passed &= ok
            rows.append({
                "implementation": name,
                "comparison": comparison,
                "max_absolute_error": f"{difference:.9g}",
                "tolerance": f"{tolerance:.9g}",
                "status": "OK" if ok else "FAIL",
            })
        passed &= predictions_match
        rows.append({
            "implementation": name,
            "comparison": "predictions",
            "max_absolute_error": "0" if predictions_match else "1",
            "tolerance": "0",
            "status": "OK" if predictions_match else "FAIL",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "functional_correctness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    tex_path = args.output / "functional_correctness.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrl}\n\\hline\nImplementacao & Comparacao & Erro maximo & Status \\\\\n\\hline\n")
        for row in rows:
            handle.write(
                f"{row['implementation']} & {tex_escape(row['comparison'])} & "
                f"{row['max_absolute_error']} & {row['status']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    for row in rows:
        print(
            f"{row['implementation']:8s} {row['comparison']:28s} "
            f"erro={row['max_absolute_error']} {row['status']}"
        )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
