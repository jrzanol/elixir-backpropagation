#!/usr/bin/env python3
"""Consolida equivalencia final e gera CSV/LaTeX."""

from __future__ import annotations

import argparse
import array
import csv
import math
import sys
from pathlib import Path


PROBABILITY_ABS_TOLERANCE = 2.0e-4
CLASSIFICATION_MISMATCH_RATE_TOLERANCE = 1.0e-5
METRIC_ABS_TOLERANCE = CLASSIFICATION_MISMATCH_RATE_TOLERANCE


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--polyhok", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cuda_config = read_single_row(args.cuda / "config.csv")
    polyhok_config = read_single_row(args.polyhok / "config.csv")
    compared_config = ["dataset", "train_ratio", "seed", "batch_size", "epochs", "learning_rate", "topology"]
    config_equal = all(cuda_config[key] == polyhok_config[key] for key in compared_config)

    metric_rows: list[dict[str, object]] = []
    passed = config_equal
    for implementation, directory in [("cuda", args.cuda), ("polyhok", args.polyhok)]:
        for row in read_rows(directory / "metrics.csv"):
            metric_rows.append({"implementation": implementation, **row})

    comparisons: list[dict[str, object]] = []
    for split in ("train", "test"):
        cuda_predictions = (args.cuda / f"{split}_predictions.bin").read_bytes()
        polyhok_predictions = (args.polyhok / f"{split}_predictions.bin").read_bytes()
        same_length = len(cuda_predictions) == len(polyhok_predictions)
        mismatches = (
            sum(left != right for left, right in zip(cuda_predictions, polyhok_predictions))
            if same_length
            else max(len(cuda_predictions), len(polyhok_predictions))
        )
        prediction_tolerance = (
            max(1, math.ceil(len(cuda_predictions) * CLASSIFICATION_MISMATCH_RATE_TOLERANCE))
            if same_length
            else 0
        )
        mismatch_rate = mismatches / len(cuda_predictions) if cuda_predictions else 0.0

        cuda_probabilities = read_floats(args.cuda / f"{split}_probabilities.f32")
        polyhok_probabilities = read_floats(args.polyhok / f"{split}_probabilities.f32")
        probability_length_equal = len(cuda_probabilities) == len(polyhok_probabilities)
        differences = [
            abs(left - right) for left, right in zip(cuda_probabilities, polyhok_probabilities)
        ]
        max_probability_error = max(differences, default=0.0)
        mean_probability_error = sum(differences) / len(differences) if differences else 0.0
        predictions_ok = same_length and mismatches <= prediction_tolerance
        probabilities_ok = probability_length_equal and max_probability_error <= PROBABILITY_ABS_TOLERANCE
        split_ok = predictions_ok and probabilities_ok
        passed &= split_ok
        comparisons.append({
            "category": f"{split}_predictions",
            "value": mismatches,
            "tolerance": prediction_tolerance,
            "status": "OK" if predictions_ok else "FAIL",
        })
        comparisons.append({
            "category": f"{split}_prediction_mismatch_rate",
            "value": f"{mismatch_rate:.9g}",
            "tolerance": f"{CLASSIFICATION_MISMATCH_RATE_TOLERANCE:.9g}",
            "status": "OK" if predictions_ok else "FAIL",
        })
        comparisons.append({
            "category": f"{split}_probability_max_abs_error",
            "value": f"{max_probability_error:.9g}",
            "tolerance": f"{PROBABILITY_ABS_TOLERANCE:.9g}",
            "status": "OK" if probabilities_ok else "FAIL",
        })
        comparisons.append({
            "category": f"{split}_probability_mean_abs_error",
            "value": f"{mean_probability_error:.9g}",
            "tolerance": "informative",
            "status": "OK",
        })

    cuda_metrics = {(row["split"]): row for row in read_rows(args.cuda / "metrics.csv")}
    polyhok_metrics = {(row["split"]): row for row in read_rows(args.polyhok / "metrics.csv")}
    for split in ("train", "test"):
        for field in ("accuracy", "precision", "recall", "f1"):
            difference = abs(float(cuda_metrics[split][field]) - float(polyhok_metrics[split][field]))
            ok = difference <= METRIC_ABS_TOLERANCE
            passed &= ok
            comparisons.append({
                "category": f"{split}_{field}_abs_error",
                "value": f"{difference:.9g}",
                "tolerance": f"{METRIC_ABS_TOLERANCE:.9g}",
                "status": "OK" if ok else "FAIL",
            })
        confusion_tolerance = max(
            1,
            math.ceil(
                int(cuda_metrics[split]["total"]) * CLASSIFICATION_MISMATCH_RATE_TOLERANCE
            ),
        )
        for field in ("tn", "fp", "fn", "tp"):
            difference = abs(int(cuda_metrics[split][field]) - int(polyhok_metrics[split][field]))
            ok = difference <= confusion_tolerance
            passed &= ok
            comparisons.append({
                "category": f"{split}_{field}_difference",
                "value": difference,
                "tolerance": confusion_tolerance,
                "status": "OK" if ok else "FAIL",
            })

    cuda_history = read_rows(args.cuda / "epoch_error.csv")
    polyhok_history = read_rows(args.polyhok / "epoch_error.csv")
    history_rows: list[dict[str, object]] = []
    if len(cuda_history) != len(polyhok_history):
        passed = False
    for cuda_row, polyhok_row in zip(cuda_history, polyhok_history):
        difference = abs(float(cuda_row["bce"]) - float(polyhok_row["bce"]))
        ok = difference <= 2.0e-4
        passed &= ok
        history_rows.append({
            "epoch": cuda_row["epoch"],
            "cuda_bce": cuda_row["bce"],
            "polyhok_bce": polyhok_row["bce"],
            "absolute_error": f"{difference:.9g}",
            "status": "OK" if ok else "FAIL",
        })

    comparisons.insert(0, {
        "category": "configuration_equal",
        "value": 1 if config_equal else 0,
        "tolerance": 1,
        "status": "OK" if config_equal else "FAIL",
    })

    for filename, rows in [
        ("equivalence_summary.csv", comparisons),
        ("equivalence_metrics.csv", metric_rows),
        ("equivalence_epoch_error.csv", history_rows),
    ]:
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with (args.output / "equivalence_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrl}\n\\hline\nComparacao & Valor & Tolerancia & Status \\\\\n\\hline\n")
        for row in comparisons:
            handle.write(
                f"{tex_escape(row['category'])} & {row['value']} & {row['tolerance']} & {row['status']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    for row in comparisons:
        print(f"{row['category']:42s} valor={row['value']} {row['status']}")
    if history_rows:
        print(f"erro BCE maximo entre epocas: {max(float(row['absolute_error']) for row in history_rows):.9g}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
