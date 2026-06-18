#!/usr/bin/env python3
"""Compara snapshots de varias epochs entre implementacoes."""

from __future__ import annotations

import argparse
import csv
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


FIELDS_BY_EPOCH = {
    0: ("weights", "biases"),
}


def compared_fields(epoch: int) -> tuple[str, ...]:
    return FIELDS_BY_EPOCH.get(epoch, ("weights", "biases", "grad_w", "grad_b"))


def tolerances(epoch: int, field: str) -> tuple[float, float]:
    if epoch == 0:
        return 1.0e-7, 1.0e-7
    if field in ("weights", "biases"):
        return 2.0e-4, 2.0e-4
    return 2.0e-1, 2.0e-3


def read_snapshots(path: Path, expected_epochs: set[int] | None) -> dict[int, dict[str, np.ndarray | str]]:
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

    if expected_epochs is not None:
        missing = sorted(expected_epochs - set(snapshots))
        if missing:
            found = ", ".join(str(epoch) for epoch in sorted(snapshots)) or "nenhum"
            expected = ", ".join(str(epoch) for epoch in sorted(expected_epochs))
            raise ValueError(f"{path}: snapshots esperados epoch={expected}; encontrados: {found}")

    return snapshots


def compare(
    reference: dict[int, dict[str, np.ndarray | str]],
    candidate: dict[int, dict[str, np.ndarray | str]],
    epochs: list[int],
) -> list[dict[str, object]]:
    reference_name = str(reference[0]["implementation"])
    candidate_name = str(candidate[0]["implementation"])
    rows: list[dict[str, object]] = []

    print(f"\n{reference_name} x {candidate_name}")
    print(
        "  "
        + "epoch".ljust(8)
        + "campo".ljust(10)
        + "status".ljust(8)
        + "erro_abs_max".ljust(18)
        + "erro_rel_max".ljust(18)
        + "indice".ljust(10)
        + "referencia".ljust(16)
        + "candidato"
    )

    for epoch in epochs:
        for field in compared_fields(epoch):
            absolute_tolerance, relative_tolerance = tolerances(epoch, field)
            row: dict[str, object] = {
                "reference": reference_name,
                "candidate": candidate_name,
                "epoch": epoch,
                "field": field,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
            }

            if epoch not in reference or epoch not in candidate:
                row.update(
                    {
                        "status": "FALHA",
                        "max_absolute_error": "",
                        "max_relative_error": "",
                        "index": "",
                        "reference_value": "",
                        "candidate_value": "",
                        "message": "snapshot ausente",
                    }
                )
                rows.append(row)
                print(f"  {epoch:<8}{field:<10}{'FALHA':<8}snapshot ausente")
                continue

            expected = reference[epoch][field]
            actual = candidate[epoch][field]
            if expected.shape != actual.shape:
                row.update(
                    {
                        "status": "FALHA",
                        "max_absolute_error": "",
                        "max_relative_error": "",
                        "index": "",
                        "reference_value": "",
                        "candidate_value": "",
                        "message": f"tamanhos {expected.size} e {actual.size}",
                    }
                )
                rows.append(row)
                print(f"  {epoch:<8}{field:<10}{'FALHA':<8}tamanhos {expected.size} e {actual.size}")
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
            row.update(
                {
                    "status": status,
                    "max_absolute_error": max_absolute,
                    "max_relative_error": max_relative,
                    "index": max_index,
                    "reference_value": expected_at_max,
                    "candidate_value": actual_at_max,
                    "message": "",
                }
            )
            rows.append(row)
            print(
                f"  {epoch:<8}{field:<10}{status:<8}"
                f"{max_absolute:<18.9g}{max_relative:<18.9g}"
                f"{max_index:<10}{expected_at_max:<16.9g}{actual_at_max:.9g}"
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "reference",
        "candidate",
        "epoch",
        "field",
        "status",
        "max_absolute_error",
        "max_relative_error",
        "index",
        "reference_value",
        "candidate_value",
        "absolute_tolerance",
        "relative_tolerance",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llrlrr}",
        "\\hline",
        "Referencia & Candidato & Epoca & Campo & Erro abs. max. & Erro rel. max. \\\\",
        "\\hline",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["reference"]),
                    tex_escape(row["candidate"]),
                    tex_escape(row["epoch"]),
                    tex_escape(row["field"]),
                    tex_escape(fmt(row["max_absolute_error"])),
                    tex_escape(fmt(row["max_relative_error"])),
                ]
            )
            + " \\\\"
        )

    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    if len(args.logs) < 2:
        parser.error("informe pelo menos dois logs")

    expected_epochs = set(range(0, args.epochs + 1)) if args.epochs is not None else None
    snapshots = [read_snapshots(path, expected_epochs) for path in args.logs]
    epochs = (
        list(range(0, args.epochs + 1))
        if args.epochs is not None
        else sorted(set.intersection(*(set(snapshot) for snapshot in snapshots)))
    )
    rows = [
        row
        for reference, candidate in itertools.combinations(snapshots, 2)
        for row in compare(reference, candidate, epochs)
    ]

    if args.csv is not None:
        write_csv(args.csv, rows)
    if args.tex is not None:
        write_tex(args.tex, rows)

    passed = all(row["status"] == "OK" for row in rows)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
