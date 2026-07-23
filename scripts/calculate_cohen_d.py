#!/usr/bin/env python3
"""Calcula o tamanho de efeito d de Cohen para o benchmark de performance."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def effect_classification(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "desprezivel"
    if magnitude < 0.5:
        return "pequeno"
    if magnitude < 0.8:
        return "medio"
    return "grande"


def cohen_d(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(left) < 2 or len(right) < 2:
        raise ValueError("cada implementacao precisa de pelo menos duas amostras")

    left_variance = statistics.variance(left)
    right_variance = statistics.variance(right)
    degrees_freedom = len(left) + len(right) - 2
    pooled_variance = (
        (len(left) - 1) * left_variance + (len(right) - 1) * right_variance
    ) / degrees_freedom
    pooled_standard_deviation = math.sqrt(pooled_variance)
    mean_difference = statistics.fmean(left) - statistics.fmean(right)

    if pooled_standard_deviation == 0.0:
        if mean_difference == 0.0:
            return 0.0, pooled_standard_deviation
        return math.copysign(math.inf, mean_difference), pooled_standard_deviation

    return mean_difference / pooled_standard_deviation, pooled_standard_deviation


def read_samples(path: Path) -> dict[tuple[str, str], list[float]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"implementation", "warmup", "metric", "milliseconds"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"colunas ausentes em {path}: {', '.join(sorted(missing))}")

        for row in reader:
            if row["warmup"] == "1":
                continue
            samples[(row["implementation"], row["metric"])].append(float(row["milliseconds"]))

    return samples


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def format_number(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.9f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula o d de Cohen entre duas implementacoes do benchmark."
    )
    parser.add_argument("--runs", type=Path, required=True, help="Arquivo performance_runs.csv")
    parser.add_argument("--output", type=Path, required=True, help="Diretorio de saida")
    parser.add_argument("--implementation-a", default="cuda")
    parser.add_argument("--implementation-b", default="polyhok")
    args = parser.parse_args()

    samples = read_samples(args.runs)
    metrics_a = {metric for implementation, metric in samples if implementation == args.implementation_a}
    metrics_b = {metric for implementation, metric in samples if implementation == args.implementation_b}
    metrics = sorted(metrics_a.intersection(metrics_b))
    if not metrics:
        raise ValueError(
            f"nenhuma metrica comum entre {args.implementation_a} e {args.implementation_b}"
        )

    rows: list[dict[str, object]] = []
    for metric in metrics:
        values_a = samples[(args.implementation_a, metric)]
        values_b = samples[(args.implementation_b, metric)]
        effect, pooled_sd = cohen_d(values_a, values_b)
        mean_a = statistics.fmean(values_a)
        mean_b = statistics.fmean(values_b)
        lower_mean = args.implementation_a if mean_a < mean_b else args.implementation_b
        if mean_a == mean_b:
            lower_mean = "iguais"

        rows.append(
            {
                "metric": metric,
                "unit": "ms",
                "implementation_a": args.implementation_a,
                "implementation_b": args.implementation_b,
                "samples_a": len(values_a),
                "samples_b": len(values_b),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "pooled_standard_deviation": pooled_sd,
                "mean_difference_a_minus_b": mean_a - mean_b,
                "cohen_d": effect,
                "absolute_cohen_d": abs(effect),
                "classification": effect_classification(effect),
                "lower_mean": lower_mean,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "performance_cohen_d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    text_path = args.output / "performance_cohen_d.txt"
    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("d de Cohen - tamanho de efeito entre implementacoes\n\n")
        handle.write("Formula: d = (media_a - media_b) / desvio_padrao_combinado.\n")
        handle.write("O sinal indica a direcao; para tempos, d negativo favorece a implementacao A.\n")
        handle.write("Classificacao de |d|: <0.2 desprezivel, <0.5 pequeno, <0.8 medio, >=0.8 grande.\n")
        handle.write("Cohen d mede magnitude do efeito e nao substitui o teste de significancia de Welch.\n\n")
        for row in rows:
            handle.write(
                f"{row['metric']}: n={row['samples_a']}/{row['samples_b']} "
                f"media={row['mean_a']:.6f}/{row['mean_b']:.6f} ms "
                f"d={format_number(float(row['cohen_d']))} "
                f"|d|={format_number(float(row['absolute_cohen_d']))} "
                f"efeito={row['classification']} menor_media={row['lower_mean']}\n"
            )

    tex_path = args.output / "performance_cohen_d_table.tex"
    with tex_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\\begin{tabular}{lrrrrl}\n")
        handle.write("\\hline\n")
        handle.write("Metrica & $n_A$ & $n_B$ & $d$ & $|d|$ & Magnitude \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            effect = float(row["cohen_d"])
            absolute_effect = float(row["absolute_cohen_d"])
            effect_tex = "$\\infty$" if math.isinf(effect) and effect > 0 else "$-\\infty$" if math.isinf(effect) else f"{effect:.3f}"
            absolute_tex = "$\\infty$" if math.isinf(absolute_effect) else f"{absolute_effect:.3f}"
            handle.write(
                f"{tex_escape(row['metric'])} & {row['samples_a']} & {row['samples_b']} & "
                f"{effect_tex} & {absolute_tex} & {tex_escape(row['classification'])} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")

    print(f"CSV: {csv_path}")
    print(f"Texto: {text_path}")
    print(f"LaTeX: {tex_path}")
    print()
    print(text_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
