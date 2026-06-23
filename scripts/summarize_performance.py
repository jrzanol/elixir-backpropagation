#!/usr/bin/env python3
"""Consolida execucoes repetidas do benchmark em CSV e LaTeX."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


WELCH_METRICS = {
    "application_total",
    "compilation",
    "epoch_average",
    "prediction_total",
    "training_total",
}


def beta_fraction(a: float, b: float, x: float) -> float:
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * beta_fraction(a, b, x) / a
    return 1.0 - bt * beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, degrees_freedom: float) -> float:
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive")
    if t_value == 0.0:
        return 0.5

    x = degrees_freedom / (degrees_freedom + t_value * t_value)
    beta = regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)
    if t_value > 0.0:
        return 1.0 - 0.5 * beta
    return 0.5 * beta


def student_t_two_tailed_p_value(t_value: float, degrees_freedom: float) -> float:
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive")
    x = degrees_freedom / (degrees_freedom + t_value * t_value)
    p_value = regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)
    return max(0.0, min(1.0, p_value))


def student_t_quantile(probability: float, degrees_freedom: float) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be between 0.5 and 1.0")

    low = 0.0
    high = 1.0
    while student_t_cdf(high, degrees_freedom) < probability:
        high *= 2.0

    for _ in range(100):
        mid = (low + high) / 2.0
        if student_t_cdf(mid, degrees_freedom) < probability:
            low = mid
        else:
            high = mid
    return high


def sample_standard_deviation(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def confidence_interval_margin(values: list[float], confidence: float = 0.95) -> float:
    if len(values) <= 1:
        return 0.0
    critical = student_t_quantile(0.5 + confidence / 2.0, len(values) - 1)
    return critical * sample_standard_deviation(values) / math.sqrt(len(values))


def welch_test(values_a: list[float], values_b: list[float]) -> dict[str, float]:
    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)
    sd_a = sample_standard_deviation(values_a)
    sd_b = sample_standard_deviation(values_b)
    n_a = len(values_a)
    n_b = len(values_b)
    variance_a = sd_a * sd_a
    variance_b = sd_b * sd_b
    se_squared = variance_a / n_a + variance_b / n_b

    if se_squared == 0.0:
        t_statistic = 0.0 if mean_a == mean_b else math.copysign(math.inf, mean_a - mean_b)
        degrees_freedom = float(min(n_a, n_b) - 1)
        p_value = 1.0 if mean_a == mean_b else 0.0
    else:
        t_statistic = (mean_a - mean_b) / math.sqrt(se_squared)
        numerator = se_squared * se_squared
        denominator = 0.0
        if n_a > 1:
            denominator += ((variance_a / n_a) ** 2) / (n_a - 1)
        if n_b > 1:
            denominator += ((variance_b / n_b) ** 2) / (n_b - 1)
        degrees_freedom = numerator / denominator if denominator else float(min(n_a, n_b) - 1)
        p_value = student_t_two_tailed_p_value(t_statistic, degrees_freedom)

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "standard_deviation_a": sd_a,
        "standard_deviation_b": sd_b,
        "t_statistic": t_statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
    }


def resolve_profile_path(raw_path: str, raw_file: Path, output_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    candidate = output_dir / path.name
    if candidate.exists():
        return candidate

    if len(path.parts) >= 2:
        candidate = output_dir / path.parts[-2] / path.name
        if candidate.exists():
            return candidate

    candidate = raw_file.parent / path.name
    if candidate.exists():
        return candidate

    if len(path.parts) >= 2:
        candidate = raw_file.parent / path.parts[-2] / path.name
        if candidate.exists():
            return candidate

    return path


def profile_events(path: Path) -> dict[str, list[float]]:
    events: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            events.setdefault(row["event"], []).append(float(row["microseconds"]))
    return events


def total(events: dict[str, list[float]], *names: str) -> float:
    return sum(sum(events.get(name, [])) for name in names)


def average(events: dict[str, list[float]], name: str) -> float:
    values = events.get(name, [])
    return sum(values) / len(values) if values else 0.0


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.raw.open(newline="", encoding="utf-8") as handle:
        executions = list(csv.DictReader(handle))

    run_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for execution in executions:
        implementation = execution["implementation"]
        run = int(execution["run"])
        warmup = execution["warmup"] == "1"
        profile_file = resolve_profile_path(execution["profile_file"], args.raw, args.output)
        events = profile_events(profile_file)
        metrics = {
            "compilation": float(execution["compile_microseconds"]),
            "model_setup": total(events, "model_setup"),
            "batch_loading": total(events, "load_train_batch", "load_predict_batch"),
            "training_total": total(events, "train_epoch"),
            "epoch_average": average(events, "train_epoch"),
            "batch_average": average(events, "train_batch"),
            "prediction_total": total(events, "predict_batch"),
            "cpu_gpu_transfer": total(events, "train_cpu_gpu_transfer", "predict_cpu_gpu_transfer"),
            "gpu_cpu_transfer": total(events, "predict_gpu_cpu_transfer"),
            "gpu_compute": total(events, "train_gpu_compute", "predict_gpu_compute"),
            "application_total": float(execution["application_microseconds"]),
        }

        for metric, value in metrics.items():
            run_rows.append({
                "implementation": implementation,
                "run": run,
                "warmup": int(warmup),
                "metric": metric,
                "milliseconds": f"{value / 1000.0:.9f}",
            })
            if not warmup:
                grouped.setdefault((implementation, metric), []).append(value / 1000.0)

    summary_rows: list[dict[str, object]] = []
    for (implementation, metric), values in sorted(grouped.items()):
        mean = statistics.mean(values)
        ci95 = confidence_interval_margin(values)
        summary_rows.append({
            "implementation": implementation,
            "metric": metric,
            "unit": "ms",
            "samples": len(values),
            "mean": f"{mean:.6f}",
            "median": f"{statistics.median(values):.6f}",
            "standard_deviation": f"{sample_standard_deviation(values):.6f}",
            "confidence_level": "0.95",
            "ci95_margin": f"{ci95:.6f}",
            "ci95_lower": f"{mean - ci95:.6f}",
            "ci95_upper": f"{mean + ci95:.6f}",
        })

    welch_rows: list[dict[str, object]] = []
    metrics = sorted({metric for _, metric in grouped})
    for metric in metrics:
        cuda_values = grouped.get(("cuda", metric), [])
        polyhok_values = grouped.get(("polyhok", metric), [])
        if not cuda_values or not polyhok_values:
            continue
        result = welch_test(cuda_values, polyhok_values)
        welch_rows.append({
            "metric": metric,
            "unit": "ms",
            "implementation_a": "cuda",
            "implementation_b": "polyhok",
            "samples_a": len(cuda_values),
            "samples_b": len(polyhok_values),
            "mean_a": f"{result['mean_a']:.6f}",
            "mean_b": f"{result['mean_b']:.6f}",
            "standard_deviation_a": f"{result['standard_deviation_a']:.6f}",
            "standard_deviation_b": f"{result['standard_deviation_b']:.6f}",
            "mean_difference_a_minus_b": f"{result['mean_a'] - result['mean_b']:.6f}",
            "t_statistic": f"{result['t_statistic']:.6f}",
            "degrees_freedom": f"{result['degrees_freedom']:.6f}",
            "p_value": f"{result['p_value']:.12g}",
            "significant_0_05": int(result["p_value"] < 0.05),
        })

    output_files = [
        ("performance_runs.csv", run_rows),
        ("performance_summary.csv", summary_rows),
        ("performance_welch_tests.csv", welch_rows),
    ]
    for filename, rows in output_files:
        if not rows:
            continue
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with (args.output / "performance_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrrrr}\n\\hline\nImplementacao & Medida & Media (ms) & Mediana (ms) & Desvio (ms) & IC 95\\% (ms) \\\\\n\\hline\n")
        for row in summary_rows:
            handle.write(
                f"{tex_escape(row['implementation'])} & {tex_escape(row['metric'])} & "
                f"{row['mean']} & {row['median']} & {row['standard_deviation']} & {row['ci95_margin']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    with (args.output / "performance_welch_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrr}\n\\hline\nMedida & Estatistica t & Graus de liberdade & p-valor \\\\\n\\hline\n")
        for row in welch_rows:
            if row["metric"] not in WELCH_METRICS:
                continue
            handle.write(
                f"{tex_escape(row['metric'])} & {row['t_statistic']} & "
                f"{row['degrees_freedom']} & {row['p_value']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    print(f"Execucoes consolidadas: {len(executions)}")
    for row in summary_rows:
        if row["metric"] in {"compilation", "training_total", "application_total"}:
            print(
                f"{row['implementation']:8s} {row['metric']:20s} "
                f"media={row['mean']} ms mediana={row['median']} ms "
                f"desvio={row['standard_deviation']} ms ic95=+-{row['ci95_margin']} ms"
            )


if __name__ == "__main__":
    main()
