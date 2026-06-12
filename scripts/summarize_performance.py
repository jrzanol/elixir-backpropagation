#!/usr/bin/env python3
"""Consolida execucoes repetidas do benchmark em CSV e LaTeX."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


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
        events = profile_events(Path(execution["profile_file"]))
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
        summary_rows.append({
            "implementation": implementation,
            "metric": metric,
            "unit": "ms",
            "samples": len(values),
            "mean": f"{statistics.mean(values):.6f}",
            "median": f"{statistics.median(values):.6f}",
            "standard_deviation": f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}",
        })

    for filename, rows in [("performance_runs.csv", run_rows), ("performance_summary.csv", summary_rows)]:
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with (args.output / "performance_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrrr}\n\\hline\nImplementacao & Medida & Media (ms) & Mediana (ms) & Desvio (ms) \\\\\n\\hline\n")
        for row in summary_rows:
            handle.write(
                f"{tex_escape(row['implementation'])} & {tex_escape(row['metric'])} & "
                f"{row['mean']} & {row['median']} & {row['standard_deviation']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    print(f"Execucoes consolidadas: {len(executions)}")
    for row in summary_rows:
        if row["metric"] in {"compilation", "training_total", "application_total"}:
            print(
                f"{row['implementation']:8s} {row['metric']:20s} "
                f"media={row['mean']} ms mediana={row['median']} ms desvio={row['standard_deviation']} ms"
            )


if __name__ == "__main__":
    main()
