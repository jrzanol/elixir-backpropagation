#!/usr/bin/env python3
"""Gera metricas ampliadas de complexidade de desenvolvimento GPU."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path


CUDA_PATTERNS = (
    "libsrc/cuda/*.ex",
    "c_src/MLPClassifierNIF/*.cpp",
    "c_src/MLPClassifierNIF/*.cu",
    "c_src/MLPClassifierNIF/*.h",
)
POLYHOK_PATTERNS = ("libsrc/polyhok/*.ex",)
PROJECT_PATTERNS = (
    "lib/*.ex",
    "scripts/*.py",
    "*.sh",
)


def source_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def language(path: Path) -> str:
    return {
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".cpp": "C++",
        ".cu": "CUDA",
        ".h": "C/C++ header",
        ".py": "Python",
        ".sh": "Shell",
    }.get(path.suffix.lower(), path.suffix.lower().lstrip("."))


def line_counts(path: Path) -> dict[str, int]:
    lines = read_text(path).splitlines()
    blank = 0
    comments = 0
    code_chars = 0
    in_block_comment = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue

        is_comment = False
        if path.suffix in (".ex", ".exs", ".py", ".sh"):
            is_comment = stripped.startswith("#")
        else:
            if in_block_comment:
                is_comment = True
                if "*/" in stripped:
                    in_block_comment = False
            elif stripped.startswith("/*"):
                is_comment = True
                if "*/" not in stripped:
                    in_block_comment = True
            elif stripped.startswith("//"):
                is_comment = True

        if is_comment:
            comments += 1
        else:
            code_chars += len(stripped)

    source = len(lines) - blank - comments
    return {
        "total_lines": len(lines),
        "source_lines": source,
        "blank_lines": blank,
        "comment_lines": comments,
        "code_characters": code_chars,
    }


def occurrences(files: list[Path], pattern: str, flags: int = 0) -> int:
    regex = re.compile(pattern, flags)
    return sum(len(regex.findall(read_text(path))) for path in files)


def unique_occurrences(files: list[Path], pattern: str) -> set[str]:
    regex = re.compile(pattern)
    values: set[str] = set()
    for path in files:
        values.update(regex.findall(read_text(path)))
    return values


def split_top_level_args(value: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False

    for char in value:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            text = "".join(current).strip()
            if text:
                args.append(text)
            current = []
        else:
            current.append(char)

    text = "".join(current).strip()
    if text:
        args.append(text)
    return args


def argument_count(signature: str) -> int:
    signature = signature.strip()
    if not signature or signature == "void":
        return 0
    return len(split_top_level_args(signature))


def find_matching(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def approx_cyclomatic(path: Path) -> int:
    text = read_text(path)
    if path.suffix in (".ex", ".exs"):
        keywords = r"\b(if|unless|case|cond|for|with|try|rescue|catch)\b|&&|\|\||->"
    elif path.suffix == ".py":
        keywords = r"\b(if|elif|for|while|except|case|with|and|or)\b"
    elif path.suffix == ".sh":
        keywords = r"\b(if|elif|for|while|case|until)\b|&&|\|\|"
    else:
        keywords = r"\b(if|else\s+if|for|while|case|catch)\b|&&|\|\||\?"
    return 1 + occurrences([path], keywords)


def elixir_functions(path: Path) -> list[dict[str, object]]:
    lines = read_text(path).splitlines()
    starts: list[tuple[int, int, str, str, int]] = []
    pattern = re.compile(r"^(\s*)(defp?|defk|defd|deft)\s+([a-zA-Z_][\w!?]*)")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            starts.append((index, len(match.group(1)), match.group(2), match.group(3), index + 1))

    rows: list[dict[str, object]] = []
    for pos, (start_index, indent, kind, name, start_line) in enumerate(starts):
        end_index = len(lines)
        for next_start, next_indent, _next_kind, _next_name, _next_line in starts[pos + 1 :]:
            if next_indent <= indent:
                end_index = next_start
                break
        size = max(1, end_index - start_index)
        rows.append({
            "function": name,
            "kind": kind,
            "start_line": start_line,
            "source_lines": size,
            "argument_count": line_argument_count(lines[start_index]),
        })
    return rows


def line_argument_count(line: str) -> int:
    start = line.find("(")
    if start < 0:
        return 0
    end = find_matching(line, start, "(", ")")
    if end < 0:
        return 0
    return argument_count(line[start + 1 : end])


def c_functions(path: Path) -> list[dict[str, object]]:
    text = read_text(path)
    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    pattern = re.compile(
        r"(?P<prefix>(?:__global__\s+)?(?:static\s+)?(?:[A-Za-z_][\w:<>\*&]+\s+)+)"
        r"(?P<name>[A-Za-z_][\w:]*)\s*\((?P<args>[^;{}]*)\)\s*(?:const\s*)?\{",
        re.MULTILINE,
    )
    control_names = {"if", "for", "while", "switch", "catch"}

    for match in pattern.finditer(text):
        name = match.group("name").split("::")[-1]
        if name in control_names or match.start() in seen:
            continue
        seen.add(match.start())
        open_brace = text.find("{", match.start())
        close_brace = find_matching(text, open_brace, "{", "}")
        start_line = text.count("\n", 0, match.start()) + 1
        end_line = text.count("\n", 0, close_brace) + 1 if close_brace >= 0 else len(lines)
        rows.append({
            "function": name,
            "kind": "kernel" if "__global__" in match.group("prefix") else "function",
            "start_line": start_line,
            "source_lines": max(1, end_line - start_line + 1),
            "argument_count": argument_count(match.group("args")),
        })

    return rows


def file_functions(path: Path) -> list[dict[str, object]]:
    if path.suffix in (".ex", ".exs"):
        return elixir_functions(path)
    if path.suffix in (".cu", ".cpp", ".h"):
        return c_functions(path)
    return []


def cuda_kernel_rows(root: Path, files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definition_pattern = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)
    launch_pattern = re.compile(r"([A-Za-z_]\w*)\s*<<<.*?>>>\s*\((.*?)\)", re.DOTALL)
    for path in files:
        text = read_text(path)
        rel = path.relative_to(root).as_posix()
        for name, args in definition_pattern.findall(text):
            rows.append({
                "implementation": "CUDA/NIF",
                "type": "definition",
                "name": name,
                "file": rel,
                "argument_count": argument_count(args),
            })
        for name, args in launch_pattern.findall(text):
            rows.append({
                "implementation": "CUDA/NIF",
                "type": "launch",
                "name": name,
                "file": rel,
                "argument_count": argument_count(args),
            })
    return rows


def polyhok_kernel_rows(root: Path, files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definition_pattern = re.compile(r"\b(defk|defd|deft)\s+([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)
    spawn_pattern = re.compile(r"PolyHok\.spawn_st\s*\((.*?)\)", re.DOTALL)
    for path in files:
        text = read_text(path)
        rel = path.relative_to(root).as_posix()
        for kind, name, args in definition_pattern.findall(text):
            rows.append({
                "implementation": "PolyHok",
                "type": kind,
                "name": name,
                "file": rel,
                "argument_count": argument_count(args),
            })
        for index, args in enumerate(spawn_pattern.findall(text), start=1):
            rows.append({
                "implementation": "PolyHok",
                "type": "spawn_st",
                "name": f"spawn_st_{index}",
                "file": rel,
                "argument_count": argument_count(args),
            })
    return rows


def architecture(log_path: Path) -> str:
    if not log_path.exists():
        return "nao detectada"
    match = re.search(r"Arquitetura CUDA: (sm_\d+)", read_text(log_path))
    return match.group(1) if match else "nao detectada"


def read_compilation_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compile_stats(rows: list[dict[str, str]], implementation: str) -> dict[str, object]:
    values = [float(row["microseconds"]) / 1000.0 for row in rows if row["implementation"] == implementation]
    if not values:
        return {"runs": 0, "mean_ms": 0.0, "median_ms": 0.0, "stdev_ms": 0.0}
    return {
        "runs": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def git_churn(root: Path, files: list[Path]) -> dict[str, object]:
    paths = [path.relative_to(root).as_posix() for path in files]
    if not paths:
        return {"commits": 0, "insertions": 0, "deletions": 0, "first_commit": "", "last_commit": ""}
    try:
        result = subprocess.run(
            ["git", "log", "--date=short", "--format=commit:%H:%ad", "--numstat", "--", *paths],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"commits": "n/a", "insertions": "n/a", "deletions": "n/a", "first_commit": "", "last_commit": ""}

    commits: list[str] = []
    dates: list[str] = []
    insertions = 0
    deletions = 0
    for line in result.stdout.splitlines():
        if line.startswith("commit:"):
            _tag, commit_hash, date = line.split(":", 2)
            commits.append(commit_hash)
            dates.append(date)
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            insertions += int(parts[0])
            deletions += int(parts[1])

    return {
        "commits": len(set(commits)),
        "insertions": insertions,
        "deletions": deletions,
        "first_commit": dates[-1] if dates else "",
        "last_commit": dates[0] if dates else "",
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def max_or_zero(values: list[int]) -> int:
    return max(values) if values else 0


def summarize_implementation(root: Path, implementation: str, files: list[Path]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    languages: Counter[str] = Counter()
    totals = Counter()
    file_rows: list[dict[str, object]] = []
    function_rows: list[dict[str, object]] = []
    function_sizes: list[int] = []
    function_args: list[int] = []

    for path in files:
        counts = line_counts(path)
        functions = file_functions(path)
        cyclomatic = approx_cyclomatic(path)
        languages[language(path)] += 1
        totals.update(counts)
        totals["cyclomatic"] += cyclomatic
        totals["files"] += 1

        function_sizes.extend(int(row["source_lines"]) for row in functions)
        function_args.extend(int(row["argument_count"]) for row in functions)
        rel = path.relative_to(root).as_posix()

        file_rows.append({
            "implementation": implementation,
            "file": rel,
            "language": language(path),
            **counts,
            "cyclomatic_approx": cyclomatic,
            "function_count": len(functions),
            "max_function_source_lines": max_or_zero([int(row["source_lines"]) for row in functions]),
        })

        for row in functions:
            function_rows.append({
                "implementation": implementation,
                "file": rel,
                **row,
            })

    summary = {
        "files": totals["files"],
        "languages": ", ".join(f"{name}: {count}" for name, count in sorted(languages.items())),
        "language_count": len(languages),
        "total_lines": totals["total_lines"],
        "source_lines": totals["source_lines"],
        "blank_lines": totals["blank_lines"],
        "comment_lines": totals["comment_lines"],
        "code_characters": totals["code_characters"],
        "cyclomatic_total_approx": totals["cyclomatic"],
        "function_count": len(function_rows),
        "avg_function_source_lines": mean([float(size) for size in function_sizes]),
        "max_function_source_lines": max_or_zero(function_sizes),
        "avg_function_argument_count": mean([float(value) for value in function_args]),
        "max_function_argument_count": max_or_zero(function_args),
    }
    return summary, file_rows, function_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metric_explanations() -> list[dict[str, object]]:
    return [
        {
            "metric": "Arquivos e linguagens",
            "purpose": "Mede a superficie estrutural da implementacao e o custo de alternar entre linguagens/ferramentas.",
            "method": "Conta arquivos, extensoes e numero de linguagens no escopo de cada implementacao.",
            "basis": "Usado como evidencia de esforco manual e superficie de modificacao.",
        },
        {
            "metric": "Linhas de codigo e caracteres",
            "purpose": "Aproxima tamanho e esforco manual de escrita/leitura.",
            "method": "Conta linhas fisicas, linhas nao vazias sem comentarios e caracteres de codigo.",
            "basis": "LOC/NOC aparecem em estudos de produtividade GPU; nao devem ser usados isoladamente.",
        },
        {
            "metric": "Complexidade ciclomatica aproximada",
            "purpose": "Aproxima numero de caminhos logicos e esforco de teste.",
            "method": "Conta decisoes sintaticas por arquivo, como if, case, for, while e operadores booleanos.",
            "basis": "McCabe/NIST relaciona complexidade ciclomatica a caminhos de teste.",
        },
        {
            "metric": "Tamanho e argumentos de funcoes/kernels",
            "purpose": "Mede granularidade, acoplamento e dificuldade de revisao.",
            "method": "Extrai funcoes/kernels e calcula tamanho medio/maximo e numero medio/maximo de argumentos.",
            "basis": "Funcoes longas e assinaturas extensas aumentam custo de compreensao e modificacao.",
        },
        {
            "metric": "Gerenciamento CPU/GPU",
            "purpose": "Mede o esforco explicito de memoria, transferencia, sincronizacao e lancamento de kernels.",
            "method": "Conta cudaMalloc/cudaMemcpy/cudaMemset/cudaFree, new_gnx/get_gnx, spawn_st, launches e sincronizacoes.",
            "basis": "Malik et al. separam memoria, comunicacao e execucao GPU como esforco conceitual.",
        },
        {
            "metric": "Observabilidade",
            "purpose": "Mede facilidade de ver estado interno, tempos e resultados durante testes/depuracao.",
            "method": "Conta snapshots, Profiler, timings, prints e checks de erro.",
            "basis": "Testabilidade depende de observabilidade e depurabilidade.",
        },
        {
            "metric": "Controlabilidade",
            "purpose": "Mede facilidade de reproduzir cenarios controlando seed, dataset, batch, epochs e debug.",
            "method": "Conta leituras de variaveis de ambiente e parametros BACKPROP_*.",
            "basis": "Testabilidade tambem depende de controlabilidade.",
        },
        {
            "metric": "Tempo de compilacao repetido",
            "purpose": "Mede custo pratico de feedback de desenvolvimento.",
            "method": "Executa os scripts de compilacao varias vezes e calcula media, mediana e desvio padrao.",
            "basis": "Tempo de feedback influencia produtividade e ciclo de depuracao.",
        },
        {
            "metric": "Churn historico",
            "purpose": "Aproxima esforco real de evolucao/manutencao quando ha historico git.",
            "method": "Conta commits, linhas adicionadas e removidas por implementacao.",
            "basis": "Churn e usado como proxy de manutencao/retrabalho, mas depende da qualidade do historico.",
        },
    ]


def add_row(rows: list[dict[str, object]], criterion: str, metric_type: str, cuda: object, polyhok: object, method: str) -> None:
    rows.append({
        "criterion": criterion,
        "type": metric_type,
        "cuda": cuda,
        "polyhok": polyhok,
        "method": method,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compilation-csv", type=Path, required=True)
    parser.add_argument("--cuda-log", type=Path, required=True)
    parser.add_argument("--polyhok-log", type=Path, required=True)
    args = parser.parse_args()

    root = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cuda_files = source_files(root, CUDA_PATTERNS)
    polyhok_files = source_files(root, POLYHOK_PATTERNS)
    project_files = source_files(root, PROJECT_PATTERNS)

    cuda_summary, cuda_file_rows, cuda_function_rows = summarize_implementation(root, "CUDA/NIF", cuda_files)
    polyhok_summary, polyhok_file_rows, polyhok_function_rows = summarize_implementation(root, "PolyHok", polyhok_files)

    kernel_rows = cuda_kernel_rows(root, cuda_files) + polyhok_kernel_rows(root, polyhok_files)
    cuda_kernel_args = [int(row["argument_count"]) for row in kernel_rows if row["implementation"] == "CUDA/NIF"]
    polyhok_kernel_args = [int(row["argument_count"]) for row in kernel_rows if row["implementation"] == "PolyHok"]

    compilation_rows = read_compilation_rows(args.compilation_csv)
    cuda_compile = compile_stats(compilation_rows, "cuda")
    polyhok_compile = compile_stats(compilation_rows, "polyhok")
    compile_summary_rows = [
        {"implementation": "CUDA/NIF", "architecture": architecture(args.cuda_log), **cuda_compile},
        {"implementation": "PolyHok", "architecture": architecture(args.polyhok_log), **polyhok_compile},
    ]

    cuda_churn = git_churn(root, cuda_files)
    polyhok_churn = git_churn(root, polyhok_files)

    cuda_memory_calls = occurrences(cuda_files, r"\bcuda(?:Malloc|Free|Memcpy|Memset)\s*\(")
    cuda_transfers = occurrences(cuda_files, r"\bcudaMemcpy\s*\(")
    cuda_sync = occurrences(cuda_files, r"\bcudaDeviceSynchronize\s*\(")
    cuda_error_checks = occurrences(cuda_files, r"\bcudaGetLastError\s*\(|\bCudaOk\s*\(")
    cuda_kernel_defs = occurrences(cuda_files, r"\b__global__\s+void\s+")
    cuda_launches = occurrences(cuda_files, r"<<<")
    cuda_nif_exports = occurrences(cuda_files, r'\{"[a-zA-Z0-9_]+",\s*\d+')

    polyhok_allocations = occurrences(polyhok_files, r"PolyHok\.new_gnx\s*\(")
    polyhok_transfers = occurrences(polyhok_files, r"PolyHok\.get_gnx\s*\(")
    polyhok_launches = occurrences(polyhok_files, r"PolyHok\.spawn_st\s*\(")
    polyhok_sync = occurrences(polyhok_files, r"PolyHok\.synchronize\s*\(")
    polyhok_kernels = occurrences(polyhok_files, r"\bdefk\s+")
    polyhok_signatures = occurrences(polyhok_files, r"\bdeft\s*\(")
    polyhok_static_arrays = occurrences(polyhok_files, r"\b(?:act|delta)\[512\]")

    cuda_observability = (
        occurrences(cuda_files, r"DEBUG_SNAPSHOT")
        + occurrences(cuda_files, r"GetLastTimings|last_timings")
        + occurrences(cuda_files, r"\bstd::printf|\bfprintf")
        + cuda_error_checks
    )
    polyhok_observability = (
        occurrences(polyhok_files, r"DEBUG_SNAPSHOT")
        + occurrences(polyhok_files, r"Profiler\.(?:record|runtime|measure)")
        + occurrences(polyhok_files, r"\bIO\.puts")
    )
    cuda_control = occurrences(cuda_files, r"std::getenv|System\.get_env") + len(unique_occurrences(cuda_files, r"BACKPROP_[A-Z0-9_]+"))
    polyhok_control = occurrences(polyhok_files, r"System\.get_env") + len(unique_occurrences(polyhok_files, r"BACKPROP_[A-Z0-9_]+"))
    project_control = occurrences(project_files, r"System\.get_env|std::getenv|os\.environ") + len(unique_occurrences(project_files, r"BACKPROP_[A-Z0-9_]+"))

    rows: list[dict[str, object]] = []
    add_row(rows, "Quantidade de arquivos", "quantitativo", cuda_summary["files"], polyhok_summary["files"], "Arquivos especificos da implementacao.")
    add_row(rows, "Linguagens utilizadas", "quantitativo", cuda_summary["languages"], polyhok_summary["languages"], "Classificacao pela extensao dos arquivos.")
    add_row(rows, "Linhas fisicas", "quantitativo", cuda_summary["total_lines"], polyhok_summary["total_lines"], "Todas as linhas dos arquivos especificos.")
    add_row(rows, "Linhas de codigo", "quantitativo", cuda_summary["source_lines"], polyhok_summary["source_lines"], "Linhas nao vazias, excluindo linhas somente de comentario.")
    add_row(rows, "Caracteres de codigo", "quantitativo", cuda_summary["code_characters"], polyhok_summary["code_characters"], "Soma dos caracteres em linhas de codigo.")
    add_row(rows, "Complexidade ciclomatica aproximada", "quantitativo", cuda_summary["cyclomatic_total_approx"], polyhok_summary["cyclomatic_total_approx"], "Contagem heuristica de decisoes sintaticas.")
    add_row(rows, "Funcoes/kernels", "quantitativo", cuda_summary["function_count"], polyhok_summary["function_count"], "Extracao heuristica de funcoes, kernels e funcoes DSL.")
    add_row(rows, "Tamanho medio de funcao", "quantitativo", f"{cuda_summary['avg_function_source_lines']:.2f}", f"{polyhok_summary['avg_function_source_lines']:.2f}", "Media de linhas por funcao/kernel.")
    add_row(rows, "Maior funcao/kernel", "quantitativo", cuda_summary["max_function_source_lines"], polyhok_summary["max_function_source_lines"], "Maior bloco funcional detectado.")
    add_row(rows, "Media de argumentos por funcao", "quantitativo", f"{cuda_summary['avg_function_argument_count']:.2f}", f"{polyhok_summary['avg_function_argument_count']:.2f}", "Media de parametros por funcao/kernel.")
    add_row(rows, "Maior assinatura de funcao", "quantitativo", cuda_summary["max_function_argument_count"], polyhok_summary["max_function_argument_count"], "Maior numero de argumentos detectado.")
    add_row(rows, "Kernels declarados", "quantitativo", cuda_kernel_defs, polyhok_kernels, "CUDA __global__ vs PolyHok defk.")
    add_row(rows, "Lancamentos GPU", "quantitativo", cuda_launches, polyhok_launches, "CUDA <<<...>>> vs PolyHok.spawn_st.")
    add_row(rows, "Argumentos medios de kernel/lancamento", "quantitativo", f"{mean([float(v) for v in cuda_kernel_args]):.2f}", f"{mean([float(v) for v in polyhok_kernel_args]):.2f}", "Assinaturas e chamadas de kernel.")
    add_row(rows, "Maior kernel/lancamento", "quantitativo", max_or_zero(cuda_kernel_args), max_or_zero(polyhok_kernel_args), "Maior quantidade de argumentos em definicao/chamada.")
    add_row(rows, "Gerenciamento de memoria GPU", "quantitativo", cuda_memory_calls, polyhok_allocations, "cudaMalloc/cudaFree/cudaMemcpy/cudaMemset vs PolyHok.new_gnx.")
    add_row(rows, "Transferencias CPU/GPU", "quantitativo", cuda_transfers, polyhok_transfers, "cudaMemcpy vs PolyHok.get_gnx; new_gnx e considerado em memoria.")
    add_row(rows, "Sincronizacoes explicitas", "quantitativo", cuda_sync, polyhok_sync, "cudaDeviceSynchronize vs PolyHok.synchronize.")
    add_row(rows, "Checks de erro/assincronia", "quantitativo", cuda_error_checks, "n/a", "cudaGetLastError/CudaOk; PolyHok encapsula parte desses checks.")
    add_row(rows, "Observabilidade", "quantitativo", cuda_observability, polyhok_observability, "Snapshots, timers, prints e checks de erro.")
    add_row(rows, "Controlabilidade da implementacao", "quantitativo", cuda_control, polyhok_control, "Variaveis de ambiente e knobs BACKPROP_* no codigo especifico.")
    add_row(rows, "Controlabilidade do projeto", "quantitativo", project_control, project_control, "Parametros compartilhados de seed, dataset, epochs, batch size e debug.")
    add_row(rows, "Tempo medio de compilacao", "quantitativo", f"{cuda_compile['mean_ms']:.3f} ms", f"{polyhok_compile['mean_ms']:.3f} ms", "Media de execucoes repetidas dos scripts oficiais.")
    add_row(rows, "Tempo mediano de compilacao", "quantitativo", f"{cuda_compile['median_ms']:.3f} ms", f"{polyhok_compile['median_ms']:.3f} ms", "Mediana de execucoes repetidas dos scripts oficiais.")
    add_row(rows, "Desvio da compilacao", "quantitativo", f"{cuda_compile['stdev_ms']:.3f} ms", f"{polyhok_compile['stdev_ms']:.3f} ms", "Desvio padrao amostral das compilacoes.")
    add_row(rows, "Churn - commits", "quantitativo", cuda_churn["commits"], polyhok_churn["commits"], "Historico git nos arquivos especificos.")
    add_row(rows, "Churn - linhas adicionadas", "quantitativo", cuda_churn["insertions"], polyhok_churn["insertions"], "Soma de insercoes em git log --numstat.")
    add_row(rows, "Churn - linhas removidas", "quantitativo", cuda_churn["deletions"], polyhok_churn["deletions"], "Soma de remocoes em git log --numstat.")
    add_row(rows, "Necessidade de codigo C/CUDA", "qualitativo", f"Obrigatoria: {cuda_kernel_defs} kernels CUDA e {cuda_nif_exports} funcoes NIF exportadas.", "Nao no codigo da aplicacao: kernels ficam em Elixir/DSL; C/CUDA permanece na dependencia PolyHok.", "Inspecao de linguagens e fronteiras.")
    add_row(rows, "Superficie de modificacao de kernel", "qualitativo", "Maior: envolve kernel CUDA, wrapper/launcher, header, C++ NIF e modulo Elixir.", "Menor: kernel e chamada ficam em Elixir/DSL, mas ainda dependem de tipos estaticos e recompilacao.", "Numero de linguagens, arquivos e fronteiras que precisam permanecer coerentes.")
    add_row(rows, "Restricoes estaticas PolyHok", "qualitativo", "Nao se aplica diretamente; CUDA ainda compila para arquitetura alvo.", f"{polyhok_signatures} deft, {polyhok_static_arrays} buffers fixos de 512 e tipos aceitos pela DSL.", "Inspecao de deft/defk, arrays fixos e logs de arquitetura.")

    write_csv(output / "development_complexity.csv", rows)
    write_csv(output / "development_complexity_files.csv", cuda_file_rows + polyhok_file_rows)
    write_csv(output / "development_complexity_functions.csv", cuda_function_rows + polyhok_function_rows)
    write_csv(output / "development_complexity_kernels.csv", kernel_rows)
    write_csv(output / "development_complexity_metric_explanations.csv", metric_explanations())
    write_csv(output / "development_complexity_compile_summary.csv", compile_summary_rows)

    with (output / "development_complexity.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{p{0.20\\textwidth}p{0.35\\textwidth}p{0.35\\textwidth}}\n")
        handle.write("\\hline\nCriterio & CUDA/NIF & PolyHok \\\\\n\\hline\n")
        for row in rows:
            handle.write(f"{tex_escape(row['criterion'])} & {tex_escape(row['cuda'])} & {tex_escape(row['polyhok'])} \\\\\n")
        handle.write("\\hline\n\\end{tabular}\n")

    with (output / "development_complexity.md").open("w", encoding="utf-8") as handle:
        handle.write("# Analise de complexidade de desenvolvimento\n\n")
        handle.write("Este relatorio compara CUDA/NIF e PolyHok por metricas quantitativas e qualitativas baseadas em evidencias do codigo. ")
        handle.write("O objetivo nao e declarar uma implementacao melhor em todos os aspectos, mas separar tamanho, esforco manual, complexidade conceitual, testabilidade, compilacao e manutencao.\n\n")
        handle.write("## Como as metricas sao feitas e para que servem\n\n")
        handle.write("| Metrica | Para que serve | Como e feita | Base teorica |\n|---|---|---|---|\n")
        for row in metric_explanations():
            handle.write(f"| {row['metric']} | {row['purpose']} | {row['method']} | {row['basis']} |\n")

        handle.write("\n## Base teorica resumida\n\n")
        handle.write("- Malik et al. comparam produtividade GPU separando LOC/NOC, memoria, comunicacao, setup de kernel e execucao GPU.\n")
        handle.write("- Kosar et al. usam dimensoes cognitivas para comparar DSL e linguagem geral em tarefas de aprendizado, compreensao e evolucao.\n")
        handle.write("- McCabe/NIST relaciona complexidade ciclomatica com quantidade de caminhos de teste.\n")
        handle.write("- Trabalhos de testabilidade destacam observabilidade e controlabilidade como propriedades centrais para testar e depurar.\n")
        handle.write("- LOC e caracteres sao evidencias de tamanho/esforco, mas nao devem ser usados isoladamente.\n\n")
        handle.write("## Referencias usadas\n\n")
        handle.write("- Malik et al., `Productivity of GPUs under different programming paradigms`: https://www.eecg.utoronto.ca/~amza/ece1747h/papers/cpe1860.pdf\n")
        handle.write("- Kosar et al., `Comparing General-Purpose and Domain-Specific Languages`: https://www.comsis.org/pdf.php?id=226-0911\n")
        handle.write("- McCabe/NIST, `Structured Testing`: https://www.eng.auburn.edu/~kchang/comp6710/readings/Integration.Testing.McCabe.NIST.pdf\n")
        handle.write("- Sharma et al., `Software Testability`: https://www.tusharma.in/preprints/EMSE2023_Testability.pdf\n")
        handle.write("- Barb et al., `A statistical study of the relevance of lines of code measures`: https://www.researchgate.net/publication/267761547_A_statistical_study_of_the_relevance_of_lines_of_code_measures_in_software_projects\n\n")

        handle.write("## Resultado consolidado\n\n")
        handle.write("| Criterio | Tipo | CUDA/NIF | PolyHok | Metodo |\n|---|---|---|---|---|\n")
        for row in rows:
            cuda = str(row["cuda"]).replace("|", "\\|")
            polyhok = str(row["polyhok"]).replace("|", "\\|")
            method = str(row["method"]).replace("|", "\\|")
            handle.write(f"| {row['criterion']} | {row['type']} | {cuda} | {polyhok} | {method} |\n")

        handle.write("\n## Arquivos gerados\n\n")
        handle.write("- `development_complexity.csv`: tabela consolidada.\n")
        handle.write("- `development_complexity_files.csv`: metricas por arquivo.\n")
        handle.write("- `development_complexity_functions.csv`: funcoes/kernels detectados.\n")
        handle.write("- `development_complexity_kernels.csv`: definicoes e lancamentos GPU.\n")
        handle.write("- `development_complexity_compile_summary.csv`: media, mediana e desvio da compilacao.\n")
        handle.write("- `development_complexity_metric_explanations.csv`: explicacao tabular das metricas.\n")
        handle.write("- `development_complexity.tex`: tabela para LaTeX.\n")

    print(f"CUDA/NIF: {cuda_summary['files']} arquivos, {cuda_summary['source_lines']} linhas de codigo")
    print(f"PolyHok : {polyhok_summary['files']} arquivos, {polyhok_summary['source_lines']} linhas de codigo")
    print(f"Compilacao CUDA/NIF: media={cuda_compile['mean_ms']:.3f} ms mediana={cuda_compile['median_ms']:.3f} ms desvio={cuda_compile['stdev_ms']:.3f} ms")
    print(f"Compilacao PolyHok : media={polyhok_compile['mean_ms']:.3f} ms mediana={polyhok_compile['median_ms']:.3f} ms desvio={polyhok_compile['stdev_ms']:.3f} ms")
    print(f"Relatorio: {output}")


if __name__ == "__main__":
    main()
