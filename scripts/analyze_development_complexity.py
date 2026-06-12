#!/usr/bin/env python3
"""Gera indicadores objetivos e analise qualitativa das implementacoes GPU."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


CUDA_PATTERNS = (
    "libsrc/cuda/*.ex",
    "c_src/MLPClassifierNIF/*.cpp",
    "c_src/MLPClassifierNIF/*.cu",
    "c_src/MLPClassifierNIF/*.h",
)
POLYHOK_PATTERNS = ("libsrc/polyhok/*.ex",)


def source_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def language(path: Path) -> str:
    return {
        ".ex": "Elixir",
        ".cpp": "C++",
        ".cu": "CUDA",
        ".h": "C/C++ header",
    }.get(path.suffix.lower(), path.suffix.lower().lstrip("."))


def line_counts(path: Path) -> tuple[int, int, int, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blank = 0
    comments = 0
    in_block_comment = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue

        if path.suffix == ".ex":
            if stripped.startswith("#"):
                comments += 1
            continue

        if in_block_comment:
            comments += 1
            if "*/" in stripped:
                in_block_comment = False
            continue

        if stripped.startswith("/*"):
            comments += 1
            if "*/" not in stripped:
                in_block_comment = True
        elif stripped.startswith("//"):
            comments += 1

    source = len(lines) - blank - comments
    return len(lines), source, blank, comments


def occurrences(files: list[Path], pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(len(regex.findall(path.read_text(encoding="utf-8", errors="replace"))) for path in files)


def architecture(log_path: Path) -> str:
    if not log_path.exists():
        return "nao detectada"
    match = re.search(r"Arquitetura CUDA: (sm_\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else "nao detectada"


def tex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-compile-us", type=int, required=True)
    parser.add_argument("--polyhok-compile-us", type=int, required=True)
    parser.add_argument("--cuda-log", type=Path, required=True)
    parser.add_argument("--polyhok-log", type=Path, required=True)
    args = parser.parse_args()

    root = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    implementation_files = {
        "CUDA/NIF": source_files(root, CUDA_PATTERNS),
        "PolyHok": source_files(root, POLYHOK_PATTERNS),
    }

    file_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for implementation, files in implementation_files.items():
        languages: Counter[str] = Counter()
        total_lines = source_lines = blank_lines = comment_lines = 0
        for path in files:
            total, source, blank, comments = line_counts(path)
            lang = language(path)
            languages[lang] += 1
            total_lines += total
            source_lines += source
            blank_lines += blank
            comment_lines += comments
            file_rows.append({
                "implementation": implementation,
                "file": path.relative_to(root).as_posix(),
                "language": lang,
                "total_lines": total,
                "source_lines": source,
                "blank_lines": blank,
                "comment_lines": comments,
            })

        summaries[implementation] = {
            "files": len(files),
            "total_lines": total_lines,
            "source_lines": source_lines,
            "languages": ", ".join(f"{name}: {count}" for name, count in sorted(languages.items())),
        }

    cuda_files = implementation_files["CUDA/NIF"]
    polyhok_files = implementation_files["PolyHok"]
    cuda_memory_calls = occurrences(cuda_files, r"\bcuda(?:Malloc|Free|Memcpy|Memset)\s*\(")
    cuda_kernels = occurrences(cuda_files, r"\b__global__\s+void\s+")
    cuda_nif_exports = occurrences(cuda_files, r'\{"[a-zA-Z0-9_]+",\s*\d+')
    polyhok_allocations = occurrences(polyhok_files, r"PolyHok\.new_gnx\s*\(")
    polyhok_launches = occurrences(polyhok_files, r"PolyHok\.spawn_st\s*\(")
    polyhok_kernels = occurrences(polyhok_files, r"\bdefk\s+")
    polyhok_signatures = occurrences(polyhok_files, r"\bdeft\s*\(")
    polyhok_static_arrays = occurrences(polyhok_files, r"\b(?:act|delta)\[512\]")

    cuda_compile_ms = args.cuda_compile_us / 1000.0
    polyhok_compile_ms = args.polyhok_compile_us / 1000.0
    cuda_arch = architecture(args.cuda_log)
    polyhok_arch = architecture(args.polyhok_log)

    rows = [
        {
            "criterion": "Quantidade de arquivos",
            "type": "quantitativo",
            "cuda": summaries["CUDA/NIF"]["files"],
            "polyhok": summaries["PolyHok"]["files"],
            "method": "Arquivos especificos da implementacao; codigo compartilhado e dependencias externas excluidos.",
        },
        {
            "criterion": "Linhas fisicas",
            "type": "quantitativo",
            "cuda": summaries["CUDA/NIF"]["total_lines"],
            "polyhok": summaries["PolyHok"]["total_lines"],
            "method": "Todas as linhas dos arquivos especificos, incluindo vazias e comentarios.",
        },
        {
            "criterion": "Linhas de codigo",
            "type": "quantitativo",
            "cuda": summaries["CUDA/NIF"]["source_lines"],
            "polyhok": summaries["PolyHok"]["source_lines"],
            "method": "Linhas nao vazias, excluindo linhas formadas somente por comentarios.",
        },
        {
            "criterion": "Linguagens utilizadas",
            "type": "quantitativo",
            "cuda": summaries["CUDA/NIF"]["languages"],
            "polyhok": summaries["PolyHok"]["languages"],
            "method": "Classificacao pela extensao dos arquivos da implementacao.",
        },
        {
            "criterion": "Tempo de compilacao",
            "type": "quantitativo",
            "cuda": f"{cuda_compile_ms:.3f} ms ({cuda_arch})",
            "polyhok": f"{polyhok_compile_ms:.3f} ms ({polyhok_arch})",
            "method": "Tempo de parede de uma compilacao completa pelos scripts oficiais; download PolyHok reutilizado quando existente.",
        },
        {
            "criterion": "Dificuldade de depuracao",
            "type": "qualitativo",
            "cuda": "Alta: fluxo Elixir -> NIF C++ -> host CUDA -> kernel; erros podem ocorrer de forma assincrona.",
            "polyhok": "Media/alta: codigo fica em Elixir, mas erros da DSL, geracao CUDA e type inference podem apontar para codigo gerado.",
            "method": "Escala baixa/media/alta baseada no numero de camadas, ferramentas e distancia entre fonte e kernel executado.",
        },
        {
            "criterion": "Gerenciamento de memoria",
            "type": "qualitativo",
            "cuda": f"Manual e alto: {cuda_memory_calls} chamadas cudaMalloc/cudaFree/cudaMemcpy/cudaMemset detectadas.",
            "polyhok": f"Abstraido, mas explicito: {polyhok_allocations} new_gnx; buffers e transferencias ainda sao controlados no host.",
            "method": "Contagem de primitivas de memoria e inspecao do ciclo de vida dos buffers.",
        },
        {
            "criterion": "Necessidade de codigo C/CUDA",
            "type": "qualitativo",
            "cuda": f"Obrigatoria: {cuda_kernels} kernels CUDA e {cuda_nif_exports} funcoes NIF exportadas.",
            "polyhok": "Nao no codigo da aplicacao: kernels sao escritos na DSL Elixir; C/CUDA permanece dentro da dependencia PolyHok.",
            "method": "Inspecao das linguagens e das fronteiras NIF presentes no escopo especifico.",
        },
        {
            "criterion": "Facilidade de modificar kernels",
            "type": "qualitativo",
            "cuda": "Media/baixa: alteracoes podem exigir ajustar kernel, launcher, header, host C++ e interface NIF.",
            "polyhok": f"Alta dentro da DSL: {polyhok_kernels} kernels e {polyhok_launches} lancamentos permanecem em Elixir; recompilacao ainda e necessaria.",
            "method": "Quantidade de fronteiras que precisam permanecer coerentes apos uma alteracao.",
        },
        {
            "criterion": "Restricoes da compilacao estatica PolyHok",
            "type": "qualitativo",
            "cuda": "Nao se aplica a DSL; ainda exige compilacao para a arquitetura CUDA alvo e tipos definidos em C/C++.",
            "polyhok": (
                f"Relevantes: {polyhok_signatures} assinaturas deft, tipos suportados pela DSL e "
                f"{polyhok_static_arrays} buffers locais fixos de 512 elementos; topologia total limitada a 512 neuronios neste projeto."
            ),
            "method": "Inspecao de deft/defk, arrays locais estaticos, validacao de topologia e compilacao por arquitetura.",
        },
    ]

    write_csv(output / "development_complexity.csv", rows)
    write_csv(output / "development_complexity_files.csv", file_rows)

    with (output / "development_complexity.tex").open("w", encoding="utf-8") as handle:
        handle.write(
            "\\begin{tabular}{p{0.18\\textwidth}p{0.35\\textwidth}p{0.35\\textwidth}}\n"
            "\\hline\nCriterio & CUDA/NIF & PolyHok \\\\\n\\hline\n"
        )
        for row in rows:
            handle.write(
                f"{tex_escape(row['criterion'])} & {tex_escape(row['cuda'])} & "
                f"{tex_escape(row['polyhok'])} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n")

    with (output / "development_complexity.md").open("w", encoding="utf-8") as handle:
        handle.write("# Analise de complexidade de desenvolvimento\n\n")
        handle.write("## Metodologia\n\n")
        handle.write(
            "A contagem principal inclui somente `libsrc/cuda`, `c_src/MLPClassifierNIF` "
            "e `libsrc/polyhok`. Modulos compartilhados, testes, scripts, artefatos gerados, "
            "`priv`, `_build` e a dependencia externa `deps/poly_hok` nao entram nas linhas "
            "atribuidas a uma implementacao. A dependencia e excluida pelo mesmo motivo que "
            "a runtime CUDA nao e contabilizada como codigo da implementacao CUDA/NIF.\n\n"
        )
        handle.write("| Criterio | CUDA/NIF | PolyHok |\n|---|---|---|\n")
        for row in rows:
            cuda = str(row["cuda"]).replace("|", "\\|")
            polyhok = str(row["polyhok"]).replace("|", "\\|")
            handle.write(f"| {row['criterion']} | {cuda} | {polyhok} |\n")

        handle.write("\n## Indicadores auxiliares\n\n")
        handle.write(f"- Chamadas manuais de memoria CUDA: {cuda_memory_calls}.\n")
        handle.write(f"- Kernels CUDA nativos: {cuda_kernels}.\n")
        handle.write(f"- Funcoes NIF exportadas: {cuda_nif_exports}.\n")
        handle.write(f"- Alocacoes `PolyHok.new_gnx`: {polyhok_allocations}.\n")
        handle.write(f"- Lancamentos `PolyHok.spawn_st`: {polyhok_launches}.\n")
        handle.write(f"- Kernels PolyHok `defk`: {polyhok_kernels}.\n")
        handle.write(f"- Assinaturas estaticas `deft`: {polyhok_signatures}.\n")

    print(f"CUDA/NIF: {summaries['CUDA/NIF']['files']} arquivos, {summaries['CUDA/NIF']['source_lines']} linhas de codigo")
    print(f"PolyHok : {summaries['PolyHok']['files']} arquivos, {summaries['PolyHok']['source_lines']} linhas de codigo")
    print(f"Compilacao CUDA/NIF: {cuda_compile_ms:.3f} ms")
    print(f"Compilacao PolyHok : {polyhok_compile_ms:.3f} ms")
    print(f"Relatorio: {output}")


if __name__ == "__main__":
    main()
