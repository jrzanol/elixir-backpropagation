# Relatorio do ambiente de execucao

Gerado em: 2026-06-23 23:33:04 +0000

Este relatorio registra o ambiente onde os benchmarks foram executados. Ele serve para tornar os resultados reproduziveis e para documentar diferencas de hardware, sistema operacional, drivers, CUDA, Elixir/Erlang e dependencias relevantes.

## Como ler

- `environment_summary.csv`: tabela chave/valor consolidada para importar em planilha.
- `raw/`: saidas brutas dos comandos usados, preservadas para auditoria.
- Valores marcados como `nao detectado` indicam que a ferramenta ou integracao nao estava disponivel no ambiente.

## Execucao

| Item | Valor |
|---|---|
| `data_hora_local` | 2026-06-23 23:32:59 +0000 |
| `projeto` | /workspace |
| `relatorio` | /workspace/reports/environment_20260623_233041 |
| `usuario` | root |
| `hostname` | 7a1edfbc468b |

## Linux

| Item | Valor |
|---|---|
| `distribuicao` | Ubuntu 24.04.4 LTS |
| `id` | ubuntu |
| `versao` | 24.04 |
| `kernel` | Linux 5.15.0-181-generic x86_64 GNU/Linux |
| `wsl_detectado` | nao |

## Windows

| Item | Valor |
|---|---|
| `caption` | nao detectado |
| `version` | nao detectado |
| `build` | nao detectado |
| `architecture` | nao detectado |

## CPU

| Item | Valor |
|---|---|
| `modelo` | Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz |
| `arquitetura` | x86_64 |
| `cpus_logicas` | 72 |
| `sockets` | 2 |
| `cores_por_socket` | 18 |
| `threads_por_core` | 2 |

## Memoria

| Item | Valor |
|---|---|
| `ram_total` | 125Gi |
| `ram_disponivel` | 113Gi |
| `swap_total` | 8.0Gi |

## GPU

| Item | Valor |
|---|---|
| `cuda_driver_runtime` | 12.8 |
| `gpu_0_nome` | NVIDIA GeForce RTX 4060 |
| `gpu_0_driver` | 570.133.07 |
| `gpu_0_memoria_total` | 8188 MiB |
| `gpu_0_compute_capability` | 8.9 |
| `gpu_0_pci_bus` | 00000000:02:00.0 |

## CUDA

| Item | Valor |
|---|---|
| `nvcc` | nvcc: NVIDIA (R) Cuda compiler driver |
| `nvcc_release` | 12.8 |
| `CUDA_HOME` | /usr/local/cuda |
| `CUDA_PATH` | nao definido |

## Elixir/Erlang

| Item | Valor |
|---|---|
| `elixir` | Erlang/OTP 27 [erts-15.2.7] [source] [64-bit] [smp:72:11] [ds:72:11:10] [async-threads:1] [jit:ns];;Elixir 1.18.4 (compiled with Erlang/OTP 27); |
| `mix` | Erlang/OTP 27 [erts-15.2.7] [source] [64-bit] [smp:72:11] [ds:72:11:10] [async-threads:1] [jit:ns];;Mix 1.18.4 (compiled with Erlang/OTP 27); |
| `erlang_otp` | 27 |
| `asdf` | elixir          1.18.4-otp-27   /workspace/.tool-versions;erlang          27.3.4          /workspace/.tool-versions; |

## Compiladores

| Item | Valor |
|---|---|
| `gcc` | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| `g++` | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| `make` | GNU Make 4.3 |
| `git` | git version 2.43.0 |

## Python

| Item | Valor |
|---|---|
| `python3` | Python 3.12.13 |
| `pip3` | pip 26.1.2 from /venv/main/lib/python3.12/site-packages/pip (python 3.12) |
| `numpy` | 2.5.0 |
| `torch` | 2.11.0+cu128 |
| `torch_cuda` | 12.8 |
| `torch_cuda_available` | True |
| `torch_gpu` | NVIDIA GeForce RTX 4060 |
| `sklearn` | nao encontrado (No module named 'sklearn') |
| `pandas` | nao encontrado (No module named 'pandas') |

## Projeto

| Item | Valor |
|---|---|
| `branch` | nao detectado |
| `commit` | nao detectado |
| `dirty_files` | 0 |
| `remote_origin` | nao detectado |
| `tool_versions` | erlang 27.3.4;elixir 1.18.4-otp-27; |

## PolyHok

| Item | Valor |
|---|---|
| `deps_poly_hok_commit` | e809eaafa1b9760aad54cb5ca2fde8f16173a705 |
| `deps_poly_hok_branch` | main |
| `deps_poly_hok_remote` | https://github.com/jrzanol/poly_hok.git |

## Arquivos brutos

- `raw/apt_install_environment_tools.txt`
- `raw/asdf_current.txt`
- `raw/cpu_lscpu.txt`
- `raw/cpu_procinfo.txt`
- `raw/disk_df.txt`
- `raw/disk_lsblk.txt`
- `raw/elixir_version.txt`
- `raw/erlang_otp.txt`
- `raw/gcc_version.txt`
- `raw/git_version.txt`
- `raw/gpp_version.txt`
- `raw/kernel_version.txt`
- `raw/make_version.txt`
- `raw/memory_free.txt`
- `raw/memory_meminfo.txt`
- `raw/mix_version.txt`
- `raw/nvcc_version.txt`
- `raw/nvidia_smi.txt`
- `raw/nvidia_smi_query.txt`
- `raw/os_release.txt`
- `raw/pci_lspci.txt`
- `raw/pip3_version.txt`
- `raw/project_backprop_env.txt`
- `raw/project_git_log.txt`
- `raw/project_git_status.txt`
- `raw/python3_version.txt`
- `raw/python_libraries.txt`
- `raw/uname.txt`
- `raw/windows_os.txt`
