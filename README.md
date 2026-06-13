# Backpropagation em GPU: CUDA/NIF e PolyHok

Implementacao e comparacao do treinamento de uma MLP em GPU usando CUDA/NIF e a DSL PolyHok.

## Requisitos

- Ubuntu ou WSL2
- GPU NVIDIA, driver e CUDA Toolkit (`nvcc`)
- Credenciais Kaggle em `.kaggle/kaggle.json` ou `~/.kaggle/kaggle.json`

## Preparacao

```bash
chmod +x ./*.sh
./0_InstallDeps.sh
```

Para os testes com PyTorch:

```bash
./0_InstallPyTorch.sh
```

## Dataset HIGGS

```bash
./0_DownloadDataset.sh erikbiswas/higgs-uci-dataset HIGGS.csv
./1_CompileDataset.sh HIGGS.csv 0 32768 0.5 no_header
```

O download extrai `.csv.gz` automaticamente. A segunda etapa processa o CSV em partes, normaliza os dados e gera batches binarios em `scripts/prepared_dataset`.

Topologia utilizada com o HIGGS: `[28, 256, 128, 1]`.

## Compilacao e execucao

```bash
./1_CompileCUDA.sh
./1_CompilePolyHok.sh

./2_RunCUDA.sh
./2_RunPolyHok.sh
./2_RunPyTorch.sh
```

## Testes

```bash
./4_TestFunctionalCorrectness.sh
./5_TestEquivalence.sh
./6_TestPerformance.sh
./7_AnalyzeDevelopmentComplexity.sh
```

Equivalencia adicional com PyTorch:

```bash
./3_VerifyEquivalence.sh
```

Monitoramento de recursos:

```bash
RESOURCE_SAMPLE_INTERVAL=0.2 ./3_ReportUsage.sh ./2_RunCUDA.sh
```

Os resultados CSV, Markdown e LaTeX sao gravados em `reports/`.
