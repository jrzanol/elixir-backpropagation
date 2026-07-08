# Analise de complexidade de desenvolvimento

Este relatorio compara CUDA/NIF e PolyHok por metricas quantitativas e qualitativas baseadas em evidencias do codigo. O objetivo nao e declarar uma implementacao melhor em todos os aspectos, mas separar tamanho, esforco manual, complexidade conceitual, testabilidade, compilacao e manutencao.

## Como as metricas sao feitas e para que servem

| Metrica | Para que serve | Como e feita | Base teorica |
|---|---|---|---|
| Arquivos e linguagens | Mede a superficie estrutural da implementacao e o custo de alternar entre linguagens/ferramentas. | Conta arquivos, extensoes e numero de linguagens no escopo de cada implementacao. | Usado como evidencia de esforco manual e superficie de modificacao. |
| Linhas de codigo e caracteres | Aproxima tamanho e esforco manual de escrita/leitura. | Conta linhas fisicas, linhas nao vazias sem comentarios e caracteres de codigo. | LOC/NOC aparecem em estudos de produtividade GPU; nao devem ser usados isoladamente. |
| Complexidade ciclomatica aproximada | Aproxima numero de caminhos logicos e esforco de teste. | Conta decisoes sintaticas por arquivo, como if, case, for, while e operadores booleanos. | McCabe/NIST relaciona complexidade ciclomatica a caminhos de teste. |
| Tamanho e argumentos de funcoes/kernels | Mede granularidade, acoplamento e dificuldade de revisao. | Extrai funcoes/kernels e calcula tamanho medio/maximo e numero medio/maximo de argumentos. | Funcoes longas e assinaturas extensas aumentam custo de compreensao e modificacao. |
| Gerenciamento CPU/GPU | Mede o esforco explicito de memoria, transferencia, sincronizacao e lancamento de kernels. | Conta cudaMalloc/cudaMemcpy/cudaMemset/cudaFree, new_gnx/get_gnx, spawn_st, launches e sincronizacoes. | Malik et al. separam memoria, comunicacao e execucao GPU como esforco conceitual. |
| Observabilidade | Mede facilidade de ver estado interno, tempos e resultados durante testes/depuracao. | Conta snapshots, Profiler, timings, prints e checks de erro. | Testabilidade depende de observabilidade e depurabilidade. |
| Controlabilidade | Mede facilidade de reproduzir cenarios controlando seed, dataset, batch, epochs e debug. | Conta leituras de variaveis de ambiente e parametros BACKPROP_*. | Testabilidade tambem depende de controlabilidade. |
| Tempo de compilacao repetido | Mede custo pratico de feedback de desenvolvimento. | Executa os scripts de compilacao varias vezes e calcula media, mediana e desvio padrao. | Tempo de feedback influencia produtividade e ciclo de depuracao. |
| Churn historico | Aproxima esforco real de evolucao/manutencao quando ha historico git. | Conta commits, linhas adicionadas e removidas por implementacao. | Churn e usado como proxy de manutencao/retrabalho, mas depende da qualidade do historico. |

## Base teorica resumida

- Malik et al. comparam produtividade GPU separando LOC/NOC, memoria, comunicacao, setup de kernel e execucao GPU.
- Kosar et al. usam dimensoes cognitivas para comparar DSL e linguagem geral em tarefas de aprendizado, compreensao e evolucao.
- McCabe/NIST relaciona complexidade ciclomatica com quantidade de caminhos de teste.
- Trabalhos de testabilidade destacam observabilidade e controlabilidade como propriedades centrais para testar e depurar.
- LOC e caracteres sao evidencias de tamanho/esforco, mas nao devem ser usados isoladamente.

## Referencias usadas

- Malik et al., `Productivity of GPUs under different programming paradigms`: https://www.eecg.utoronto.ca/~amza/ece1747h/papers/cpe1860.pdf
- Kosar et al., `Comparing General-Purpose and Domain-Specific Languages`: https://www.comsis.org/pdf.php?id=226-0911
- McCabe/NIST, `Structured Testing`: https://www.eng.auburn.edu/~kchang/comp6710/readings/Integration.Testing.McCabe.NIST.pdf
- Sharma et al., `Software Testability`: https://www.tusharma.in/preprints/EMSE2023_Testability.pdf
- Barb et al., `A statistical study of the relevance of lines of code measures`: https://www.researchgate.net/publication/267761547_A_statistical_study_of_the_relevance_of_lines_of_code_measures_in_software_projects

## Resultado consolidado

| Criterio | Tipo | CUDA/NIF | PolyHok | Metodo |
|---|---|---|---|---|
| Quantidade de arquivos | quantitativo | 7 | 3 | Arquivos especificos da implementacao. |
| Linguagens utilizadas | quantitativo | C++: 1, C/C++ header: 2, CUDA: 2, Elixir: 2 | Elixir: 3 | Classificacao pela extensao dos arquivos. |
| Linhas fisicas | quantitativo | 1196 | 715 | Todas as linhas dos arquivos especificos. |
| Linhas de codigo | quantitativo | 985 | 602 | Linhas nao vazias, excluindo linhas somente de comentario. |
| Caracteres de codigo | quantitativo | 30854 | 15174 | Soma dos caracteres em linhas de codigo. |
| Complexidade ciclomatica aproximada | quantitativo | 174 | 73 | Contagem heuristica de decisoes sintaticas. |
| Funcoes/kernels | quantitativo | 60 | 44 | Extracao heuristica de funcoes, kernels e funcoes DSL. |
| Tamanho medio de funcao | quantitativo | 15.62 | 15.86 | Media de linhas por funcao/kernel. |
| Maior funcao/kernel | quantitativo | 95 | 123 | Maior bloco funcional detectado. |
| Media de argumentos por funcao | quantitativo | 2.48 | 2.05 | Media de parametros por funcao/kernel. |
| Maior assinatura de funcao | quantitativo | 8 | 8 | Maior numero de argumentos detectado. |
| Kernels declarados | quantitativo | 3 | 4 | CUDA __global__ vs PolyHok defk. |
| Lancamentos GPU | quantitativo | 3 | 4 | CUDA <<<...>>> vs PolyHok.spawn_st. |
| Argumentos medios de kernel/lancamento | quantitativo | 6.00 | 5.40 | Assinaturas e chamadas de kernel. |
| Maior kernel/lancamento | quantitativo | 8 | 13 | Maior quantidade de argumentos em definicao/chamada. |
| Gerenciamento de memoria GPU | quantitativo | 29 | 17 | cudaMalloc/cudaFree/cudaMemcpy/cudaMemset vs PolyHok.new_gnx. |
| Transferencias CPU/GPU | quantitativo | 7 | 2 | cudaMemcpy vs PolyHok.get_gnx; new_gnx e considerado em memoria. |
| Sincronizacoes explicitas | quantitativo | 2 | 2 | cudaDeviceSynchronize vs PolyHok.synchronize. |
| Checks de erro/assincronia | quantitativo | 15 | n/a | cudaGetLastError/CudaOk; PolyHok encapsula parte desses checks. |
| Observabilidade | quantitativo | 38 | 15 | Snapshots, timers, prints e checks de erro. |
| Controlabilidade da implementacao | quantitativo | 7 | 6 | Variaveis de ambiente e knobs BACKPROP_* no codigo especifico. |
| Controlabilidade do projeto | quantitativo | 39 | 39 | Parametros compartilhados de seed, dataset, epochs, batch size e debug. |
| Tempo medio de compilacao | quantitativo | 8178.422 ms | 12683.667 ms | Media de execucoes repetidas dos scripts oficiais. |
| Tempo mediano de compilacao | quantitativo | 8175.360 ms | 12676.294 ms | Mediana de execucoes repetidas dos scripts oficiais. |
| Desvio da compilacao | quantitativo | 208.236 ms | 109.843 ms | Desvio padrao amostral das compilacoes. |
| Churn - commits | quantitativo | 0 | 0 | Historico git nos arquivos especificos. |
| Churn - linhas adicionadas | quantitativo | 0 | 0 | Soma de insercoes em git log --numstat. |
| Churn - linhas removidas | quantitativo | 0 | 0 | Soma de remocoes em git log --numstat. |
| Necessidade de codigo C/CUDA | qualitativo | Obrigatoria: 3 kernels CUDA e 6 funcoes NIF exportadas. | Nao no codigo da aplicacao: kernels ficam em Elixir/DSL; C/CUDA permanece na dependencia PolyHok. | Inspecao de linguagens e fronteiras. |
| Superficie de modificacao de kernel | qualitativo | Maior: envolve kernel CUDA, wrapper/launcher, header, C++ NIF e modulo Elixir. | Menor: kernel e chamada ficam em Elixir/DSL, mas ainda dependem de tipos estaticos e recompilacao. | Numero de linguagens, arquivos e fronteiras que precisam permanecer coerentes. |
| Restricoes estaticas PolyHok | qualitativo | Nao se aplica diretamente; CUDA ainda compila para arquitetura alvo. | 6 deft, 3 buffers fixos de 512 e tipos aceitos pela DSL. | Inspecao de deft/defk, arrays fixos e logs de arquitetura. |

## Arquivos gerados

- `development_complexity.csv`: tabela consolidada.
- `development_complexity_files.csv`: metricas por arquivo.
- `development_complexity_functions.csv`: funcoes/kernels detectados.
- `development_complexity_kernels.csv`: definicoes e lancamentos GPU.
- `development_complexity_compile_summary.csv`: media, mediana e desvio da compilacao.
- `development_complexity_metric_explanations.csv`: explicacao tabular das metricas.
- `development_complexity.tex`: tabela para LaTeX.
