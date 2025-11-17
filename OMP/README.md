# Projeto PCD: K-means 1D Paralelo com OpenMP

Este repositório contém a implementação paralela do algoritmo K-means unidimensional utilizando OpenMP, como parte da disciplina de Programação Concorrente e Distribuída. Inclui o código K-means principal e um orquestrador de testes para análise de desempenho.

## Estrutura do Projeto

* `kmeans_1d_omp.c`: Código fonte da implementação K-means paralela com OpenMP.
* `test_runner.c`: Código fonte do orquestrador de testes.
* `dados.csv`: Arquivo de exemplo com os pontos de dados (N linhas).
* `centroides_iniciais.csv`: Arquivo de exemplo com os centróides iniciais (K linhas).

## Geração dos Dados de Teste (Opcional)

**Requisitos:**
* Python 3
* Biblioteca NumPy (`pip install numpy`)

**Uso:**
1.  **Modifique os Parâmetros (Opcional):** Abra o arquivo `gerar_dados.py` e ajuste as variáveis `N` (número de pontos) e `K` (número de clusters) no início do script conforme desejado. Você também pode ajustar os `CENTROIDES_REAIS` e o `DESVIO_PADRAO`.
2.  **Execute o Script:** No terminal, na pasta do projeto, execute:
    ```bash
    python3 gerar_dados.py
    ```
    Isso irá gerar (ou sobrescrever) os arquivos `dados.csv` e `centroides_iniciais.csv` com os novos parâmetros. Lembre-se que o script utiliza uma semente (`np.random.seed(42)`) para garantir a reprodutibilidade dos dados gerados.

## Compilação

Para compilar os programas, utilize um compilador C com suporte a OpenMP (exemplo o GCC).

1.  **Compilar o Executável K-means:**
    ```bash
    gcc -O2 -fopenmp -std=c99 OMP/kmeans_1d_omp.c -o OMP/kmeans_1d_omp -lm
    ```
    * `-O2`: Nível de otimização.
    * `-fopenmp`: Habilita o suporte OpenMP.
    * `-std=c99`: Define o padrão C99.
    * `-lm`: Linka a biblioteca matemática.

2.  **Compilar o Orquestrador de Testes:**
    ```bash
    gcc -O2 OMP/test_runner.c -o OMP/test_runner
    ```

## Execução

O orquestrador de testes (`test_runner`) automatiza a execução para análise de desempenho. Certifique-se de que os arquivos `dados.csv` e `centroides_iniciais.csv` estejam presentes na mesma pasta.

1.  **Executar Teste de Escalonamento (Threads):**
    Roda o K-means com 1, 2, 4, 8 e 16 threads (média de 3 execuções) e imprime a tabela de tempos e SSE final.
    ```bash
    ./OMP/test_runner scaling
    ```

2.  **Executar Teste de Agendamento (Schedule):**
    Roda o K-means com 4, 8 e 16 threads, testando as políticas `static` e `dynamic` com diferentes `chunk sizes` (média de 3 execuções).
    ```bash
    ./OMP/test_runner schedule
    ```

3.  **Executar o K-means Diretamente (Opcional):**
    Você também pode executar o `kmeans_1d_omp` diretamente, controlando o número de threads com a variável de ambiente `OMP_NUM_THREADS` e passando os parâmetros de schedule.
    ```bash
    # Exemplo: Rodar com 8 threads, schedule static, chunk default
    export OMP_NUM_THREADS=8
    ./OMP/kmeans_1d_omp dados.csv centroides_iniciais.csv 50 1e-6 static 0 
    ```
