# K-means 1D Distribuído com MPI - Etapa 3

Este diretório contém a implementação do algoritmo K-means utilizando **Message Passing Interface (MPI)**, focado em arquiteturas de memória distribuída (clusters).

## Estrutura do Código

A implementação (`kmeans_1d_mpi.c`) segue o modelo Mestre-Escravo (SPMD):

1.  **Rank 0 (Mestre):** Lê os arquivos CSV e distribui os dados.
2.  **Distribuição:** Utiliza `MPI_Scatterv` para dividir os pontos igualmente (com tratamento de sobras) entre todos os processos.
3.  **Processamento:**
    * Cada processo calcula distâncias e somas parciais localmente.
    * **Sincronização:** Utiliza `MPI_Allreduce` para somar as médias parciais e atualizar os centróides globalmente em cada iteração.
    * Não há *broadcast* redundante de centróides dentro do loop (otimização).
4.  **Coleta:** O Rank 0 recolhe os resultados finais com `MPI_Gatherv` e salva no disco.

## Requisitos

* Compilador MPI (`mpicc`), como OpenMPI ou MPICH.
* Executor MPI (`mpirun` ou `mpiexec`).

## Compilação

```bash
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm