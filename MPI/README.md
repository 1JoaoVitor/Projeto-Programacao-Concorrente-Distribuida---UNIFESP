# K-means 1D Distribuído com MPI - Etapa 3

Este diretório contém a implementação do algoritmo K-means para arquiteturas de **memória distribuída** utilizando o padrão MPI (Message Passing Interface).

A solução segue o modelo **SPMD** (Single Program, Multiple Data) com estratégia Mestre-Escravo otimizada para minimizar tráfego de rede.

## Estrutura da Implementação

* **Rank 0 (Mestre):** Único responsável por ler os arquivos de entrada do disco.
* **Distribuição (`MPI_Scatterv`):** Os dados são particionados e enviados para a memória local de cada processo worker.
* **Sincronização (`MPI_Allreduce`):** A atualização dos centróides é feita através de uma redução global a cada iteração, permitindo que todos os processos tenham a visão atualizada do modelo sem necessidade de *broadcast* explícito pelo mestre.
* **Coleta (`MPI_Gatherv`):** Ao final, os resultados (vetor de atribuição) são recolhidos pelo Rank 0 para salvamento.

## Requisitos

* **Compilador MPI:** `mpicc` (OpenMPI ou MPICH).
* **Executor MPI:** `mpirun`.
* **Cluster (Opcional):** Para execução distribuída real, são necessárias múltiplas máquinas conectadas em rede com SSH configurado sem senha (chaves pública/privada).

## Compilação

Estando na raiz do projeto ou dentro da pasta `MPI/`, utilize o `mpicc`.

```bash
# Exemplo compilando de dentro da pasta MPI/
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
```

## Execução
O programa aceita os seguintes argumentos:

```
mpirun -np <N_PROCS> ./kmeans_1d_mpi <dados.csv> <centroides.csv> [max_iter] [eps] [--sa
``` 
### 1. Execução Local
Para testar na sua própria máquina (usando múltiplos núcleos):

```
mpirun -np 4 ./kmeans_1d_mpi ../Dados/dados.csv ../Dados/centroides_iniciais.csv 50 1e-6
```
### 2. Execução Distribuída (Cluster)
Para rodar em múltiplas máquinas, você precisa de um arquivo de hosts (hosts ou hostfile) listando os IPs e a quantidade de "slots" (processos) por máquina.

Exemplo:

```
localhost slots=2
192.135.13.41 slots=2
```

Pré-requisitos do Cluster:

- O executável kmeans_1d_mpi deve existir em todas as máquinas, exatamente no mesmo caminho absoluto (ex: /home/aluno/MPI/kmeans_1d_mpi).
- O SSH sem senha deve estar configurado entre a máquina mestre e as escravas.

## Saída Esperada
O programa imprimirá no terminal do Mestre:

- Tempos detalhados: Leitura, Scatter (Distribuição), Computação e Gather (Coleta).

- O SSE Final para validação.

- Uma linha formatada em CSV para facilitar scripts de benchmarking.


Link do colab utilizado: https://colab.research.google.com/drive/1uKopFKgd2t1gyXyJApdaCIirqrqfTnsa?usp=sharing






