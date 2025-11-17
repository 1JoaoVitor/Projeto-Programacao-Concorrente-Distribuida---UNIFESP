# K-means 1D com CUDA (GPU) - Etapa 2

Este diretório contém a implementação paralela do algoritmo K-means unidimensional utilizando a arquitetura CUDA para execução em GPUs NVIDIA.

Esta implementação segue o design **"Opção A"**:
* **Assignment (Atribuição):** Executado em paralelo na GPU (1 thread por ponto).
* **Update (Atualização):** Executado serialmente na CPU (Host), após copiar os resultados de volta.

## Estrutura

* `Kmeans_1d_cuda.cu`: Código fonte híbrido (CPU/CUDA) que atua também como *Test Harness* (Orquestrador de Testes).

## Requisitos

* **Hardware:** GPU NVIDIA (Testes realizados em Tesla T4).
* **Software:** NVIDIA CUDA Toolkit (compilador `nvcc`).

## Compilação

Para compilar o código, utilize o `nvcc`.

### Em ambiente local (com GPU NVIDIA genérica):
```bash
nvcc -O2 Kmeans_1d_cuda.cu -o kmeans_1d_cuda