# K-means 1D com CUDA (GPU) - Etapa 2

Este diretório contém a implementação paralela do algoritmo K-means unidimensional utilizando a arquitetura CUDA para execução em GPUs NVIDIA.

Esta implementação segue o design **"Opção A"**:
* **Assignment (Atribuição):** Executado em paralelo na GPU (1 thread por ponto).
* **Update (Atualização):** Executado serialmente na CPU (Host), após copiar os resultados de volta.

## Estrutura

* `Kmeans_1d_cuda.cu`: Código fonte híbrido (CPU/CUDA) que atua também como código de testes.

## Requisitos

* **Hardware:** GPU NVIDIA (Testes realizados em Tesla T4).
* **Software:** NVIDIA CUDA Toolkit (compilador `nvcc`).

## Compilação

Para compilar o código, utilize o `nvcc`.

### Em ambiente local (com GPU NVIDIA genérica):
```bash
nvcc -O2 Kmeans_1d_cuda.cu -o kmeans_1d_cuda
```

### No Google Colab (GPU Tesla T4):

Devido a incompatibilidades de toolchain no ambiente Colab, recomenda-se compilar especificando a arquitetura da GPU (sm_75) para evitar erros de execução JIT:

```bash
nvcc -O2 -gencode arch=compute_75,code=sm_75 Kmeans_1d_cuda.cu -o kmeans_1d_cuda
```

## Execução

O programa foi modificado para funcionar como uma bateria de testes automatizada. Ele ignora uma execução única e, em vez disso, roda o algoritmo múltiplas vezes variando o Tamanho do Bloco (Block Size) para encontrar a configuração ideal.

```bash
./kmeans_1d_cuda <dados.csv> <centroides.csv> <max_iter> <eps>
```

Por exemplo: 
```bash
./kmeans_1d_cuda ../Dados/dados.csv ../Dados/centroides_iniciais.csv 50 1e-6
```

### Saída esperada

O programa exibirá uma tabela contendo a média de 3 execuções para diferentes tamanhos de bloco (ex: 128, 256, 512, 1024), detalhando:

Tempo Total: Tempo de parede completo.

Kernel: Tempo de processamento na GPU.

H2D/D2H: Tempos de transferência de memória.

CPU Ovhd: Overhead do passo de atualização no Host.

SSE Final: Para validação de corretude.