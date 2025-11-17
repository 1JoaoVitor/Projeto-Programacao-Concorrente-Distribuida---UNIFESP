/*
 * kmeans_1d_cuda.cu
 * Implementação do K-means 1D com o passo de 'assignment' na GPU (CUDA)
 * e o passo de 'update' no Host (CPU), seguindo a "Opção A".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * Kernel de Assignment: 1 thread por ponto
 * Cada thread calcula a menor distância para seu ponto 'i' e escreve
 * o índice do cluster em 'assign_dev' e o erro em 'errors_dev'.
 */
__global__
void assignment_kernel(const double *X_dev, const double *C_dev,
                       int *assign_dev, double *errors_dev,
                       int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        double bestd = 1e300;
        int best = -1;

        for (int c = 0; c < K; c++) {
            double diff = X_dev[i] - C_dev[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }

        assign_dev[i] = best;
        errors_dev[i] = bestd;
    }
}


//funções para arquivos

static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir %s\n", path); exit(1); }
    int rows = 0; char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { only_ws = 0; break; }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R <= 0) { fprintf(stderr, "Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) { fprintf(stderr, "Sem memoria para %d linhas\n", R); exit(1); }
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir %s\n", path); free(A); exit(1); }
    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { only_ws = 0; break; }
        }
        if (only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok) { fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if (r > R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for (int i = 0; i < N; i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for (int c = 0; c < K; c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}


static void update_step_1d(const double *X, double *C, const int *assign, int N, int K) {
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if (!sum || !cnt) { fprintf(stderr, "Sem memoria no update\n"); exit(1); }
    for (int i = 0; i < N; i++) {
        int a = assign[i];
        sum[a] += X[i];
        cnt[a] += 1;
    }
    for (int c = 0; c < K; c++) {
        if (cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0];
    }
    free(sum); free(cnt);
}


// Struct para testes de uma vez só (similar ao teste runner da etapa passada)
typedef struct {
    double sse;
    double total_ms;
    double kernel_ms;
    double h2d_ms;
    double d2h_ms;
    double cpu_ms;
} TestResult;

/* Função K-means - CUDA */
static void kmeans_1d_cuda(const double *X_host, double *C_host, int *assign_host,
                           int N, int K, int max_iter, double eps,
                           int *iters_out, double *sse_out,
                           float *total_kernel_time_ms,
                           float *total_h2d_time_ms,
                           float *total_d2h_time_ms,
                           int blockSize)
{
    //Alocar memória no Host (CPU)
    double *errors_host = (double*)malloc((size_t)N * sizeof(double));
    if (!errors_host) { fprintf(stderr, "Sem memoria para errors_host\n"); exit(1); }

    //Alocar memória no Device (GPU)
    double *X_dev, *C_dev, *errors_dev;
    int *assign_dev;
    cudaMalloc((void**)&X_dev, (size_t)N * sizeof(double));
    cudaMalloc((void**)&C_dev, (size_t)K * sizeof(double));
    cudaMalloc((void**)&assign_dev, (size_t)N * sizeof(int));
    cudaMalloc((void**)&errors_dev, (size_t)N * sizeof(double));

    //Medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float event_time_ms;

    *total_kernel_time_ms = 0;
    *total_h2d_time_ms = 0;
    *total_d2h_time_ms = 0;

    // Cópia de dados do host para device
    cudaEventRecord(start);
    cudaMemcpy(X_dev, X_host, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&event_time_ms, start, stop);
    *total_h2d_time_ms += event_time_ms;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for (it = 0; it < max_iter; it++) {
        cudaEventRecord(start);
        cudaMemcpy(C_dev, C_host, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&event_time_ms, start, stop);
        *total_h2d_time_ms += event_time_ms;


        int gridSize = (N + blockSize - 1) / blockSize;

        cudaEventRecord(start);

        assignment_kernel<<<gridSize, blockSize>>>(X_dev, C_dev, assign_dev, errors_dev, N, K);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&event_time_ms, start, stop);
        *total_kernel_time_ms += event_time_ms;

        cudaEventRecord(start);
        cudaMemcpy(assign_host, assign_dev, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(errors_host, errors_dev, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&event_time_ms, start, stop);
        *total_d2h_time_ms += event_time_ms;

        sse = 0.0;
        for (int i = 0; i < N; i++) {
            sse += errors_host[i];
        }

        // Verificando convergência
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) { it++; break; }
        prev_sse = sse;

        update_step_1d(X_host, C_host, assign_host, N, K);
    }

    *iters_out = it;
    *sse_out = sse;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(X_dev);
    cudaFree(C_dev);
    cudaFree(assign_dev);
    cudaFree(errors_dev);
    free(errors_host);
}

TestResult run_single_test(int blockSize, int N, int K, const double *X_host, const double *C_initial_host, int max_iter, double eps) {

    // Aloca memória de host para esta execução
    double *C_host = (double*)malloc((size_t)K * sizeof(double));
    int *assign_host = (int*)malloc((size_t)N * sizeof(int));
    if (!C_host || !assign_host) {
        fprintf(stderr, "Erro de alocação no run_test\n");
        exit(1);
    }

    // Reseta os centróides para o estado inicial
    memcpy(C_host, C_initial_host, (size_t)K * sizeof(double));

    // Medição de tempo total
    struct timespec start_total, end_total;
    clock_gettime(CLOCK_MONOTONIC, &start_total);

    int iters = 0; double sse = 0.0;
    float kernel_time = 0, h2d_time = 0, d2h_time = 0;

    kmeans_1d_cuda(X_host, C_host, assign_host, N, K, max_iter, eps,
                   &iters, &sse, &kernel_time, &h2d_time, &d2h_time,
                   blockSize);

    clock_gettime(CLOCK_MONOTONIC, &end_total);
    double total_ms = (end_total.tv_sec - start_total.tv_sec) * 1000.0 +
                      (end_total.tv_nsec - start_total.tv_nsec) / 1000000.0;

    double cpu_time = total_ms - (kernel_time + h2d_time + d2h_time);

    // Libera memória desta execução
    free(C_host);
    free(assign_host);

    // Retorna a estrutura de resultados
    TestResult result = {sse, total_ms, kernel_time, h2d_time, d2h_time, cpu_time};
    return result;
}

// Função main fazendo os variados testes de uma vez só

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-6]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;

    printf("Lendo arquivos de entrada (pode demorar)... \n");
    int N = 0, K = 0;
    double *X_host = read_csv_1col(pathX, &N);
    double *C_initial_host = read_csv_1col(pathC, &K); // Carrega os centróides originais

    // Configurações dos testes
    int blockSizes_to_test[] = {128, 256, 512, 1024};
    int num_tests = sizeof(blockSizes_to_test) / sizeof(blockSizes_to_test[0]);
    const int NUM_RUNS = 3; // média de 3

    printf("--- Iniciando Testes de Bloco CUDA (N=%d, K=%d) ---\n", N, K);
    printf("--- Média de %d execuções por teste ---\n\n", NUM_RUNS);
    printf("+------------+---------------+---------------+---------------+--------------+---------------+----------+\n");
    printf("| Block Size | Tempo Total (ms)| Kernel (ms)   | H2D (ms)      | D2H (ms)     | CPU Ovhd (ms) | SSE Final|\n");
    printf("+------------+---------------+---------------+---------------+--------------+---------------+----------+\n");

    for (int i = 0; i < num_tests; i++) {
        int current_blockSize = blockSizes_to_test[i];

        double avg_total = 0, avg_kernel = 0, avg_h2d = 0, avg_d2h = 0, avg_cpu = 0;
        double last_sse = 0;

        for (int run = 0; run < NUM_RUNS; run++) {
            TestResult r = run_single_test(current_blockSize, N, K, X_host, C_initial_host, max_iter, eps);

            avg_total += r.total_ms;
            avg_kernel += r.kernel_ms;
            avg_h2d += r.h2d_ms;
            avg_d2h += r.d2h_ms;
            avg_cpu += r.cpu_ms;
            last_sse = r.sse;
        }


        avg_total /= NUM_RUNS;
        avg_kernel /= NUM_RUNS;
        avg_h2d /= NUM_RUNS;
        avg_d2h /= NUM_RUNS;
        avg_cpu /= NUM_RUNS;


        printf("| %-10d | %-13.1f | %-13.1f | %-13.1f | %-12.1f | %-13.1f | %.6f |\n",
               current_blockSize, avg_total, avg_kernel, avg_h2d, avg_d2h, avg_cpu, last_sse);
    }

    printf("+------------+---------------+---------------+---------------+--------------+---------------+----------+\n");

    free(X_host);
    free(C_initial_host);
    return 0;
}
