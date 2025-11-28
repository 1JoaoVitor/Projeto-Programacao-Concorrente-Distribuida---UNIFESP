
/*
 * kmeans_1d_mpi.c
 * Implementação K-means 1D usando MPI (Memória Distribuída).
 * Estratégia: Rank 0 lê dados, distribui (Scatter), e recolhe (Gather).
 * Reduções globais usam MPI_Allreduce.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// --- Funções Auxiliares (I/O apenas para Rank 0) ---
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int rows = 0; char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) if (*p != ' ' && *p != '\t' && *p != '\n') { only_ws = 0; break; }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R <= 0) return NULL;
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) return NULL;
    FILE *f = fopen(path, "r");
    if (!f) { free(A); return NULL; }
    int r = 0; char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) if (*p != ' ' && *p != '\t' && *p != '\n') { only_ws = 0; break; }
        if (only_ws) continue;
        A[r++] = atof(strtok(line, ",; \t"));
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_csv_int(const char *path, const int *data, int N) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < N; i++) fprintf(f, "%d\n", data[i]);
    fclose(f);
}

static void write_csv_double(const char *path, const double *data, int N) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < N; i++) fprintf(f, "%.6f\n", data[i]);
    fclose(f);
}

// --- Função Principal ---
int main(int argc, char **argv) {
    // 1. Inicialização MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Quem sou eu?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Quantos somos?

    // Argumentos
    if (argc < 3) {
        if (rank == 0) printf("Uso: mpirun -np <P> %s dados.csv centroides.csv [args]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;

    // --- 2. Leitura de Dados (Apenas Rank 0) ---
    int N_global = 0, K = 0;
    double *X_global = NULL;
    double *C = NULL; // Todos terão C completo

    if (rank == 0) {
        printf("MPI Master (Rank 0) lendo arquivos...\n");
        X_global = read_csv_1col(argv[1], &N_global);
        C = read_csv_1col(argv[2], &K);
        if (!X_global || !C) {
            fprintf(stderr, "Erro na leitura.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // --- 3. Broadcast de Metadados (N e K) ---
    // Rank 0 avisa a todos qual é o tamanho de N e K
    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Se não sou rank 0, aloco C (Rank 0 já alocou na leitura)
    if (rank != 0) C = (double*)malloc(K * sizeof(double));

    // Broadcast dos Centróides Iniciais (Todos precisam começar iguais)
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- 4. Distribuição de Dados (Scatterv) ---
    // Precisamos dividir N pontos entre 'size' processos.
    // Como a divisão pode não ser exata, usamos arrays de contagem (sendcounts) e deslocamento (displs).

    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));

    int base_count = N_global / size;
    int remainder = N_global % size;

    // Calcula quantos pontos cada processo vai receber
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int my_N = sendcounts[rank]; // Quantos pontos EU vou processar
    double *my_X = (double*)malloc(my_N * sizeof(double));
    int *my_assign = (int*)malloc(my_N * sizeof(int));

    // Scatterv: Rank 0 envia pedaços diferentes para cada um
    // X_global -> my_X
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE,
                 my_X, my_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Rank 0 pode liberar a memória global gigante de X agora (opcional, mas bom pra economizar RAM)
    // Mas vamos manter para o Gather no final, se a RAM permitir.

    // --- 5. Loop K-means Distribuído ---
    double prev_sse = 1e300;
    double global_sse = 0.0;
    int it;

    double t0 = MPI_Wtime(); // Cronômetro MPI

    for (it = 0; it < max_iter; it++) {
        // A. Broadcast C atualizado (Garantia que todos têm o mesmo C)
        // Na prática, como usamos Allreduce depois, todos já têm os dados para atualizar C,
        // mas um Bcast explícito do Rank 0 é seguro contra erros de ponto flutuante divergentes.
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // B. Assignment Local
        double local_sse = 0.0;

        // Estruturas para acumular soma e contagem locais
        double *local_sum = (double*)calloc(K, sizeof(double));
        int *local_cnt = (int*)calloc(K, sizeof(int));

        for (int i = 0; i < my_N; i++) {
            double bestd = 1e300;
            int best = -1;
            for (int c = 0; c < K; c++) {
                double diff = my_X[i] - C[c];
                double d = diff * diff;
                if (d < bestd) { bestd = d; best = c; }
            }
            my_assign[i] = best;
            local_sse += bestd;

            local_sum[best] += my_X[i];
            local_cnt[best]++;
        }

        // C. Redução Global (MPI_Allreduce)
        // Somar SSE local de todos -> SSE global em todos
        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Somar arrays de soma e contagem de todos -> global em todos
        // Como 'C' é pequeno, isso é rápido.
        double *global_sum = (double*)malloc(K * sizeof(double));
        int *global_cnt = (int*)malloc(K * sizeof(int));

        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // D. Atualizar Centróides (Cada um faz o seu, já que todos têm os dados globais)
        // Isso evita ter que o Rank 0 calcule e faça outro Broadcast.
        for (int c = 0; c < K; c++) {
            if (global_cnt[c] > 0) C[c] = global_sum[c] / global_cnt[c];
            // Se vazio, mantém (ou podia adotar estratégia de reset, mas precisa ser determinístico)
        }

        free(local_sum); free(local_cnt);
        free(global_sum); free(global_cnt);

        // E. Checar Convergência
        double rel = fabs(global_sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) { it++; break; }
        prev_sse = global_sse;
    }

    double t1 = MPI_Wtime();

    // --- 6. Recolher Resultados (Gatherv) ---
    // Rank 0 recolhe o vetor 'assign' de todos para salvar no arquivo
    int *assign_global = NULL;
    if (rank == 0) {
        assign_global = (int*)malloc(N_global * sizeof(int));
    }

    MPI_Gatherv(my_assign, my_N, MPI_INT,
                assign_global, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // --- 7. Finalização e Saída ---
    if (rank == 0) {
        double ms = (t1 - t0) * 1000.0;
        printf("--- MPI (%d processos) ---\n", size);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", it, global_sse, ms);

        // Saída CSV simples para o seu test_runner ler, se for adaptar
        // printf("%d,%.1f,%.6f\n", size, ms, global_sse);

        // Salvar arquivos (Opcional, pode ser lento com 50M)
        // write_csv_int("assign_mpi.csv", assign_global, N_global);
        // write_csv_double("centroids_mpi.csv", C, K);

        free(X_global);
        free(assign_global);
    }

    free(my_X);
    free(my_assign);
    free(C);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}