#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// I/O apenas para Rank 0
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int rows = 0; char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++)
            if (*p != ' ' && *p != '\t' && *p != '\n') { only_ws = 0; break; }
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
        for (char *p = line; *p; p++)
            if (*p != ' ' && *p != '\t' && *p != '\n') { only_ws = 0; break; }
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


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("Uso: mpirun -np <P> %s dados.csv centroides.csv [max_iter] [epsilon] [--save-output]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int save_output = 0;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--save-output") == 0) save_output = 1;
    }

    // leitura de Dados - rank 0
    int N_global = 0, K = 0;
    double *X_global = NULL;
    double *C = NULL;

    double t_read_start = MPI_Wtime();
    if (rank == 0) {
        printf("=== K-means 1D MPI ===\n");
        printf("Processos: %d\n", size);
        X_global = read_csv_1col(argv[1], &N_global);
        C = read_csv_1col(argv[2], &K);
        if (!X_global || !C) {
            fprintf(stderr, "ERRO: Falha na leitura dos arquivos.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Dados: N=%d pontos, K=%d clusters\n", N_global, K);
    }
    double t_read_end = MPI_Wtime();

    //broadcast
    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) C = (double*)malloc(K * sizeof(double));
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // scatterv
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));

    int base_count = N_global / size;
    int remainder = N_global % size;

    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int my_N = sendcounts[rank];
    double *my_X = (double*)malloc(my_N * sizeof(double));
    int *my_assign = (int*)malloc(my_N * sizeof(int));

    double t_scatter_start = MPI_Wtime();
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE,
                 my_X, my_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();

    // k-means de fato
    double prev_sse = DBL_MAX;
    double global_sse = 0.0;
    int it;

    double *local_sum = (double*)malloc(K * sizeof(double));
    int *local_cnt = (int*)malloc(K * sizeof(int));
    double *global_sum = (double*)malloc(K * sizeof(double));
    int *global_cnt = (int*)malloc(K * sizeof(int));

    double t_compute_start = MPI_Wtime();

    for (it = 0; it < max_iter; it++) {
        double local_sse = 0.0;
        memset(local_sum, 0, K * sizeof(double));
        memset(local_cnt, 0, K * sizeof(int));

        for (int i = 0; i < my_N; i++) {
            double bestd = DBL_MAX;
            int best = 0;
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

        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int c = 0; c < K; c++) {
            if (global_cnt[c] > 0) C[c] = global_sum[c] / global_cnt[c];
        }

        double rel_change = fabs(global_sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel_change < eps) { it++; break; }
        prev_sse = global_sse;
    }

    double t_compute_end = MPI_Wtime();

    // gatherv
    int *assign_global = NULL;
    if (rank == 0) {
        assign_global = (int*)malloc(N_global * sizeof(int));
    }

    double t_gather_start = MPI_Wtime();
    MPI_Gatherv(my_assign, my_N, MPI_INT,
                assign_global, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    double t_gather_end = MPI_Wtime();

    // resultados
    if (rank == 0) {
        double t_read = t_read_end - t_read_start;
        double t_scatter = t_scatter_end - t_scatter_start;
        double t_compute = t_compute_end - t_compute_start;
        double t_gather = t_gather_end - t_gather_start;
        double t_total = t_read + t_scatter + t_compute + t_gather;

        printf("\n=== RESULTADOS ===\n");
        printf("Iterações: %d\n", it);
        printf("SSE final: %.6f\n", global_sse);
        printf("\n=== TEMPOS (ms) ===\n");
        printf("Leitura:       %8.2f ms\n", t_read * 1000.0);
        printf("Scatter:       %8.2f ms\n", t_scatter * 1000.0);
        printf("Computação:    %8.2f ms\n", t_compute * 1000.0);
        printf("Gather:        %8.2f ms\n", t_gather * 1000.0);
        printf("TOTAL:         %8.2f ms\n", t_total * 1000.0);

        printf("\n=== CSV (processos,tempo_ms,sse) ===\n");
        printf("%d,%.2f,%.6f\n", size, t_compute * 1000.0, global_sse);

        if (save_output) {
            write_csv_int("assign_mpi.csv", assign_global, N_global);
            write_csv_double("centroids_mpi.csv", C, K);
            printf("\nArquivos salvos: assign_mpi.csv, centroids_mpi.csv\n");
        }

        free(X_global);
        free(assign_global);
    }

    free(my_X);
    free(my_assign);
    free(C);
    free(sendcounts);
    free(displs);
    free(local_sum);
    free(local_cnt);
    free(global_sum);
    free(global_cnt);

    MPI_Finalize();
    return 0;
}
