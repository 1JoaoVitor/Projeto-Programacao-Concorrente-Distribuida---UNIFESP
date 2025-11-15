#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 

void run_scaling_tests(const char* executable, const char* data_file, const char* centroids_file);
void run_schedule_tests(const char* executable, const char* data_file, const char* centroids_file);

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Uso: %s [test_name]\n", argv[0]);
        printf("Nomes de teste disponíveis:\n");
        printf("  scaling       - Testa o escalonamento com threads (1, 2, 4, 8, 16)\n");
        printf("  schedule      - Testa afinamento de schedule e chunk size\n");
        return 1;
    }

    const char* test_name = argv[1];
    const char* data_file = "Dados/dados.csv";
    const char* centroids_file = "Dados/centroides_iniciais.csv";
    const char* executable = "./OMP/kmeans_1d_omp";

    if (strcmp(test_name, "scaling") == 0) {
        run_scaling_tests(executable, data_file, centroids_file);
    } else if (strcmp(test_name, "schedule") == 0) {
        run_schedule_tests(executable, data_file, centroids_file);
    } else {
        fprintf(stderr, "Erro: Nome de teste '%s' é inválido.\n", test_name);
    }
    
    return 0;
}

void run_scaling_tests(const char* executable, const char* data_file, const char* centroids_file) {
    int threads_to_test[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(threads_to_test) / sizeof(threads_to_test[0]);
    const int NUM_RUNS_PER_TEST = 3;

    printf("\n--- Teste de Escalonamento de Threads ---\n");
    printf("Média de %d execuções por teste.\n\n", NUM_RUNS_PER_TEST);
    printf("+---------+--------------------+------------------+\n");
    printf("| Threads | Tempo Médio (ms)   | SSE Final        |\n");
    printf("+---------+--------------------+------------------+\n");

    for (int i = 0; i < num_tests; i++) {
        int num_threads = threads_to_test[i];
        double total_ms = 0.0, final_sse = 0.0;
        for (int run = 0; run < NUM_RUNS_PER_TEST; run++) {
            char command[512];
            sprintf(command, "OMP_NUM_THREADS=%d %s %s %s", num_threads, executable, data_file, centroids_file);
            FILE *pipe = popen(command, "r");
            if (!pipe) { fprintf(stderr, "Erro ao executar o comando!\n"); exit(1); }
            int threads_out; double ms_out, sse_out;
            if (fscanf(pipe, "%d,%lf,%lf", &threads_out, &ms_out, &sse_out) == 3) {
                total_ms += ms_out;
                final_sse = sse_out;
            } else { fprintf(stderr, "Erro ao ler saída para %d threads.\n", num_threads); exit(1); }
            pclose(pipe);
        }
        printf("| %-7d | %-18.1f | %-16.6f |\n", num_threads, total_ms / NUM_RUNS_PER_TEST, final_sse);
    }
    printf("+---------+--------------------+------------------+\n");
}


void run_schedule_tests(const char* executable, const char* data_file, const char* centroids_file) {
    int threads_to_test[] = {4, 8, 16};
    const char* schedule_names[] = { "static", "dynamic" };
    
    omp_sched_t schedules_to_test[] = { omp_sched_static, omp_sched_dynamic };
    
    int chunk_sizes[] = {0, 1, 65536}; // 0 = default

    int num_thread_tests = sizeof(threads_to_test) / sizeof(threads_to_test[0]);
    int num_schedule_tests = sizeof(schedule_names) / sizeof(schedule_names[0]);
    int num_chunk_tests = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    
    const int NUM_RUNS_PER_CONFIG = 3;

    printf("\n--- Teste de Afinamento (Schedule e Chunk Size) ---\n");
    printf("Média de %d execuções por teste.\n\n", NUM_RUNS_PER_CONFIG);
    printf("+---------+-----------+--------------+------------------+\n");
    printf("| Threads | Schedule  | Chunk Size   | Tempo Médio (ms) |\n");
    printf("+---------+-----------+--------------+------------------+\n");

    for (int t = 0; t < num_thread_tests; t++) {
        int num_threads = threads_to_test[t];
        for (int s = 0; s < num_schedule_tests; s++) {
            for (int c = 0; c < num_chunk_tests; c++) {
                double total_ms = 0.0;
                for (int run = 0; run < NUM_RUNS_PER_CONFIG; run++) {
                    char command[512];
                    sprintf(command, "OMP_NUM_THREADS=%d %s %s %s 50 1e-6 %s %d",
                            num_threads, executable, data_file, centroids_file, schedule_names[s], chunk_sizes[c]);
                    FILE *pipe = popen(command, "r");
                    if (!pipe) { fprintf(stderr, "Erro ao executar o comando!\n"); exit(1); }
                    int threads_out; double ms_out, sse_out;
                    if (fscanf(pipe, "%d,%lf,%lf", &threads_out, &ms_out, &sse_out) != 3) {
                       fprintf(stderr, "Erro ao ler saída do schedule test.\n"); exit(1);
                    }
                    total_ms += ms_out;
                    pclose(pipe);
                }
                char chunk_str[20];
                if (chunk_sizes[c] == 0) sprintf(chunk_str, "default");
                else sprintf(chunk_str, "%d", chunk_sizes[c]);
                printf("| %-7d | %-9s | %-12s | %-16.1f |\n", num_threads, schedule_names[s], chunk_str, total_ms / NUM_RUNS_PER_CONFIG);
            }
        }
        if (t < num_thread_tests - 1) printf("+---------+-----------+--------------+------------------+\n");
    }
    printf("+---------+-----------+--------------+------------------+\n");
}