// Full-dataset GCC inference test: read fall.bin + labels.bin, run every sample.
// For temporal datasets (N-MNIST, CIFAR10-DVS, DVS Gesture); MNIST uses main_mnist.c (raw IDX).
//
// fall.bin   : N * ENCODED_SIZE float32, row-major (one sample per row)
// labels.bin : N int32 labels (N is taken from the file size)
//
// Usage: ./test_full [data_dir]   (directory holding fall.bin + labels.bin, default ".")

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "model.h"

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 10
#endif

#ifndef INPUT_ENCODING_REPEAT
#define INPUT_ENCODING_REPEAT 0
#endif

// For repeat encoding the model receives D values and replicates internally.
// For temporal/rate encoding the model receives T*D values.
#if INPUT_ENCODING_REPEAT
#define ENCODED_SIZE ((size_t)INPUT_DIM)
#else
#define ENCODED_SIZE ((size_t)TIMESTEPS * INPUT_DIM)
#endif

static int argmax(const output_t *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

int main(int argc, char *argv[]) {
    const char *data_dir = (argc > 1) ? argv[1] : ".";
    // Optional argv[2] caps the sample count - lets a fixed-point (ap_fixed) sweep
    // estimate accuracy on a subset in minutes instead of running the full set.
    int max_samples = (argc > 2) ? atoi(argv[2]) : 0;
    // Optional argv[3] = start offset into the seeded-shuffled sample order. Enables resume:
    // "2000 0" runs the first 2000, "8000 2000" runs the next 8000; same seed -> same shuffle,
    // so the two slices are disjoint and combine into the full set with no overlap.
    int offset = (argc > 3) ? atoi(argv[3]) : 0;

    char labels_path[512], fall_path[512];
    snprintf(labels_path, sizeof(labels_path), "%s/labels.bin", data_dir);
    snprintf(fall_path,   sizeof(fall_path),   "%s/fall.bin",   data_dir);

    // Determine N from labels.bin size
    FILE *lf = fopen(labels_path, "rb");
    if (!lf) { perror(labels_path); return 1; }
    fseek(lf, 0, SEEK_END);
    long lsize = ftell(lf);
    rewind(lf);
    int total = (int)(lsize / sizeof(int32_t));
    if (total <= 0) { fprintf(stderr, "labels.bin is empty\n"); return 1; }
    // When capped, evaluate a REPRODUCIBLE RANDOM subset - not the first n samples. The
    // test set is grouped by label (the first ~1000 are all '0'), so a prefix is wildly
    // unrepresentative. A fixed seed gives every arch the same subset, so runs compare fairly.
    int cnt = (max_samples > 0 && max_samples < total) ? max_samples : total;
    int start = offset;   if (start < 0) start = 0;   if (start > total) start = total;
    int end = start + cnt; if (end > total) end = total;
    int n = end - start;   // samples actually evaluated this run

    int32_t *labels = (int32_t *)malloc(total * sizeof(int32_t));
    if (!labels) { fprintf(stderr, "malloc failed\n"); return 1; }
    if ((int)fread(labels, sizeof(int32_t), total, lf) != total) {
        fprintf(stderr, "labels.bin read error\n"); return 1;
    }
    fclose(lf);

    int *idx = (int *)malloc(total * sizeof(int));
    if (!idx) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < total; i++) idx[i] = i;
    if (n < total || start > 0) {              // full Fisher-Yates so any [start,end) slice is a fair, resumable subset
        srand(12345);                          // first n iterations match the old partial shuffle, so [0,n) is unchanged
        for (int i = 0; i < total - 1; i++) {
            int j = i + rand() % (total - i);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
    }

    FILE *df = fopen(fall_path, "rb");
    if (!df) { perror(fall_path); return 1; }

    float    *sample_f  = (float    *)malloc(ENCODED_SIZE * sizeof(float));
    input_t  *sample_in = (input_t  *)malloc(ENCODED_SIZE * sizeof(input_t));
    output_t  output[OUTPUT_DIM];
    if (!sample_f || !sample_in) { fprintf(stderr, "malloc failed\n"); return 1; }

    printf("GCC inference test: %d samples, TIMESTEPS=%d, INPUT_DIM=%d, data_dir=%s\n",
           n, (int)TIMESTEPS, (int)INPUT_DIM, data_dir);

    int correct = 0;
    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for (int i = start; i < end; i++) {
        long off = (long)idx[i] * (long)ENCODED_SIZE * (long)sizeof(float);
        if (fseek(df, off, SEEK_SET) != 0 ||
            fread(sample_f, sizeof(float), ENCODED_SIZE, df) != ENCODED_SIZE) {
            fprintf(stderr, "fall.bin read error at sample %d (idx %d)\n", i, idx[i]);
            break;
        }

        for (size_t j = 0; j < ENCODED_SIZE; j++) {
#if INPUT_TYPE_IS_INT
            sample_in[j] = (input_t)roundf(sample_f[j]);
#else
            sample_in[j] = (input_t)sample_f[j];
#endif
        }

        inference(sample_in, output);

        if (argmax(output, OUTPUT_DIM) == (int)labels[idx[i]])
            correct++;

        // Match FPGA board log format: [N/total] correct=X acc=Y%  avg=Zms
        int done = i - start + 1;
        if (done == 1 || done % 100 == 0 || done == n) {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed_ms = (t_now.tv_sec - t_start.tv_sec) * 1000.0
                              + (t_now.tv_nsec - t_start.tv_nsec) / 1e6;
            double avg_ms = elapsed_ms / done;
            printf("[%d/%d] correct=%d acc=%.4f%%  avg=%.4fms\n",
                   done, n, correct,
                   100.0f * correct / done,
                   avg_ms);
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double total_ms = (t_now.tv_sec - t_start.tv_sec) * 1000.0
                    + (t_now.tv_nsec - t_start.tv_nsec) / 1e6;
    printf("\n  [RESULT] %d/%d correct  acc=%.4f%%  avg_latency=%.4fms\n",
           correct, n, 100.0f * correct / n, total_ms / n);

    fclose(df);
    free(labels);
    free(sample_f);
    free(sample_in);
    return 0;
}
