// MNIST SNN runner: Poisson rate-encoding, 10K test set, float or fixed-point interface.
//
// Compile: g++ -std=c++11 -O2 model.cpp fc_layer*.cpp lif_layer*.cpp conv_layer*.cpp pool_layer*.cpp reset_node.cpp main_mnist.c -lm -o test
// Run: ./test
//
// Per-sample results go to sw_test.log (or $SNN2B_TEST_LOG) in the board's UART format.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "model.h"

#ifndef INPUT_DIM_FLAT
#define INPUT_DIM_FLAT 784
#endif

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 10
#endif

#define IMAGE_SIZE 784
// MNIST raw IDX paths; argv[1] overrides the dir at runtime.
// Default reaches repo-root data/ from projects/<name>/cpp/ (three levels up).
static char IMAGE_FILE[1024] = "../../../data/MNIST/raw/t10k-images-idx3-ubyte";
static char LABEL_FILE[1024] = "../../../data/MNIST/raw/t10k-labels-idx1-ubyte";

static void load_mnist_image(float input[IMAGE_SIZE], int index) {
    FILE* f = fopen(IMAGE_FILE, "rb");
    if (!f) {
        perror("Failed to open MNIST image file");
        exit(EXIT_FAILURE);
    }
    fseek(f, 16 + index * IMAGE_SIZE, SEEK_SET);
    uint8_t buf[IMAGE_SIZE];
    size_t n = fread(buf, 1, IMAGE_SIZE, f);
    fclose(f);
    if (n != IMAGE_SIZE) {
        fprintf(stderr, "Failed to read MNIST image %d\n", index);
        exit(EXIT_FAILURE);
    }
#if INPUT_ENCODING_REPEAT
    for (int i = 0; i < IMAGE_SIZE; i++) {
        input[i] = (buf[i] / 255.0f - 0.1307f) / 0.3081f;
    }
#else
    for (int i = 0; i < IMAGE_SIZE; i++) {
        input[i] = buf[i] / 255.0f;
    }
#endif
}

static uint8_t load_mnist_label(int index) {
    FILE* f = fopen(LABEL_FILE, "rb");
    if (!f) {
        perror("Failed to open MNIST label file");
        exit(EXIT_FAILURE);
    }
    fseek(f, 8 + index, SEEK_SET);
    uint8_t label;
    size_t n = fread(&label, 1, 1, f);
    fclose(f);
    if (n != 1) {
        fprintf(stderr, "Failed to read MNIST label %d\n", index);
        exit(EXIT_FAILURE);
    }
    return label;
}

// Repeat encoding: replicate normalized image across T timesteps (matches training)
static void repeat_encode(
    const float *input, 
    float *output,  // length = TIMESTEPS * INPUT_DIM_FLAT
    int timesteps
) {
    for (int t = 0; t < timesteps; ++t) {
        float *frame = output + (size_t)t * INPUT_DIM_FLAT;
        for (int j = 0; j < INPUT_DIM_FLAT; ++j) {
            frame[j] = input[j];
        }
    }
}

static void rate_encode(
    const float *input, 
    float *output,  // length = TIMESTEPS * INPUT_DIM_FLAT
    int timesteps
) {
    for (int t = 0; t < timesteps; ++t) {
        float *frame = output + (size_t)t * INPUT_DIM_FLAT;
        for (int j = 0; j < INPUT_DIM_FLAT; ++j) {
            float p = input[j];
            frame[j] = ((float)rand() / (float)RAND_MAX < p) ? 1.0f : 0.0f;
        }
    }
}

// argmax for output_t array
static int argmax_out(const output_t *x, int n) {
    output_t max_val = x[0];
    int max_idx = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        snprintf(IMAGE_FILE, sizeof(IMAGE_FILE), "%s/raw/t10k-images-idx3-ubyte", argv[1]);
        snprintf(LABEL_FILE, sizeof(LABEL_FILE), "%s/raw/t10k-labels-idx1-ubyte", argv[1]);
    }
    srand(42);  // Fixed seed for reproducible rate coding
    int correct = 0;
    int total = 10000;

    // single-frame image (always float for reading raw MNIST)
    float image[IMAGE_SIZE];
    output_t output[OUTPUT_DIM];

    const char *log_env = getenv("SNN2B_TEST_LOG");
    const char *log_path = log_env ? log_env : "sw_test.log";
    FILE *log_file = NULL;
    if (log_path[0] != '\0') {
        log_file = fopen(log_path, "w");
        if (!log_file) {
            perror(log_path);
            return 1;
        }
    }
    int on_terminal = isatty(fileno(stdout));
    int print_every = total / 10;

#if INPUT_ENCODING_REPEAT
    // Repeat encoding
    const size_t INPUT_ELEMS = INPUT_DIM_FLAT;
#else
    const size_t INPUT_ELEMS = (size_t)TIMESTEPS * INPUT_DIM_FLAT;
#endif

    float *encoded_float = (float *)malloc(INPUT_ELEMS * sizeof(float));
    input_t *input_conv = (input_t *)malloc(INPUT_ELEMS * sizeof(input_t));
    if (!encoded_float || !input_conv) {
        perror("malloc failed");
        return 1;
    }

    printf("Running MNIST SNN C test (%d samples)...\n", total);
    printf("  TIMESTEPS=%d, INPUT_DIM_FLAT=%d, ENCODING=%s\n",
           (int)TIMESTEPS, (int)INPUT_DIM_FLAT,
           INPUT_ENCODING_REPEAT ? "repeat" : "rate");

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    double last_draw_ms = -1e9;

    for (int image_index = 0; image_index < total; image_index++) {
        load_mnist_image(image, image_index);
        uint8_t true_label = load_mnist_label(image_index);

#if INPUT_ENCODING_REPEAT
        for (int j = 0; j < INPUT_DIM_FLAT; ++j) {
            encoded_float[j] = image[j];
        }
#else
        // Rate encoding: generate T*D Poisson spikes
        rate_encode(image, encoded_float, TIMESTEPS);
#endif

        // Convert float -> input_t
        for (size_t j = 0; j < INPUT_ELEMS; ++j) {
#if INPUT_TYPE_IS_INT
            input_conv[j] = (input_t)roundf(encoded_float[j]);
#else
            input_conv[j] = (input_t)(encoded_float[j]);
#endif
        }

        inference(input_conv, output);

        int pred = argmax_out(output, OUTPUT_DIM);
        if (pred == true_label) {
            correct++;
        }

        int done = image_index + 1;
        double acc = 100.0 * correct / done;

        if (log_file) {
            fprintf(log_file, "[%d] predict=%d label=%d %s\n",
                    image_index, pred, true_label,
                    pred == true_label ? "CORRECT" : "WRONG");
            if (done % print_every == 0)
                fprintf(log_file, "[%d/%d] acc=%.4f%%\n", done, total, acc);
        }

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed_ms = (t_now.tv_sec - t_start.tv_sec) * 1000.0
                          + (t_now.tv_nsec - t_start.tv_nsec) / 1e6;
        if (on_terminal) {
            if (elapsed_ms - last_draw_ms >= 100.0 || done == total) {
                double avg_ms = elapsed_ms / done;
                int remaining_s = (int)(avg_ms * (total - done) / 1000.0);
                printf("\r[%d/%d] acc=%.2f%%  avg=%.3fms  ETA %dm%02ds   ",
                       done, total, acc, avg_ms, remaining_s / 60, remaining_s % 60);
                fflush(stdout);
                last_draw_ms = elapsed_ms;
            }
        } else if (done % print_every == 0) {
            printf("[%d/%d] acc=%.4f%%\n", done, total, acc);
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double total_ms = (t_now.tv_sec - t_start.tv_sec) * 1000.0
                    + (t_now.tv_nsec - t_start.tv_nsec) / 1e6;
    if (on_terminal) printf("\n");

    double acc = 100.0 * correct / total;
    printf("\n  [RESULT] %d/%d correct  acc=%.4f%%  avg_latency=%.4fms\n",
           correct, total, acc, total_ms / total);
    if (log_file) {
        fprintf(log_file, "\n  [RESULT] %d/%d correct  acc=%.4f%%  avg_latency=%.4fms\n",
                correct, total, acc, total_ms / total);
        fclose(log_file);
    }
    free(encoded_float);
    free(input_conv);
    return 0;
}
