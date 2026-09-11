// MNIST SNN runner: Poisson rate-encoding, 10K test set, float or fixed-point interface.
//
// Compile: g++ -std=c++11 -O2 model.cpp fc_layer*.cpp lif_layer*.cpp conv_layer*.cpp pool_layer*.cpp reset_node.cpp main_mnist.c -lm -o test
// Run: ./test

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <time.h>
#include <math.h>
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

    FILE *log_file = fopen("test.log", "w");
    if (!log_file) {
        perror("Failed to open log file");
        return 1;
    }

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
    printf("  TIMESTEPS=%d, INPUT_DIM_FLAT=%d, ENCODING=%s\n\n",
           (int)TIMESTEPS, (int)INPUT_DIM_FLAT,
           INPUT_ENCODING_REPEAT ? "repeat" : "rate");

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

        fprintf(log_file, "Image %d: pred=%d, true=%d, %s\n",
                image_index, pred, true_label,
                pred == true_label ? "correct" : "wrong");

        if ((image_index + 1) % 500 == 0 || image_index + 1 == total) {
            float acc = 100.0f * correct / (image_index + 1);
            printf("  [%5d/%5d] acc=%.2f%%\r", image_index + 1, total, acc);
            fflush(stdout);
        }
    }
    printf("\n");

    float acc = 100.0f * correct / total;
    printf("[RESULT] Accuracy: %.2f%% (%d/%d)\n", acc, correct, total);

    fprintf(log_file, "SUMMARY: Accuracy = %.2f%% (%d/%d)\n", acc, correct, total);
    fclose(log_file);
    free(encoded_float);
    free(input_conv);
    return 0;
}
