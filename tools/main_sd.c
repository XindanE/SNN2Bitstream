// SNN inference on ZCU104 with SD-card data loading.

#include "xparameters.h"
#include "xinference.h"
// Vitis HLS >= 2022 appends "_r" to m_axi pointer-argument register names.
// run_vitis_app.tcl defines HLS_2022_PLUS so the call sites below stay version-independent.
#ifdef HLS_2022_PLUS
#define XInference_Set_input_offset  XInference_Set_input_r_offset
#define XInference_Set_output_offset XInference_Set_output_r_offset
#endif
#include "xil_printf.h"
#include "xuartps.h"
#include "xgpio.h"
#include "xtime_l.h"
#include "xil_cache.h"
#include <unistd.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// FatFs for SD card access
#include "ff.h"

// Auto-generated model config: dimensions, HLS_FIXED_MODE, and the interface type macros.
#include "model.h"

#define USE_PL              1  // 1: FPGA inference (PL), 0: ARM software inference (PS)
// USE_FIXED auto-detected from model.h: SQ/SPQ define HLS_FIXED_MODE, S/SP do not. Can be overridden.
#ifndef USE_FIXED
#ifdef HLS_FIXED_MODE
#define USE_FIXED           1  // Fixed-point interface (SQ/SPQ)
#else
#define USE_FIXED           0  // Float interface (S/SP)
#endif
#endif

// SD card data directory. Override in model.h via SD_DATA_DIR, or define manually.
#ifndef SD_DATA_DIR
#define SD_DATA_DIR "0:"
#endif
#define SD_FALL_FILE   SD_DATA_DIR "/fall.bin"
#define SD_LABEL_FILE  SD_DATA_DIR "/labels.bin"

// Number of test samples. model.h can define SD_NUM_SAMPLES to override (e.g., CIFAR10-DVS=1000).
#ifdef SD_NUM_SAMPLES
#define NUM_SAMPLES         SD_NUM_SAMPLES
#else
#define NUM_SAMPLES         10000
#endif
#define START_SAMPLE        0
#define END_SAMPLE          NUM_SAMPLES

// COUNTS_PER_SECOND defined by xtime_l.h (= XPAR_CPU_CORTEXA53_0_TIMESTAMP_CLK_FREQ)

// Float mode defaults: model.h only defines these macros in fixed-point (SQ/SPQ) configs
#ifndef INPUT_TOTAL_W
#define INPUT_TYPE_IS_INT    0
#define INPUT_TYPE_UNSIGNED  0
#define INPUT_TOTAL_W        32
#define INPUT_INT_W          32
#endif
#ifndef OUTPUT_TOTAL_W
#define OUTPUT_TYPE_IS_INT   0
#define OUTPUT_TYPE_UNSIGNED 0
#define OUTPUT_TOTAL_W       32
#define OUTPUT_INT_W         32
#endif

#ifndef INPUT_ENCODING_REPEAT
#define INPUT_ENCODING_REPEAT 0
#endif
#if INPUT_ENCODING_REPEAT
#define ENCODED_SIZE        INPUT_DIM  // repeat: D values (HLS reuses per timestep)
#else
#define ENCODED_SIZE        (INPUT_DIM * TIMESTEPS)  // rate/temporal: T*D values
#endif
#define ENCODED_BYTES       (ENCODED_SIZE * sizeof(float))  // SD card data is always float32
#define OUTPUT_SIZE         OUTPUT_DIM
#define INPUT_FRAC          (INPUT_TOTAL_W - INPUT_INT_W)
#define INPUT_SCALE         (1 << INPUT_FRAC)
#define OUTPUT_FRAC         (OUTPUT_TOTAL_W - OUTPUT_INT_W)
#define OUTPUT_SCALE        (1 << OUTPUT_FRAC)

// Hardware device IDs
#define UART_DEVICE_ID      XPAR_XUARTPS_0_DEVICE_ID
#define GPIO_DEVICE_ID      XPAR_GPIO_0_DEVICE_ID
#define LED_CHANNEL         1

// Global variables
static FATFS fatfs;
static FIL fil;
static XGpio Gpio;
static XUartPs Uart_Ps;

// SD card read buffer (always float32, since fall.bin stores float32)
static float sd_read_buf[ENCODED_SIZE] __attribute__((aligned(64)));

#if USE_FIXED && USE_PL
// Fixed-point mode: AXI DMA buffer type selected by bit width
#if INPUT_TOTAL_W <= 8
typedef int8_t axi_input_t;
#elif INPUT_TOTAL_W <= 16
typedef int16_t axi_input_t;
#else
typedef int32_t axi_input_t;
#endif
#if OUTPUT_TOTAL_W <= 8
typedef int8_t axi_output_t;
#elif OUTPUT_TOTAL_W <= 16
typedef int16_t axi_output_t;
#else
typedef int32_t axi_output_t;
#endif
static axi_input_t input_buf[ENCODED_SIZE] __attribute__((aligned(64)));
static axi_output_t output_buf[OUTPUT_SIZE] __attribute__((aligned(64)));
#elif !USE_PL
// PS software inference: use model.h input_t/output_t (may be uint8_t or float)
static input_t ps_input_buf[ENCODED_SIZE] __attribute__((aligned(64)));
static output_t ps_output_buf[OUTPUT_SIZE] __attribute__((aligned(64)));
#else
// Float PL mode: DMA buffers use float
static float input_buf[ENCODED_SIZE] __attribute__((aligned(64)));
static float output_buf[OUTPUT_SIZE] __attribute__((aligned(64)));
#endif

static int labels[NUM_SAMPLES];

// Helper functions

void print_float(float f) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.4f", f);
    xil_printf("%s", buffer);
}

int init_uart() {
    XUartPs_Config *Config = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (Config == NULL) return XST_FAILURE;

    int Status = XUartPs_CfgInitialize(&Uart_Ps, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) return XST_FAILURE;

    XUartPs_SetBaudRate(&Uart_Ps, 115200);
    return XST_SUCCESS;
}

int init_gpio() {
    int Status = XGpio_Initialize(&Gpio, GPIO_DEVICE_ID);
    if (Status != XST_SUCCESS) {
        xil_printf("GPIO init failed!\r\n");
        return XST_FAILURE;
    }
    XGpio_SetDataDirection(&Gpio, LED_CHANNEL, 0x0);
    return XST_SUCCESS;
}

void show_class_on_led(int cls) {
    XGpio_DiscreteWrite(&Gpio, LED_CHANNEL, (u32)(cls & 0xF));
}

int argmax_float(float *arr, int n) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

#if USE_FIXED && USE_PL
int argmax_fixed(axi_output_t *arr, int n) {
    int max_idx = 0;
    axi_output_t max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}
#endif

// Fixed-point conversion

#if USE_FIXED && USE_PL
// float to HLS input binary representation: integer types round to nearest, fixed-point types scale by 2^FRAC.
static inline axi_input_t float_to_fixed_input(float val) {
    int32_t r;
#if INPUT_TYPE_IS_INT
    r = (val >= 0) ? (int32_t)(val + 0.5f) : (int32_t)(val - 0.5f);
#else
    float scaled = val * INPUT_SCALE;
    r = (scaled >= 0) ? (int32_t)(scaled + 0.5f) : (int32_t)(scaled - 0.5f);
#endif
    // Saturation clamp
#if INPUT_TYPE_UNSIGNED
    if (r < 0) r = 0;
    if (r > ((1 << INPUT_TOTAL_W) - 1)) r = (1 << INPUT_TOTAL_W) - 1;
#else
    int32_t max_val = (1 << (INPUT_TOTAL_W - 1)) - 1;
    int32_t min_val = -(1 << (INPUT_TOTAL_W - 1));
    if (r > max_val) r = max_val;
    if (r < min_val) r = min_val;
#endif
    return (axi_input_t)r;
}

// HLS output binary representation back to float.
static inline float fixed_output_to_float(axi_output_t val) {
#if OUTPUT_TYPE_IS_INT
    return (float)val;
#else
    return (float)val / (float)OUTPUT_SCALE;
#endif
}
#endif

// SD card operations

int mount_sd() {
    FRESULT res = f_mount(&fatfs, "0:/", 1);
    if (res != FR_OK) {
        xil_printf("ERROR: SD card mount failed (res=%d)\r\n", res);
        return XST_FAILURE;
    }
    xil_printf("SD card mounted successfully.\r\n");

    return XST_SUCCESS;
}

int load_labels(const char *path, int *labels_out, int num_samples) {
    FRESULT res;
    UINT bytes_read;

    res = f_open(&fil, path, FA_READ);
    if (res != FR_OK) {
        xil_printf("ERROR: Cannot open %s (res=%d)\r\n", path, res);
        return XST_FAILURE;
    }

    res = f_read(&fil, labels_out, num_samples * sizeof(int), &bytes_read);
    f_close(&fil);

    if (res != FR_OK || bytes_read != num_samples * sizeof(int)) {
        xil_printf("ERROR: Failed to read labels (read %u bytes)\r\n", bytes_read);
        return XST_FAILURE;
    }

    xil_printf("Loaded %d labels from %s\r\n", num_samples, path);
    return XST_SUCCESS;
}

// fall.bin file handle - opened once in main, seeked per sample
static FIL fall_fil;

int load_sample(int idx) {
    FRESULT res;
    UINT bytes_read;

    res = f_lseek(&fall_fil, (FSIZE_t)idx * ENCODED_BYTES);
    if (res != FR_OK) {
        xil_printf("ERROR: Seek to sample %d failed (res=%d)\r\n", idx, res);
        return XST_FAILURE;
    }
    res = f_read(&fall_fil, sd_read_buf, ENCODED_BYTES, &bytes_read);

    if (res != FR_OK || bytes_read != ENCODED_BYTES) {
        xil_printf("ERROR: Failed to read sample %d (read %u bytes)\r\n", idx, bytes_read);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

// Main

int main() {
    int correct = 0;
    int tested = 0;
    XTime t_start, t_end;
    u64 total_ticks = 0;

    xil_printf("\r\n");
    xil_printf("  SNN Inference (SD: %s)\r\n", SD_DATA_DIR);
    xil_printf("  Sample range: %d - %d (%d samples)\r\n", START_SAMPLE, END_SAMPLE, END_SAMPLE - START_SAMPLE);
    xil_printf("  Input size:   %d values (%d bytes on SD)\r\n", ENCODED_SIZE, ENCODED_BYTES);
    xil_printf("  Mode:         %s\r\n", USE_PL ? "FPGA (PL)" : "ARM (PS)");
#if USE_FIXED && USE_PL
    xil_printf("  Interface:    Fixed-point (input: %dW/%dI, output: %dW/%dI)\r\n",
               INPUT_TOTAL_W, INPUT_INT_W, OUTPUT_TOTAL_W, OUTPUT_INT_W);
#else
    xil_printf("  Interface:    Float\r\n");
#endif
    xil_printf("==============================================\r\n\r\n");

    // Initialize hardware
    if (init_uart() != XST_SUCCESS) {
        xil_printf("UART init failed!\r\n");
        return XST_FAILURE;
    }

    if (init_gpio() != XST_SUCCESS) {
        return XST_FAILURE;
    }

    // Mount SD card
    if (mount_sd() != XST_SUCCESS) {
        return XST_FAILURE;
    }

    // Load all labels into memory
    if (load_labels(SD_LABEL_FILE, labels, NUM_SAMPLES) != XST_SUCCESS) {
        return XST_FAILURE;
    }

#if USE_PL
    // Initialize HLS inference IP
    XInference inference_ip;
    int status = XInference_Initialize(&inference_ip, XPAR_INFERENCE_0_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("ERROR: Inference IP init failed!\r\n");
        return XST_FAILURE;
    }
    xil_printf("Inference IP initialized.\r\n\r\n");
#endif

    // Open fall.bin once; use f_lseek per sample to avoid FAT32 per-file overhead
    {
        FRESULT res = f_open(&fall_fil, SD_FALL_FILE, FA_READ);
        if (res != FR_OK) {
            xil_printf("ERROR: Cannot open %s (res=%d)\r\n", SD_FALL_FILE, res);
            return XST_FAILURE;
        }
        xil_printf("Opened: %s\r\n", SD_FALL_FILE);
    }

    xil_printf("Starting inference from sample %d to %d...\r\n\r\n", START_SAMPLE, END_SAMPLE - 1);

    // Print a cumulative-accuracy checkpoint ~10 times across the run; interval scales with test-set size.
    int print_every = (END_SAMPLE - START_SAMPLE) / 10;
    if (print_every < 1) print_every = 1;

    // Main inference loop
    for (int i = START_SAMPLE; i < END_SAMPLE; i++) {
        // Load one sample from SD card (float32)
        if (load_sample(i) != XST_SUCCESS) {
            xil_printf("Skipping sample %d due to load error\r\n", i);
            continue;
        }

#if USE_PL
        memset(output_buf, 0, sizeof(output_buf));
#else
        memset(ps_output_buf, 0, sizeof(ps_output_buf));
#endif

#if USE_PL
  #if USE_FIXED
        // Fixed-point FPGA inference: convert float -> int before passing to HLS IP
        for (int j = 0; j < ENCODED_SIZE; j++) {
            input_buf[j] = float_to_fixed_input(sd_read_buf[j]);
        }

        Xil_DCacheFlushRange((UINTPTR)input_buf, ENCODED_SIZE * sizeof(axi_input_t));
        Xil_DCacheFlushRange((UINTPTR)output_buf, OUTPUT_SIZE * sizeof(axi_output_t));

        XInference_Set_input_offset(&inference_ip, (u64)input_buf);
        XInference_Set_output_offset(&inference_ip, (u64)output_buf);

        XTime_GetTime(&t_start);
        XInference_Start(&inference_ip);
        while (!XInference_IsDone(&inference_ip));
        XTime_GetTime(&t_end);

        Xil_DCacheInvalidateRange((UINTPTR)output_buf, OUTPUT_SIZE * sizeof(axi_output_t));
  #else
    #if defined(BAMBU_BRAM_MODE) && BAMBU_BRAM_MODE
        // Bambu BRAM mode: write data directly into wrapper BRAM, no DDR/cache needed
        XInference_WriteInputBRAM(&inference_ip, sd_read_buf, ENCODED_SIZE);

        XTime_GetTime(&t_start);
        XInference_Start(&inference_ip);
        while (!XInference_IsDone(&inference_ip));
        XTime_GetTime(&t_end);

        XInference_ReadOutputBRAM(&inference_ip, output_buf, OUTPUT_SIZE);
    #else
        // Float FPGA inference: pass float directly to HLS IP
        memcpy(input_buf, sd_read_buf, ENCODED_BYTES);

        Xil_DCacheFlushRange((UINTPTR)input_buf, ENCODED_BYTES);
        Xil_DCacheFlushRange((UINTPTR)output_buf, OUTPUT_SIZE * sizeof(float));

        XInference_Set_input_offset(&inference_ip, (u64)input_buf);
        XInference_Set_output_offset(&inference_ip, (u64)output_buf);

        XTime_GetTime(&t_start);
        XInference_Start(&inference_ip);
        while (!XInference_IsDone(&inference_ip));
        XTime_GetTime(&t_end);

        Xil_DCacheInvalidateRange((UINTPTR)output_buf, OUTPUT_SIZE * sizeof(float));
    #endif
  #endif
#else
        // ARM software inference: convert float -> input_t before calling inference()
        for (int j = 0; j < ENCODED_SIZE; j++) {
            float val = sd_read_buf[j];
#if INPUT_TYPE_IS_INT
            int32_t r = (val >= 0) ? (int32_t)(val + 0.5f) : (int32_t)(val - 0.5f);
#else
            int32_t r = (int32_t)(val);
#endif
            ps_input_buf[j] = (input_t)r;
        }

        XTime_GetTime(&t_start);
        inference(ps_input_buf, ps_output_buf);
        XTime_GetTime(&t_end);
#endif

        total_ticks += (t_end - t_start);

        // Get prediction
        int pred;
#if USE_FIXED && USE_PL
        pred = argmax_fixed(output_buf, OUTPUT_SIZE);
#elif !USE_PL
        {
            int max_idx = 0;
            output_t max_val = ps_output_buf[0];
            for (int k = 1; k < OUTPUT_SIZE; k++) {
                if (ps_output_buf[k] > max_val) {
                    max_val = ps_output_buf[k];
                    max_idx = k;
                }
            }
            pred = max_idx;
        }
#else
        pred = argmax_float(output_buf, OUTPUT_SIZE);
#endif
        int label = labels[i];

        if (pred == label) correct++;
        tested++;

        //show_class_on_led(pred);

        // Per-sample line: just whether this sample was correct.
        xil_printf("[%d] predict=%d label=%d %s\r\n",
                   i, pred, label, (pred == label) ? "CORRECT" : "WRONG");

        // ~10 cumulative-accuracy checkpoints interspersed across the run.
        if (tested % print_every == 0) {
            float acc = 100.0f * correct / tested;
            xil_printf("[%d/%d] acc=", tested, END_SAMPLE - START_SAMPLE);
            print_float(acc);
            xil_printf("%%\r\n");
        }
    }

    f_close(&fall_fil);

    // Final result: one line, avg latency printed once here.
    float final_acc = 100.0f * correct / tested;
    double avg_ms = (double)total_ticks / COUNTS_PER_SECOND * 1000.0 / tested;

    xil_printf("\r\n  [RESULT] %d/%d correct  acc=", correct, tested);
    print_float(final_acc);
    xil_printf("%%  avg_latency=");
    print_float(avg_ms);
    xil_printf("ms\r\n");

    return XST_SUCCESS;
}
