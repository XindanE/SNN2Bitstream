// Driver for the Bambu inference AXI wrapper (BRAM-buffered, not DDR).
// PS writes input to the wrapper BRAM via AXI-Lite, starts inference, then reads the output registers.
//
// AXI-Lite address map (256KB):
//   0x00000  Control/Status  [W] bit 0 = ap_start; [R] bit 0 = ap_start, bit 1 = ap_done, bit 2 = ap_idle
//   0x10000  Input BRAM[0..N-1]   (float, 32-bit)
//   0x30000  Output[0..M-1]       (float, 32-bit)
//
// Copy this file as "xinference.h" into the Vitis SW project src/ and define BAMBU_BRAM_MODE before including.

#ifndef XINFERENCE_H
#define XINFERENCE_H

#include "xil_io.h"
#include "xil_types.h"
#include "xparameters.h"

// Bambu wrapper register/BRAM offsets
#define XINFERENCE_CTRL_REG             0x00
#define XINFERENCE_INPUT_BRAM_OFFSET    0x10000
#define XINFERENCE_OUTPUT_BRAM_OFFSET   0x30000

// Enable BRAM mode (can also be set via compiler -D flag)
#ifndef BAMBU_BRAM_MODE
#define BAMBU_BRAM_MODE 1
#endif

// Base address auto-detection from xparameters.h
#ifndef BAMBU_INFERENCE_BASEADDR
  #if defined(XPAR_BAMBU_WRAPPER_0_S_AXI_CONTROL_BASEADDR)
    #define BAMBU_INFERENCE_BASEADDR XPAR_BAMBU_WRAPPER_0_S_AXI_CONTROL_BASEADDR
  #elif defined(XPAR_BAMBU_INFERENCE_AXI_WRAPPER_0_S_AXI_CONTROL_BASEADDR)
    #define BAMBU_INFERENCE_BASEADDR XPAR_BAMBU_INFERENCE_AXI_WRAPPER_0_S_AXI_CONTROL_BASEADDR
  #elif defined(XPAR_BAMBU_WRAPPER_0_BASEADDR)
    #define BAMBU_INFERENCE_BASEADDR XPAR_BAMBU_WRAPPER_0_BASEADDR
  #elif defined(XPAR_BAMBU_INFERENCE_AXI_WRAPPER_0_BASEADDR)
    #define BAMBU_INFERENCE_BASEADDR XPAR_BAMBU_INFERENCE_AXI_WRAPPER_0_BASEADDR
  #else
    #error "Bambu wrapper base address not found in xparameters.h. " \
           "Define BAMBU_INFERENCE_BASEADDR before including this header."
  #endif
#endif

// HLS-compatible device ID
#ifndef XPAR_INFERENCE_0_DEVICE_ID
#define XPAR_INFERENCE_0_DEVICE_ID 0
#endif

// HLS-compatible type
typedef struct {
    u32 Control_BaseAddress;
    u32 IsReady;
} XInference;

// HLS-compatible API

static inline int XInference_Initialize(XInference *InstancePtr, u16 DeviceId) {
    (void)DeviceId;
    InstancePtr->Control_BaseAddress = BAMBU_INFERENCE_BASEADDR;
    InstancePtr->IsReady = 1;
    return 0;  // XST_SUCCESS
}

static inline void XInference_Start(XInference *InstancePtr) {
    Xil_Out32(InstancePtr->Control_BaseAddress + XINFERENCE_CTRL_REG, 0x1);
}

static inline u32 XInference_IsDone(XInference *InstancePtr) {
    u32 status = Xil_In32(InstancePtr->Control_BaseAddress + XINFERENCE_CTRL_REG);
    return (status >> 1) & 0x1;  // bit 1 = ap_done
}

static inline u32 XInference_IsIdle(XInference *InstancePtr) {
    u32 status = Xil_In32(InstancePtr->Control_BaseAddress + XINFERENCE_CTRL_REG);
    return (status >> 2) & 0x1;  // bit 2 = ap_idle
}

// In BRAM mode, Set_input/output_offset are no-ops (addresses are fixed in HW)
static inline void XInference_Set_input_offset(XInference *InstancePtr, u64 Data) {
    (void)InstancePtr; (void)Data;
}

static inline void XInference_Set_output_offset(XInference *InstancePtr, u64 Data) {
    (void)InstancePtr; (void)Data;
}

// BRAM data transfer functions

// Write float array to wrapper's input BRAM via AXI-Lite
static inline void XInference_WriteInputBRAM(XInference *InstancePtr,
                                              const float *data, int count) {
    u32 base = InstancePtr->Control_BaseAddress + XINFERENCE_INPUT_BRAM_OFFSET;
    const u32 *src = (const u32 *)data;
    for (int i = 0; i < count; i++) {
        Xil_Out32(base + (u32)i * 4, src[i]);
    }
}

// Read float array from wrapper's output registers via AXI-Lite
static inline void XInference_ReadOutputBRAM(XInference *InstancePtr,
                                              float *data, int count) {
    u32 base = InstancePtr->Control_BaseAddress + XINFERENCE_OUTPUT_BRAM_OFFSET;
    u32 *dst = (u32 *)data;
    for (int i = 0; i < count; i++) {
        dst[i] = Xil_In32(base + (u32)i * 4);
    }
}

#endif  // XINFERENCE_H
