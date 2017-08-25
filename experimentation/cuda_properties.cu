#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Version %d.%d\n", prop.major, prop.minor);
    printf("  Compute Mode: %d\n", prop.computeMode);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Multi Processor Count: %d\n\n", prop.multiProcessorCount);
    printf("  TCC Driver: %d\n\n", prop.tccDriver);
    printf("  Total Global Mem: %d\n\n", prop.totalGlobalMem);
    printf("  Shared Mem Per Block: %d\n\n", prop.sharedMemPerBlock);
    printf("  Registers Per Block: %d\n\n", prop.regsPerBlock);
    printf("  Warpsize: %d\n\n", prop.warpSize);
    printf("  MemPitch: %d\n\n", prop.memPitch);
    printf("  MaxThreadsPerBlock: %d\n\n", prop.maxThreadsPerBlock);
    printf("  Can Map Host Memory: %d\n\n", prop.canMapHostMemory);
  }
}
