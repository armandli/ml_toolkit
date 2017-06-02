#!/bin/sh

#nvcc -Wno-deprecated-gpu-targets -lcublas -lcurand -O3 -std=c++11 -o test_cublas test_cublas.cu
nvcc  -lcublas -lcurand -O3 -std=c++11 -o test_cublas test_cublas.cu
