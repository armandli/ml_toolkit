#!/bin/sh

nvcc -lcublas -lcurand -O3 -std=c++11 -o test_cublas test-cublas.cpp
