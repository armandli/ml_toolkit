opencl_helloworld: opencl_helloworld.cpp
	g++ -std=c++11 -g -o opencl_helloworld opencl_helloworld.cpp -lOpenCL -I/opt/cuda/include
