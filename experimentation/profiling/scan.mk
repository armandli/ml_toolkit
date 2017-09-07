app=scan

SOURCES=scan.cu

OBJECTS=$(SOURCES:.cu=.o)

all: $(app)

DEBUG=
OTHER_OPT= -funroll-loops #not working well
OPT= -O3 --compiler-options -ftree-vectorize --compiler-options -march=native --compiler-options -mfpmath=sse

LIBS=-lopenblas -lpthread

INCLUDES= -I./ -I/user/include

CXXFLAGS=-std=c++11 $(INCLUDES) $(LIBS) $(OPT) $(DEBUG) -Wno-deprecated-gpu-targets

COMPILER=nvcc

$(app): %: %.cu
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
