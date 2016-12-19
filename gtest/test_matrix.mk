app=test_matrix

SOURCES=test_matrix.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=
OTHER_OPT= -funroll-loops -march=native -mfpmath=sse #not working well
OPT= -O3 -ftree-vectorize

LIBS=-lboost_program_options -lgtest -lgtest_main

INCLUDES=-I../ -I./

CXXFLAGS= -std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
