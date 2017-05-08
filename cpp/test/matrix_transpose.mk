app=matrix_transpose

SOURCES=matrix_transpose.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=
OTHER_OPT= -funroll-loops -march=native -mfpmath=sse #not working well
OPT= -O3 -ftree-vectorize

LIBS=-lboost_program_options

INCLUDES=-I../ -I./

CXXFLAGS= -std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
